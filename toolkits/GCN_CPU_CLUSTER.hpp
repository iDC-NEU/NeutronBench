#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"

class GCN_CPU_CLUSTER_impl {
 public:
  int iterations;
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType drop_rate;
  ValueType alpha;
  ValueType beta1;
  ValueType beta2;
  ValueType epsilon;
  ValueType decay_rate;
  ValueType decay_epoch;
  NtsVar local_mask;
  float train_correct;
  float train_num;
  float val_correct;
  float val_num;
  float test_correct;
  float test_num;
  // graph
  VertexSubset* active;
  // graph with no edge data
  Graph<Empty>* graph;
  // std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum* gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar global_label;
  NtsVar global_mask;
  NtsVar MASK;
  // GraphOperation *gt;
  PartitionedGraph* partitioned_graph;
  // Variables
  std::vector<Parameter*> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  // Sampler* train_sampler;
  // Sampler* val_sampler;
  // Sampler* test_sampler;
  FullyRepGraph* fully_rep_graph;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  int max_batch_num;
  int min_batch_num;
  torch::nn::Dropout drpmodel;
  ntsPeerRPC<ValueType, VertexId> rpc;

  GCN_CPU_CLUSTER_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
                       bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    graph->init_gnnctx_fanout(graph->config->fanout_string);
    // rtminfo initialize
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
  }
  void init_graph() {
    fully_rep_graph = new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");
    // sampler=new Sampler(fully_rep_graph,0,graph->vertices);

    // cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx = new nts::ctx::NtsContext();
  }
  void init_nn() {
    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    gnndatum = new GNNDatum(graph->gnnctx, graph);
    // gnndatum->random_generate();
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
    }

    // creating tensor to save Label and Mask
    gnndatum->registLabel(L_GT_C);
    gnndatum->registGlobalLabel(global_label);
    gnndatum->registMask(MASK);
    gnndatum->registGlobalMask(global_mask);
    // if (graph->partition_id == 0) {
    //  std::cout << global_label << " " << global_mask << std::endl;
    // }

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
    }

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
    }
    drpmodel = torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);

    // X[i] is vertex representation at layer i
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }

    X[0] = F.set_requires_grad(true);

    rpc.set_comm_num(graph->partitions - 1);
    rpc.register_function("get_feature", [&](std::vector<VertexId> vertexs) {
      int start = graph->partition_offset[graph->partition_id];
      int feature_size = F.size(1);
      ValueType* ntsVarBuffer = graph->Nts->getWritableBuffer(F, torch::DeviceType::CPU);
      std::vector<std::vector<ValueType>> result_vector;
      result_vector.resize(vertexs.size());

#pragma omp parallel for
      for (int i = 0; i < vertexs.size(); i++) {
        result_vector[i].resize(feature_size);
        memcpy(result_vector[i].data(), ntsVarBuffer + (vertexs[i] - start) * feature_size,
               feature_size * sizeof(ValueType));
      }
      return result_vector;
    });
  }

  long getCorrect(NtsVar input, NtsVar target, int type = 0) {
    // NtsVar predict = input.log_softmax(1).argmax(1);
    // NtsVar mask = (local_mask == type);
    // left = left.masked_select(mask.unsqueeze(1).expand({left.size(0), left.size(1)})).view({-1, left.size(1)});
    // right = right.masked_select(mask.view({mask.size(0)}));

    // std::cout << "before " << input.size(0) << std::endl;
    // input = input.masked_select(mask.view({mask.size(0)}));
    NtsVar mask = (local_mask == type);
    input = input.masked_select(mask.unsqueeze(1).expand({-1, input.size(1)})).view({-1, input.size(1)});
    // std::cout << "after " << input.size(0) << std::endl;
    target = target.masked_select(mask.view({mask.size(0)}));
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    if (type == 0) {
      train_correct += output.sum().item<long>();
      train_num += output.size(0);
    } else if (type == 1) {
      val_correct += output.sum().item<long>();
      val_num += output.size(0);
    } else {
      test_correct += output.sum().item<long>();
      test_num += output.size(0);
    }
    return output.sum(0).item<long>();
  }

  void Test(long s, NtsVar& target, NtsVar& mask) {  // 0 train, //1 eval //2 test
    NtsVar mask_train = mask.eq(s);
    NtsVar all_train = X[graph->gnnctx->layer_size.size() - 1]
                           .argmax(1)
                           .to(torch::kLong)
                           .eq(target)
                           .to(torch::kLong)
                           .masked_select(mask_train.view({mask_train.size(0)}));
    long p_correct = all_train.sum(0).item<long>();
    long p_train = all_train.size(0);
    float acc_train = 1.0 * p_correct / p_train;
    if (graph->partition_id == 0) {
      if (s == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc_train, p_train, p_correct);
      } else if (s == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc_train, p_train, p_correct);
      } else if (s == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc_train, p_train, p_correct);
      }
    }
  }

  void Loss(NtsVar left, NtsVar right, int type = 0) {
    NtsVar mask = (local_mask == type);
    left = left.masked_select(mask.unsqueeze(1).expand({left.size(0), left.size(1)})).view({-1, left.size(1)});
    // left = left[mask.nonzero().view(-1)];
    right = right.masked_select(mask.view({mask.size(0)}));
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = left.log_softmax(1);
    NtsVar loss_ = torch::nll_loss(a, right);
    if (ctx->training == true) {
      ctx->appendNNOp(left, loss_);
    }
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      // P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      if (graph->gnnctx->l_v_num == 0) {
        P[i]->all_reduce_to_gradient(torch::zeros_like(P[i]->W));
      } else {
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      }
      // update parameters with Adam optimizer
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }

  void UpdateZero() {
    for (int l = 0; l < (graph->gnnctx->layer_size.size() - 1); l++) {
      //          std::printf("process %d epoch %d last before\n", graph->partition_id, curr_epoch);
      P[l]->all_reduce_to_gradient(torch::zeros({P[l]->row, P[l]->col}, torch::kFloat));
      //          std::printf("process %d epoch %d last after\n", graph->partition_id, curr_epoch);
      P[l]->learnC2C_with_decay_Adam();
      P[l]->next();
    }
  }

  void Forward(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = true;
    sampler->ClusterGCNSample(graph->gnnctx->layer_size.size() - 1, graph->config->batch_size, 1000);
    // int batch_num = sampler->size();
    // MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // acc=0.0;
    // correct = 0;
    train_correct = 0;
    train_num = 0;
    val_correct = 0;
    val_num = 0;
    test_correct = 0;
    test_num = 0;
    batch = 0;
    SampledSubgraph* sg = sampler->get_one();
    // std::vector<NtsVar> X;
    NtsVar d;
    X.resize(graph->gnnctx->layer_size.size(), d);

    //  X[0]=nts::op::get_feature(sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->src(),F,graph);
    // X[0]=nts::op::get_feature(sg->sampled_sgs[0]->src(),F,graph);
    // std::cout << "dst num " << sg->sampled_sgs.back()->dst().size() << std::endl;
    X[0] = nts::op::get_feature_from_global(rpc, sg->sampled_sgs[0]->src(), sg->sampled_sgs[0]->src_size, F, graph);
    // printf("get feature done\n");

    // NtsVar target_lab=nts::op::get_label(sg->sampled_sgs.back()->dst(),L_GT_C,graph);
    NtsVar target_lab = nts::op::get_label_from_global(sg->sampled_sgs.back()->dst(), sg->sampled_sgs.back()->v_size,
                                                       global_label, graph);
    // printf("get label done\n");
    // std::cout << "target_lab " << target_lab.size(0)  << std::endl;
    //  graph->rtminfo->forward = true;
    for (int l = 0; l < (graph->gnnctx->layer_size.size() - 1); l++) {  // forward
      // printf("process layer %d\n", l);
      //  int hop=(graph->gnnctx->layer_size.size()-2)-l;
      // if(l!=0){
      //     X[l] = drpmodel(X[l]);
      // }
      NtsVar Y_i = ctx->runGraphOp<nts::op::MiniBatchFuseOp>(sg, graph, l, X[l]);
      X[l + 1] = ctx->runVertexForward(
          [&](NtsVar n_i) {
            if (l == (graph->gnnctx->layer_size.size() - 2)) {
              return P[l]->forward(n_i);
            } else {
              // return torch::relu(P[l]->forward(n_i));
              return torch::dropout(P[l]->forward(n_i), drop_rate, ctx->is_train());
            }
          },
          Y_i);
    }

    local_mask = torch::zeros_like(target_lab, at::TensorOptions().dtype(torch::kLong));
    auto vec_dst = sg->sampled_sgs.back()->dst();
    // printf("start local mask\n");
    for (int i = 0; i < sg->sampled_sgs.back()->v_size; ++i) {
      // local_mask[i] = MASK[vec_dst[i]].item<long>();
      local_mask[i] = global_mask[vec_dst[i]].item<long>();
    }
    // printf("end local mask\n");

    Loss(X[graph->gnnctx->layer_size.size() - 1], target_lab, 0);
    getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab, 0);
    MPI_Allreduce(MPI_IN_PLACE, &train_correct, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &train_num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // acc = 1.0 * correct / train_nodes;

    getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab, 1);
    MPI_Allreduce(MPI_IN_PLACE, &val_correct, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &val_num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab, 2);
    MPI_Allreduce(MPI_IN_PLACE, &test_correct, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &test_num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // graph->rtminfo->forward = false;
    if (ctx->training) {
      ctx->self_backward(false);
      Update();
    }

    batch++;

    // Test(0, target_lab, mask);
    // Test(1, target_lab, mask);
    // Test(2, target_lab, mask);

    sampler->clear_queue();
    sampler->restart();
  }

  // NtsVar get_mask() {

  // }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      int type = gnndatum->local_mask[i];
      if (type == 0) {
        train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 1) {
        val_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 2) {
        test_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      }
    }

    Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids);

    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      // printf("########### epoch %d ###########\n", i_i);
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }

      ctx->train();
      Forward(train_sampler, 0);
      float train_acc = train_correct / train_num;
      float val_acc = val_correct / val_num;
      float test_acc = test_correct / test_num;
      printf("Epoch %03d train_acc %.3f val_acc %.3f test_acc %.3f\n", i_i, train_acc, val_acc, test_acc);

      //      if (graph->partition_id == 0)
      //        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
      //                  << std::endl;
    }
    delete active;
    rpc.exit();
  }
};
