#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"

class GCN_CPU_LAYER_impl {
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
  // graph
  VertexSubset *active;
  // graph with no edge data
  Graph<Empty> *graph;
  // std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  // GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext *ctx;
  // Sampler* train_sampler;
  // Sampler* val_sampler;
  // Sampler* test_sampler;
  FullyRepGraph *fully_rep_graph;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  long train_nodes;
  int max_batch_num;
  int min_batch_num;
  torch::nn::Dropout drpmodel;
  ntsPeerRPC<ValueType, VertexId> rpc;

  GCN_CPU_LAYER_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false) {
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
    gnndatum->registMask(MASK);

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
      ValueType *ntsVarBuffer = graph->Nts->getWritableBuffer(F, torch::DeviceType::CPU);
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

  long getCorrect(NtsVar &input, NtsVar &target) {
    // NtsVar predict = input.log_softmax(1).argmax(1);
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    return output.sum(0).item<long>();
  }

  void Test(long s) {  // 0 train, //1 eval //2 test
    NtsVar mask_train = MASK.eq(s);
    NtsVar all_train = X[graph->gnnctx->layer_size.size() - 1]
                           .argmax(1)
                           .to(torch::kLong)
                           .eq(L_GT_C)
                           .to(torch::kLong)
                           .masked_select(mask_train.view({mask_train.size(0)}));
    NtsVar all = all_train.sum(0);
    long *p_correct = all.data_ptr<long>();
    long g_correct = 0;
    long p_train = all_train.size(0);
    long g_train = 0;
    MPI_Datatype dt = get_mpi_data_type<long>();
    MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    float acc_train = 0.0;
    if (g_train > 0) acc_train = float(g_correct) / g_train;
    if (graph->partition_id == 0) {
      if (s == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
      }
    }
  }

  void Loss(NtsVar &left, NtsVar &right) {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = left.log_softmax(1);
    NtsVar loss_;
    loss_ = torch::nll_loss(a, right);
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

  float Forward(Sampler *sampler, int type = 0) {
    graph->rtminfo->forward = true;

    // layer sampling
    while (sampler->sample_not_finished()) {
      sampler->LayerUniformSample(graph->gnnctx->layer_size.size() - 1, graph->config->batch_size,
                                  graph->gnnctx->fanout);
    }
    int batch_num = sampler->size();
    // std::cout << "batch_num " << batch_num << std::endl;
    MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    // printf("batch_num %d min %d max %d\n", batch_num, min_batch_num, max_batch_num);
    sampler->isValidSampleGraph();

    // std::cout << "sample is done" << std::endl;
    SampledSubgraph *sg;
    correct = 0;
    train_nodes = 0;
    batch = 0;
    while (sampler->has_rest()) {
      sg = sampler->get_one();
      // for (int i = 0; i < 2; ++i) {
      //   printf("layer %d v_size %d e_size %d\n", i, sg->sampled_sgs[i]->v_size
      //           ,sg->sampled_sgs[i]->e_size);
      // }

      std::vector<NtsVar> X;
      NtsVar d;
      X.resize(graph->gnnctx->layer_size.size(), d);

      //  X[0]=nts::op::get_feature(sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->src(),F,graph);
      // X[0]=nts::op::get_feature(sg->sampled_sgs[0]->src(),F,graph);
      X[0] = nts::op::get_feature_from_global(rpc, sg->sampled_sgs[0]->src().data(), sg->sampled_sgs[0]->src_size, F,
                                              graph);

      NtsVar target_lab =
          nts::op::get_label(sg->sampled_sgs.back()->dst().data(), sg->sampled_sgs.back()->v_size, L_GT_C, graph);
      //  graph->rtminfo->forward = true;
      for (int l = 0; l < (graph->gnnctx->layer_size.size() - 1); l++) {  // forward

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
      Loss(X[graph->gnnctx->layer_size.size() - 1], target_lab);
      correct += getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab);
      train_nodes += target_lab.size(0);
      // sg->sampled_sgs.back()->dst()
      // graph->rtminfo->forward = false;
      if (ctx->training) {
        ctx->self_backward(false);
        Update();
      }
      batch++;
    }
    while (ctx->training && batch != max_batch_num) {
      UpdateZero();
      // std::cout << "sync update zero " << batch << std::endl;
      batch++;
    }
    rpc.stop_running();

    sampler->clear_queue();
    sampler->restart();
    // acc = 1.0 * correct / sampler->work_range[1];
    // std::cout << "before " << correct << " " << train_nodes << std::endl;
    MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    acc = 1.0 * correct / train_nodes;
    // std::cout << "after " << correct << " " << train_nodes << std::endl;
    // printf("train_ndoes %d\n", train_nodes);
    return acc;
    // if (type == 0) {
    //   printf("Train Acc: %f %d %d\n", acc, correct, sampler->work_range[1]);
    // } else if (type == 1) {
    //   printf("Eval Acc: %f %d %d\n", acc, correct, sampler->work_range[1]);
    // } else if (type == 2) {
    //   printf("Test Acc: %f %d %d\n", acc, correct, sampler->work_range[1]);
    // }
  }

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

    Sampler *train_sampler = new Sampler(fully_rep_graph, train_nids);
    Sampler *eval_sampler = new Sampler(fully_rep_graph, val_nids);
    Sampler *test_sampler = new Sampler(fully_rep_graph, test_nids);

    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }

      ctx->train();
      float train_acc = Forward(train_sampler, 0);

      ctx->eval();
      float val_acc = Forward(eval_sampler, 1);
      float test_acc = Forward(test_sampler, 2);
      if (graph->partition_id == 0) {
        printf("Epoch %03d train_acc %.3f val_acc %.3f test_acc %.3f\n", i_i, train_acc, val_acc, test_acc);
      }
    }
    delete active;
    rpc.exit();
  }
};
