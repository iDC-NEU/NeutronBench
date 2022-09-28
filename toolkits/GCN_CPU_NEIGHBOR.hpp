#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"

class GCN_CPU_NEIGHBOR_impl {
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
  //std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  //GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  FullyRepGraph* fully_rep_graph;
  double train_sample_time = 0;
  double train_compute_time = 0;
  double mpi_comm_time = 0;
  double rpc_comm_time = 0;
  double rpc_wait_time = 0;

  
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
  // double sample_cost = 0;
  std::vector<torch::nn::BatchNorm1d> bn1d;

  ntsPeerRPC<ValueType, VertexId> rpc;

  GCN_CPU_NEIGHBOR_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false) {
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
    fully_rep_graph=new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");
       
    //cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx=new nts::ctx::NtsContext();
  }

  void get_batch_num() {
    VertexId max_vertex = 0;
    VertexId min_vertex = std::numeric_limits<VertexId>::max();
    for(int i = 0; i < graph->partitions; i++){
        max_vertex = std::max(graph->partition_offset[i+1] - graph->partition_offset[i], max_vertex);
        min_vertex = std::min(graph->partition_offset[i+1] - graph->partition_offset[i], min_vertex);
    }
    max_batch_num = max_vertex / graph->config->batch_size;
    min_batch_num = min_vertex / graph->config->batch_size;
    if(max_vertex % graph->config->batch_size != 0) {
        max_batch_num++;
    }
    if(min_vertex % graph->config->batch_size != 0) {
        min_batch_num++;
    }
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
    // std::cout << "start nn" << std::endl;
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                       graph->config->label_file,
                                       graph->config->mask_file);
    }
    // std::cout << "read done" << std::endl;

    // creating tensor to save Label and Mask
    if (graph->config->classes > 1) {
      gnndatum->registLabel(L_GT_C, gnndatum->local_label, gnndatum->gnnctx->l_v_num, graph->config->classes);
      // std::cout << "10 " << L_GT_C[10] << std::endl;
      // for (int i = 0; i < 121; ++i) {
      //   std::cout << gnndatum->local_label[100 * 121 + i] << " ";
      // }std::cout << std::endl;
      // std::cout << "100 " << L_GT_C[100] << std::endl;
      // assert(false);
    } else {
      gnndatum->registLabel(L_GT_C);
    }
    // std::cout << L_GT_C << std::endl;
    gnndatum->registMask(MASK);

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
      if(i < graph->gnnctx->layer_size.size() - 2)
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i])); 
  
    }

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
    }
    drpmodel = torch::nn::Dropout(
        torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(
        gnndatum->local_feature,
        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        torch::DeviceType::CPU);

    // X[i] is vertex representation at layer i
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    
    X[0] = F.set_requires_grad(true);

    rpc.set_comm_num(graph->partitions - 1);
    rpc.register_function("get_feature", [&](std::vector<VertexId> vertexs){
        int start = graph->partition_offset[graph->partition_id];
        int feature_size = F.size(1);
        ValueType* ntsVarBuffer = graph->Nts->getWritableBuffer(F, torch::DeviceType::CPU);
        std::vector<std::vector<ValueType>> result_vector;
        result_vector.resize(vertexs.size());

        #pragma omp parallel for
        for(int i = 0; i < vertexs.size(); i++) {
            result_vector[i].resize(feature_size);
            memcpy(result_vector[i].data(), ntsVarBuffer + (vertexs[i] - start) * feature_size,
                   feature_size * sizeof(ValueType));
        }
        return result_vector;
    });

  }



  void Update() {
    for (int i = 0; i < P.size(); i++) {
    // accumulate the gradient using all_reduce
    if (ctx->is_train() && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      if (graph->gnnctx->l_v_num == 0) {
        P[i]->all_reduce_to_gradient(torch::zeros_like(P[i]->W));
      } else {
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      }
    if (ctx->is_train() && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
    // update parameters with Adam optimizer
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }

    void UpdateZero() {
      for(int l=0;l<(graph->gnnctx->layer_size.size()-1);l++){
//          std::printf("process %d epoch %d last before\n", graph->partition_id, curr_epoch);
          P[l]->all_reduce_to_gradient(torch::zeros({P[l]->row, P[l]->col}, torch::kFloat));
//          std::printf("process %d epoch %d last after\n", graph->partition_id, curr_epoch);
          // P[l]->learnC2C_with_decay_Adam();
          // P[l]->next();
      }
  }
  
  float Forward(Sampler* sampler, int type=0) {
    graph->rtminfo->forward = true;
      
    // node sampling
    // std::cout << type << " start sample" << std::endl;
    double sample_cost = 0;
    sample_cost -= get_time();
    while(sampler->sample_not_finished()) {
      // sampler->reservoir_sample(graph->gnnctx->layer_size.size()-1,
      //                           graph->config->batch_size,
      //                           graph->gnnctx->fanout);
      // printf("batch_type %d\n", graph->config->batch_type);
      double tmp_start = -get_time();
      sampler->reservoir_sample(graph->gnnctx->layer_size.size()-1,
                                graph->config->batch_size,
                                graph->gnnctx->fanout, graph->config->batch_type, ctx->is_train());
      // printf("# sample_one cost %.3f\n", tmp_start + get_time());
      // assert(tmp_cnt++ < 3);
    }
    sample_cost += get_time();
    if (type ==  0 && graph->rtminfo->epoch >= 3) {
        train_sample_time += sample_cost;
    }
    int batch_num = sampler->size();
    // printf("## sample_nodes %d batch_num %d sample_cost %.3f\n", sampler->sample_nids.size(), batch_num, sample_cost);
  

    // int batch_num = sampler->size();
    if (type == 0 && graph->rtminfo->epoch >= 3)
      mpi_comm_time -= get_time();
    MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (type == 0 && graph->rtminfo->epoch >= 3)
      mpi_comm_time += get_time();
    // printf("batch_num %d min %d max %d\n", batch_num, min_batch_num, max_batch_num);
    // sampler->isValidSampleGraph();
    // std::cout << type << " neighbor sample done" << std::endl;

    SampledSubgraph *sg; 
    correct = 0;
    train_nodes = 0;
    batch=0;
    // if(min_batch_num == 0) {
    //     rpc.keep_running();
    // }
    // std::cout << min_batch_num << " " << max_batch_num << std::endl;
    double forward_cost = 0;
    forward_cost -= get_time();
    double tmp_time = 0;
    float f1 = 0;
    while(sampler->has_rest()){
      // std::cout << "process batch " << batch<< std::endl;
        sg=sampler->get_one();
        // for (int i = 1; i >= 0; --i) {
        //   printf("layer %d v_size %d e_size %d\n", i, sg->sampled_sgs[i]->v_size
        //           ,sg->sampled_sgs[i]->e_size);
        // }
        std::vector<NtsVar> X;
        NtsVar d;
        X.resize(graph->gnnctx->layer_size.size(),d);
      
      //  std::cout << "get feture done" << std::endl;
        // X[0]=nts::op::get_feature(sg->sampled_sgs[0]->src(),F,graph);
        // rpc.keep_running();
        tmp_time = -get_time();
        X[0] = nts::op::get_feature_from_global(rpc, sg->sampled_sgs[0]->src(), F, graph);
        tmp_time += get_time();
        if (type == 0 && graph->rtminfo->epoch >= 3)
          rpc_comm_time += tmp_time;
        // std::cout << batch_num << " "<< batch <<  " get feature done" << std::endl;
        // rpc.stop_running();
      //  std::cout << "get feature done" << std::endl;
        NtsVar target_lab=nts::op::get_label(sg->sampled_sgs.back()->dst(), L_GT_C, graph);
      //  std::cout << "get label done" << std::endl;
      //  graph->rtminfo->forward = true;
        for(int l=0;l<(graph->gnnctx->layer_size.size()-1);l++) {//forward
          //  int hop=(graph->gnnctx->layer_size.size()-2)-l;
        //  std::cout << "start process layer " << l << std::endl;
            NtsVar Y_i=ctx->runGraphOp<nts::op::MiniBatchFuseOp>(sg,graph, l, X[l]);
            // printf("run graphop done\n");
            X[l + 1]=ctx->runVertexForward([&](NtsVar n_i){
                if (l==(graph->gnnctx->layer_size.size()-2)) {
                  return P[l]->forward(n_i);
                  // return torch::mean(P[l]->forward(n_i));
                }else{
                  // if (graph->config->batch_norm) {
                    // n_i = this->bn1d[l](n_i); // for arxiv dataset
                  // }
                  return torch::dropout(P[l]->forward(n_i), drop_rate, ctx->is_train());
                  // return torch::relu(P[l]->forward(n_i));
                }
            },
            Y_i);
            // printf("run vertex forward done\n");
        } 
        // std::cout << "start loss" << std::endl;
        auto loss_ = Loss(X[graph->gnnctx->layer_size.size()-1], target_lab, graph->config->classes == 1);
        // std::cout << loss_.item<float>() << std::endl;
        // std::cout << "loss after " << loss_.data_ptr() << std::endl;

        if (ctx->training == true) {
          ctx->appendNNOp(X[graph->gnnctx->layer_size.size()-1], loss_);
        }
        // std::cout << "loss done" << std::endl;
        correct += get_correct(X[graph->gnnctx->layer_size.size()-1], target_lab, graph->config->classes == 1);
        f1 += f1_score(X[graph->gnnctx->layer_size.size()-1], target_lab, graph->config->classes == 1);
        // std::cout << "correct done" << std::endl;
        train_nodes += target_lab.size(0);

        // sg->sampled_sgs.back()->dst()
        // graph->rtminfo->forward = false;
        if (ctx->training) {
          ctx->self_backward(false);
          // std::cout << "backward done" << std::endl;
          Update();
          // std::cout << "update done" << std::endl;
          for (int i = 0; i < P.size(); i++) {
            P[i]->zero_grad();
          }
          // std::cout << "zero_grad done" << std::endl;
        }
        
        batch++;
        // if(batch == min_batch_num) {
        //   rpc.keep_running();
        // }
    }
    // std::cout << "train " << ctx->training << " batch " << batch << " max_batch_num " << max_batch_num << std::endl;
    // max_batch_num = 50;
    if (type == 0 && graph->rtminfo->epoch >= 3)
      rpc_wait_time -= get_time();
    while(ctx->training && batch !=max_batch_num){
      UpdateZero();
      // std::cout << "sync update zero " << batch << std::endl;
      batch++;
    }
    rpc.stop_running();
    if (type == 0 && graph->rtminfo->epoch >= 3)
      rpc_wait_time += get_time();

    sampler->clear_queue();
    sampler->restart();
    // acc = 1.0 * correct / sampler->work_range[1];
    // std::cout << "before " << correct << " " << train_nodes << std::endl;
    if (type == 0 && graph->rtminfo->epoch >= 3)
      mpi_comm_time -= get_time();
    MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (type == 0 && graph->rtminfo->epoch >= 3)
     mpi_comm_time += get_time();
    acc = 1.0 * correct / train_nodes;
    // std::cout << "after " << correct << " " << train_nodes << std::endl;
    // printf("train_ndoes %d\n", train_nodes);
    forward_cost += get_time();
    if (graph->partition_id == 0)
    printf("\thost %d batch_num %d sample_cost %.3f forward_cost %.3f\n", 
          graph->partition_id, batch_num, sample_cost, forward_cost);
    return graph->config->classes > 1 ? f1 / batch_num : acc;
    // if (type == 0) {
    //   printf("Train Acc: %f %d %d\n", acc, correct, sampler->work_range[1]);
    // } else if (type == 1) {
    //   printf("Eval Acc: %f %d %d\n", acc, correct, sampler->work_range[1]);
    // } else if (type == 2) {
    //   printf("Test Acc: %f %d %d\n", acc, correct, sampler->work_range[1]);
    // }
  }

  void shuffle_vec(std::vector<VertexId>& vec) {
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
    std::shuffle (vec.begin(), vec.end(), std::default_random_engine(seed));
  }

  void run() {
    double run_time = -get_time();
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n",
               iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    int batch_type = graph->config->batch_type;
    // std::cout << "l_v_num " << graph->gnnctx->l_v_num << std::endl;
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
    // for (int i = graph->partition_offset[graph->partition_id]; i < graph->partition_offset[graph->partition_id + 1]; ++i) {
      int type = gnndatum->local_mask[i];
      // std::cout << i << " " << type << " " << i + graph->partition_offset[graph->partition_id] << std::endl;
      if (type == 0) {
        train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 1) {
        val_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 2) {
        test_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      }
    }
    if (batch_type == 1) {
      // std::cout << "before shuffle train " << train_nids << std::endl;
      shuffle_vec(train_nids);
      // std::cout << "after shuffle train " << train_nids << std::endl;
      // assert (false);
      shuffle_vec(val_nids);
      shuffle_vec(test_nids);
    }
    // std::cout << "shuufel done" << std::endl;
    // std::cout << "type " << batch_type  << std::endl;
    if (batch_type == 3 || batch_type == 4) {
      // std::cout << "start sort" << std::endl;
      // for (auto &it : train_nids) {
      //   std::cout << it << "-" << graph->in_degree_for_backward[it] << std::endl;
      // }
      // std::cout << "deg print doen " << std::endl;
      if (batch_type == 3) {
         std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] > graph->in_degree_for_backward[y];
        });
      } else if (batch_type == 4){
        std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] < graph->in_degree_for_backward[y];
        });
      }
      // std::cout << "end sort" << std::endl;
      // for (int i = 0; i < 5; ++i) {
      //   std::cout << graph->in_degree_for_backward[train_nids[i]] << " ";
      // }printf("...");
      int sz = train_nids.size();
      // for (int i = sz-5; i < sz; ++i) {
      //   std::cout << graph->in_degree_for_backward[train_nids[i]] << " ";
      // }printf("\n");
      train_nids.erase(train_nids.begin() + static_cast<int>(sz*0.8), train_nids.end());
      // std::cout << "before sz " << sz << " after erase sz " << train_nids.size() << std::endl;
      // std::cout << "print vec " << train_nids << std::endl; 
      shuffle_vec(train_nids);
      // assert (false);
      // auto erased = std::erase_if(cnt, [](const auto& x) { return (x - '0') % 2 == 0; });
    }
    // std::cout << "pre done" << std::endl;
    std::cout << "train/val/test: " << train_nids.size() << " " << val_nids.size() << " " << test_nids.size() << std::endl;
    Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids);
    Sampler* eval_sampler = new Sampler(fully_rep_graph, val_nids);
    Sampler* test_sampler = new Sampler(fully_rep_graph, test_nids);

    double train_time =  0;
    double val_time =  0;
    double test_time =  0;
    for (int i_i = 0; i_i < iterations; i_i++) {
      // std::cout << "epoc " << i_i << std::endl;
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      ctx->train();
      // std::cout << "start train" << std::endl;
      if (i_i >= 3)
        train_time -= get_time();
      float train_acc = Forward(train_sampler, 0);
      if (i_i >= 3)
        train_time += get_time();
      // std::cout << "end train" << std::endl;
      
      ctx->eval();
      // std::cout << "satr eval" << std::endl;
      if (i_i >= 3)
        val_time -= get_time();
      float val_acc = Forward(eval_sampler, 1);
      if (i_i >= 3)
        val_time += get_time();
      // std::cout << "end eval" << std::endl;

      // std::cout << "start test" << std::endl;
      if (i_i >= 3)
        test_time -= get_time();
      float test_acc = Forward(test_sampler, 2);
      if (i_i >= 3)
        test_time += get_time();
      // std::cout << "end test" << std::endl;

      if (graph->partition_id == 0) {
        printf("Epoch %03d train_acc %.3f val_acc %.3f test_acc %.3f\n", 
              i_i, train_acc, val_acc, test_acc);
      }
    }
    double comm_time = mpi_comm_time + rpc_comm_time + rpc_wait_time;
    double compute_time = train_time - comm_time;
    printf("TIME(%d) sample %.3f compute_time %.3f comm_time %.3f mpi_comm %.3f rpc_comm %.3f rpc_wait_time %.3f\n", 
          graph->partition_id, train_sample_time, compute_time, comm_time, mpi_comm_time, rpc_comm_time, rpc_wait_time);

  if (graph->partition_id == 0)
    printf("Avg epoch train_time %.3f val_time %.3f test_time %.3f\n", 
          train_time / (iterations - 3), 
          val_time / (iterations - 3), 
          test_time / (iterations - 3));

    delete active;
    rpc.exit();
    run_time += get_time();
    printf("runtime cost %.3f\n", run_time);

  }

};


  // long get_correct(NtsVar &input, NtsVar &target) {
  //   // auto x = torch::ones({2, 3});
  //   // std::cout << "x0 " << x[0].all().item<int>() << std::endl;
  //   // std::cout << "x1 " << x[1].all().item<int>() << std::endl;
  //   // assert(false);
  //   long ret;
  //   if (graph->config->classes > 1) {
  //     ret = 0;
  //     // NtsVar predict = input.sigmoid();
  //     // std::cout << "sigmoid " << predict << std::endl;
  //     // predict = torch::where(predict > 0.5, 1, 0);
  //     NtsVar predict = torch::where(torch::sigmoid(input) > 0.5, 1, 0);
  //     // std::cout << predict << std::endl;
  //     auto equal = predict == target;
  //     for (int i = 0; i < input.size(0); ++i) {
  //       ret += equal[i].all().item<int>();
  //     }
  //     // for (int i = 0; i < input.size(0); ++i) {
  //     //   auto tmp = predict[i].to(torch::kLong).eq(target[i]).to(torch::kLong);
  //     //   ret += tmp.all().item<int>();
  //     // }
  //   } else {
  //     NtsVar predict = input.argmax(1);
  //     NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
  //     ret = output.sum(0).item<long>();
  //   }
  //   return ret;
  // }

  // float f1_score(NtsVar &input, NtsVar &target) {
  //   float ret;
  //   if (graph->config->classes > 1) {
  //     NtsVar predict = input.sigmoid();
  //     predict = torch::where(predict > 0.5, 1, 0);

  //     // f1 = f1_score(x_true, y_pre, average="micro")
  //     auto all_pre = predict.sum();
  //     auto x_tmp = torch::where(target == 0, 2, 1);
  //     auto true_p = (x_tmp == predict).sum();
  //     auto all_true = target.sum();
  //     auto precision = true_p / all_pre;
  //     auto recall = true_p / all_true;
  //     auto f2 = 2 * precision * recall / (precision + recall);
  //     ret =  f2.item<float>();
  //   } else {
  //     NtsVar predict = input.argmax(1);
  //     NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
  //     ret = output.sum(0).item<long>();
  //   }
  //   return ret;
  // }  

  // void Test(long s) { // 0 train, //1 eval //2 test
  //   NtsVar mask_train = MASK.eq(s);
  //   NtsVar all_train =
  //       X[graph->gnnctx->layer_size.size() - 1]
  //           .argmax(1)
  //           .to(torch::kLong)
  //           .eq(L_GT_C)
  //           .to(torch::kLong)
  //           .masked_select(mask_train.view({mask_train.size(0)}));
  //   NtsVar all = all_train.sum(0);
  //   long *p_correct = all.data_ptr<long>();
  //   long g_correct = 0;
  //   long p_train = all_train.size(0);
  //   long g_train = 0;
  //   MPI_Datatype dt = get_mpi_data_type<long>();
  //   MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
  //   MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
  //   float acc_train = 0.0;
  //   if (g_train > 0)
  //     acc_train = float(g_correct) / g_train;
  //   if (graph->partition_id == 0) {
  //     if (s == 0) {
  //       LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
  //     } else if (s == 1) {
  //       LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
  //     } else if (s == 2) {
  //       LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
  //     }
  //   }
  // }
 
  // void Loss(NtsVar &output,NtsVar &target) {
  //   //  return torch::nll_loss(a,L_GT_C);
  //   // std::cout << "start loss" << std::endl;
  //   NtsVar loss_; 

  //   if (graph->config->classes > 1) {
  //     // tensor.to(torch::kLong)
  //     // loss_ = torch::binary_cross_entropy_with_logits(output, target.to(torch::kFloat));
  //     loss_ = torch::binary_cross_entropy(torch::sigmoid(output), target.to(torch::kFloat));

  //   } else {
  //     torch::Tensor a = output.log_softmax(1);
  //     loss_ = torch::nll_loss(a, target);
  //   }
  //   // std::cout << "loss " << loss_.item<float>() << std::endl;
  //   if (ctx->training == true) {
  //     ctx->appendNNOp(output, loss_);
  //   }
  // }