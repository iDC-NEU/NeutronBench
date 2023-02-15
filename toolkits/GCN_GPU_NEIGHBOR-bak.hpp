#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/cuda_memory.hpp"
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
  int layers;
  // graph
  VertexSubset* active;
  // graph with no edge data
  Graph<Empty>* graph;
  // std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum* gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  // GraphOperation *gt;
  PartitionedGraph* partitioned_graph;
  // Variables
  std::vector<Parameter*> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  FullyRepGraph* fully_rep_graph;
  double train_sample_time = 0;
  double train_compute_time = 0;
  double mpi_comm_time = 0;
  double rpc_comm_time = 0;
  double rpc_wait_time = 0;
  float loss_epoch = 0;

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

  ntsPeerRPC<ValueType, VertexId>* rpc;
  int hosts;

  GCN_CPU_NEIGHBOR_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
                        bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    graph->init_gnnctx_fanout(graph->config->fanout_string);
    reverse(graph->gnnctx->fanout.begin(), graph->gnnctx->fanout.end());
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
    hosts = graph->partitions;
    if (hosts > 1) {
      rpc = new ntsPeerRPC<ValueType, VertexId>();
    } else {
      rpc = nullptr;
    }
  }

  void init_active() {
    active = graph->alloc_vertex_subset();
    active->fill();
  }

  void init_graph() {
    fully_rep_graph = new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");

    // cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx = new nts::ctx::NtsContext();
  }

  void get_batch_num() {
    VertexId max_vertex = 0;
    VertexId min_vertex = std::numeric_limits<VertexId>::max();
    for (int i = 0; i < graph->partitions; i++) {
      max_vertex = std::max(graph->partition_offset[i + 1] - graph->partition_offset[i], max_vertex);
      min_vertex = std::min(graph->partition_offset[i + 1] - graph->partition_offset[i], min_vertex);
    }
    max_batch_num = max_vertex / graph->config->batch_size;
    min_batch_num = min_vertex / graph->config->batch_size;
    if (max_vertex % graph->config->batch_size != 0) {
      max_batch_num++;
    }
    if (min_vertex % graph->config->batch_size != 0) {
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
    layers = graph->gnnctx->layer_size.size() - 1;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    gnndatum = new GNNDatum(graph->gnnctx, graph);
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
    }
    // creating tensor to save Label and Mask
    if (graph->config->classes > 1) {
      gnndatum->registLabel(L_GT_C, gnndatum->local_label, gnndatum->gnnctx->l_v_num, graph->config->classes);
    } else {
      gnndatum->registLabel(L_GT_C);
    }
    gnndatum->registMask(MASK);

    for (int i = 0; i < layers; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      if (graph->config->batch_norm && i < layers - 1) {
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i]));
      }
    }

    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
    }

    // drpmodel = torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);

    // X[i] is vertex representation at layer i
    for (int i = 0; i < layers + 1; i++) {
      NtsVar d;
      X.push_back(d);
    }

    X[0] = F.set_requires_grad(true);

    if (hosts > 1) {
      rpc->set_comm_num(graph->partitions - 1);
      rpc->register_function("get_feature", [&](std::vector<VertexId> vertexs) {
        int start = graph->partition_offset[graph->partition_id];
        int feature_size = F.size(1);
        ValueType* ntsVarBuffer = graph->Nts->getWritableBuffer(F, torch::DeviceType::CPU);
        std::vector<std::vector<ValueType>> result_vector;
        result_vector.resize(vertexs.size());

        omp_set_num_threads(threads);
#pragma omp parallel for
        for (int i = 0; i < vertexs.size(); i++) {
          result_vector[i].resize(feature_size);
          memcpy(result_vector[i].data(), ntsVarBuffer + (vertexs[i] - start) * feature_size,
                 feature_size * sizeof(ValueType));
        }
        return result_vector;
      });
    }
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
      // P[i]->learnC2C_with_Adam();
      P[i]->next();
    }
  }

  void UpdateZero() {
    for (int l = 0; l < layers; l++) {
      P[l]->all_reduce_to_gradient(torch::zeros({P[l]->row, P[l]->col}, torch::kFloat));
    }
  }

  NtsVar vertexForward(NtsVar& n_i) {
    int l = graph->rtminfo->curr_layer;
    if (l == layers - 1) {  // last layer
      return P[l]->forward(n_i);
    } else {
      if (graph->config->batch_norm) {
        n_i = this->bn1d[l](n_i);  // for arxiv dataset
      }
      return torch::dropout(torch::relu(P[l]->forward(n_i)), drop_rate, ctx->is_train());
    }
  }

  float Forward(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = true;
    correct = 0;
    train_nodes = 0;
    float f1_epoch = 0;
    batch = 0;
    loss_epoch = 0;
    double nn_cost = 0;
    double backward_cost = 0;

    if (graph->config->batch_type != SEQUENCE) {  // shuffle
      shuffle_vec(sampler->sample_nids);
    }

    // node sampling
    sampler->zero_debug_time();
    double sample_cost = 0;
    double get_feature_cost = 0;
    double get_label_cost = 0;
    double forward_nn_cost = 0;
    double forward_acc_cost = 0;
    double forward_graph_cost = 0;
    double forward_loss_cost = 0;
    double forward_other_cost = 0;
    double forward_append_cost = 0;
    double backward_nn_time = 0;
    double update_cost = 0;

    double train_cost = 0;
    double inner_cost = 0;
    double generate_csr_time = 0;
    double convert_time = 0;
    double debug_time = 0;

    int batch_num = sampler->batch_nums;
    if (hosts > 1) {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
    } else {
      max_batch_num = batch_num;
    }

    // LOG_DEBUG("epoch %d start compute", graph->rtminfo->epoch);
    double used_gpu_mem, total_gpu_mem;
    for (VertexId i = 0; i < sampler->batch_nums; ++i) {
      sample_cost -= get_time();
      sampler->sample_one(graph->config->batch_type, ctx->is_train());
      sample_cost += get_time();
      // LOG_DEBUG("epoch %d batch %d sample_one done", graph->rtminfo->epoch, i);

      auto ssg = sampler->subgraph;
      if (graph->config->mini_pull > 0) {  // generate csr structure for backward of pull mode
        generate_csr_time -= get_time();
        for (auto p : ssg->sampled_sgs) {
          convert_time -= get_time();
          // LOG_DEBUG("batch %d start generate csr %p", i, p);
          p->generate_csr_from_csc();
          // LOG_DEBUG("batch %d generate csr done", i);
          convert_time += get_time();
          debug_time -= get_time();
          /////////// check
          // p->debug_generate_csr_from_csc();
          debug_time += get_time();
        }
        // }
        generate_csr_time += get_time();
        LOG_DEBUG("epoch %d batch %d pull gemnerate csr done", graph->rtminfo->epoch, i);
      }

      if (type == 0 && graph->rtminfo->epoch >= 3) train_sample_time += sample_cost;

      // sampler->reverse_sgs();
      // std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());

      train_cost -= get_time();
      forward_other_cost -= get_time();
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
      // std::vector<NtsVar> X;
      // NtsVar d;
      // X.resize(graph->gnnctx->layer_size.size(), d);
      forward_other_cost += get_time();
      // rpc.keep_running();
      get_feature_cost -= get_time();
      if (hosts > 1) {
        X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size,
                                                F, graph);
        // if (type == 0 && graph->rtminfo->epoch >= 3) rpc_comm_time += tmp_time;
      } else {
        X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
      }
      get_feature_cost += get_time();
      // LOG_DEBUG("epoch %d batch %d get feature done", graph->rtminfo->epoch, i);

      get_label_cost -= get_time();
      NtsVar target_lab =
          nts::op::get_label(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
      get_label_cost += get_time();
      // LOG_DEBUG("label.size() %d", target_lab.size(0));
      // LOG_DEBUG("epoch %d batch %d get label done", graph->rtminfo->epoch, i);

      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        forward_graph_cost -= get_time();
        NtsVar Y_i = ctx->runGraphOp<nts::op::MiniBatchFuseOp>(ssg, graph, l, X[l]);
        forward_graph_cost += get_time();
        // LOG_DEBUG("\tbatch %d layer %d graph compute done", i, l);
        // std::cout << Y_i << std::endl;
        // assert(false);

        forward_nn_cost -= get_time();
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
        forward_nn_cost += get_time();
        // LOG_DEBUG("\tbatch %d layer %d nn compute done", i, l);
      }

      forward_loss_cost -= get_time();
      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();
      forward_loss_cost += get_time();
      // LOG_DEBUG("batch %d loss compute done", i);

      if (ctx->training == true) {
        forward_append_cost -= get_time();
        ctx->appendNNOp(X[layers], loss_);
        forward_append_cost += get_time();

        backward_cost -= get_time();
        ctx->b_nn_time = 0;
        ctx->self_backward(false);
        backward_nn_time += ctx->b_nn_time;
        backward_cost += get_time();
        // LOG_DEBUG("batch %d backward done", i);

        update_cost -= get_time();
        Update();
        update_cost += get_time();
        // LOG_DEBUG("batch %d update done", i);
      }

      forward_acc_cost -= get_time();
      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      forward_acc_cost += get_time();
      // LOG_DEBUG("batch %d acc compute done", i);

      train_cost += get_time();
      // std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
      // sampler->reverse_sgs();
    }
    loss_epoch /= sampler->batch_nums;
    // LOG_DEBUG("sample_cost %.3f", sample_cost);
    // LOG_DEBUG("generate_csr %.3f, convert %.3f debug %.3f ", generate_csr_time, convert_time, debug_time);

    if (hosts > 1) {
      if (type == 0 && graph->rtminfo->epoch >= 3) rpc_wait_time -= get_time();
      while (ctx->training && batch != max_batch_num) {
        UpdateZero();
        batch++;
      }
      rpc->stop_running();
      if (type == 0 && graph->rtminfo->epoch >= 3) rpc_wait_time += get_time();
    }

    sampler->restart();

    if (type == 0) {
      LOG_INFO("trainning cost %.3f", train_cost);
    } else {
      LOG_INFO("evaluation cost %.3f", train_cost);
    }
    LOG_INFO("  get_feature/lable (%.3f/%.3f)", get_feature_cost, get_label_cost);
    LOG_INFO("  forward: graph %.3f, nn %.3f, loss %.3f, acc %.3f, update %.3f", forward_graph_cost, forward_nn_cost,
             forward_loss_cost, forward_acc_cost, update_cost);
    LOG_INFO("  backward cost %.3f, b_nn_time %.3f", backward_cost, backward_nn_time);
    // LOG_DEBUG("forward other %.3f, append %.3f, backward cost %.3f train cost %.3f", forward_other_cost,
    // forward_append_cost, backward_cost, train_cost); LOG_DEBUG("forward other %.3f, backward cost %.3f train cost
    // %.3f", forward_other_cost, backward_cost, train_cost); LOG_DEBUG("forward all %.3f", forward_graph_cost +
    // forward_nn_cost + forward_loss_cost + forward_acc_cost + update_cost + forward_other_cost + forward_append_cost +
    // backward_cost
    //           +get_feature_cost + get_label_cost);
    printf("\n");

    if (graph->config->classes > 1) {
      // LOG_DEBUG("use f1_score!");
      f1_epoch /= sampler->batch_nums;
      MPI_Allreduce(MPI_IN_PLACE, &f1_epoch, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      acc = f1_epoch / hosts;
    } else {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
      acc = 1.0 * correct / train_nodes;
    }

    return acc;
  }

  void zero_grad() {
    // LOG_DEBUG("P.sizes %d", P.size());
    for (int i = 0; i < P.size(); i++) {
      // LOG_DEBUG("zero_grad(%d) addr %p", i, P[i]);
      P[i]->zero_grad();
    }
  }

  void shuffle_vec(std::vector<VertexId>& vec) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
  }

  float run() {
    double pre_time = -get_time();
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    BatchType batch_type = graph->config->batch_type;
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      // for (int i = graph->partition_offset[graph->partition_id]; i < graph->partition_offset[graph->partition_id +
      // 1]; ++i) {
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

    if (batch_type == RANDOM || batch_type == DELLOW || batch_type == DELHIGH) {  // random
      if (batch_type == DELHIGH) {
        std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] > graph->in_degree_for_backward[y];
        });
      } else if (batch_type == DELLOW) {
        std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] < graph->in_degree_for_backward[y];
        });
      } else {  // random del nodes
        shuffle_vec(train_nids);
      }
      int sz = train_nids.size();
      train_nids.erase(train_nids.begin() + static_cast<int>(sz * (1 - graph->config->del_frac)), train_nids.end());

      // shuffle_vec(train_nids);
    }

    printf("label rate: %.3f, train/val/test: (%d/%d/%d) (%.3f/%.3f/%.3f)\n",
           1.0 * (train_nids.size() + val_nids.size() + test_nids.size()) / graph->vertices, train_nids.size(),
           val_nids.size(), test_nids.size(), train_nids.size() * 1.0 / graph->vertices,
           val_nids.size() * 1.0 / graph->vertices, test_nids.size() * 1.0 / graph->vertices);

    Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids);
    Sampler* eval_sampler = new Sampler(fully_rep_graph, val_nids);
    Sampler* test_sampler = new Sampler(fully_rep_graph, test_nids);
    // LOG_DEBUG("sampler contructor is done");
    pre_time += get_time();

    double train_time = 0;
    double val_time = 0;
    double test_time = 0;
    float best_val_acc = 0;
    double run_time = -get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;

      ctx->train();
      if (i_i >= graph->config->time_skip) train_time -= get_time();
      float train_acc = Forward(train_sampler, 0);
      float train_loss = loss_epoch;
      if (i_i >= graph->config->time_skip) train_time += get_time();
      // LOG_DEBUG("epoch %d train Forward() done", i_i);

      ctx->eval();
      if (i_i >= graph->config->time_skip) val_time -= get_time();
      float val_acc = Forward(eval_sampler, 1);
      float val_loss = loss_epoch;
      best_val_acc = std::max(best_val_acc, val_acc);
      if (i_i >= graph->config->time_skip) val_time += get_time();
      // LOG_DEBUG("epoch %d eval Forward() done", i_i);

      // std::cout << "start test" << std::endl;
      // if (i_i >= graph->config->time_skip) test_time -= get_time();
      // float test_acc = Forward(test_sampler, 2);
      // if (i_i >= graph->config->time_skip) test_time += get_time();
      // std::cout << "end test" << std::endl;

      float test_acc = 0;
      if (graph->partition_id == 0) {
        printf("Epoch %03d loss %.3f train_acc %.3f val_acc %.3f test_acc %.3f\n\n", i_i, train_loss, train_acc,
               val_acc, test_acc);
      }
    }
    printf("best val acc: %.3f\n", best_val_acc);

    double comm_time = mpi_comm_time + rpc_comm_time + rpc_wait_time;
    double compute_time = train_time - comm_time;
    printf(
        "train TIME(%d) sample %.3f compute_time %.3f comm_time %.3f mpi_comm %.3f rpc_comm %.3f rpc_wait_time %.3f\n",
        graph->partition_id, train_sample_time, compute_time, comm_time, mpi_comm_time, rpc_comm_time, rpc_wait_time);
    printf("train/val/test time: %.3f, %.3f, %.3f\n", train_time, val_time, test_time);

    if (hosts > 1) rpc->exit();
    run_time += get_time();
    printf("run(): pre_time %.3fs runtime: %.3fs\n", pre_time, run_time);

    if (graph->partition_id == 0)
      printf("Avg epoch train_time %.3f val_time %.3f test_time %.3f\n",
             train_time / (iterations - graph->config->time_skip), val_time / (iterations - graph->config->time_skip),
             test_time / (iterations - graph->config->time_skip));

    delete active;
    return best_val_acc;
  }
};