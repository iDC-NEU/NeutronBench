#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"

class GCN_GPU_NEIGHBOR_impl {
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
  ValueType best_val_acc;
  double start_time;
  int layers;
  double used_gpu_mem, total_gpu_mem;
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
  NtsVar MASK_gpu;
  // GraphOperation *gt;
  PartitionedGraph* partitioned_graph;
  // Variables
  std::vector<Parameter*> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  FullyRepGraph* fully_rep_graph;
  double train_compute_time = 0;
  double mpi_comm_time = 0;
  double rpc_comm_time = 0;
  double rpc_wait_time = 0;
  float loss_epoch = 0;
  Sampler* train_sampler = nullptr;
  Sampler* eval_sampler = nullptr;
  Sampler* test_sampler = nullptr;
  // double gcn_start_time = 0;
  double gcn_run_time = 0;
  // int batch_size_switch_idx = 0;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  long train_nodes;
  int max_batch_num;
  int min_batch_num;
  std::string dataset_name;
  torch::nn::Dropout drpmodel;
  // double sample_cost = 0;
  std::vector<torch::nn::BatchNorm1d> bn1d;

  ntsPeerRPC<ValueType, VertexId>* rpc;
  int hosts;
  Cuda_Stream* cuda_stream;
  // std::unordered_map<std::string, std::vector<int>> batch_size_mp;
  // std::vector<int> batch_size_vec;

  GCN_GPU_NEIGHBOR_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
                        bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    // graph->init_gnnctx_fanout(graph->config->fanout_string);
    graph->init_gnnctx_fanout(graph->gnnctx->fanout, graph->config->fanout_string);
    graph->init_gnnctx_fanout(graph->gnnctx->val_fanout, graph->config->val_fanout_string);
    assert(graph->gnnctx->fanout.size() == graph->gnnctx->val_fanout.size());
    reverse(graph->gnnctx->fanout.begin(), graph->gnnctx->fanout.end());
    reverse(graph->gnnctx->val_fanout.begin(), graph->gnnctx->val_fanout.end());
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
    best_val_acc = 0;
    cuda_stream = new Cuda_Stream();

    // batch_size_mp["ppi"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 9716};
    // batch_size_mp["ppi-large"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44906};
    // batch_size_mp["flickr"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44625};
    // batch_size_mp["AmazonCoBuy_computers"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8250};
    // batch_size_mp["ogbn-arxiv"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    // 65536, 90941}; batch_size_mp["AmazonCoBuy_photo"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4590};

    // batch_size_switch_idx = 0;
    // batch_size_vec = graph->config->batch_size_vec;
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
    // const uint64_t seed = 2000;
    // torch::manual_seed(seed);
    // torch::cuda::manual_seed_all(seed);

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
    MASK_gpu = MASK.cuda();
    gnndatum->generate_gpu_data();

    torch::Device GPU(torch::kCUDA, 0);

    for (int i = 0; i < layers; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      if (graph->config->batch_norm && i < layers - 1) {
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i]));
        // bn1d.back().to(GPU);
        // bn1d.back().cuda();
      }
    }

    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      P[i]->to(GPU);
      P[i]->Adam_to_GPU();
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

// omp_set_num_threads(threads);
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
      P[i]->learnC2G_with_decay_Adam();
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
        // std::cout << n_i.device() << std::endl;
        // std::cout << this->bn1d[l].device() << std::endl;
        n_i = this->bn1d[l](n_i);  // for arxiv dataset
        // n_i = torch::batch_norm(n_i);
      }
      return torch::dropout(torch::relu(P[l]->forward(n_i)), drop_rate, ctx->is_train());
    }
  }

  // float Forward(Sampler* sampler, int type = 0) {
  std::pair<float, double> Forward(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = true;
    correct = 0;
    train_nodes = 0;
    float f1_epoch = 0;
    batch = 0;
    loss_epoch = 0;
    double nn_cost = 0;
    double backward_cost = 0;

    // enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS};
    if (graph->config->batch_type == SHUFFLE || graph->config->batch_type == RANDOM ||
        graph->config->batch_type == DELLOW || graph->config->batch_type == DELHIGH) {
      shuffle_vec_seed(sampler->sample_nids);
    }

    // node sampling
    // sampler->zero_debug_time();
    double epoch_sample_time = 0;
    double epoch_trans_graph_time = 0;
    double epoch_trans_feature_time = 0;
    double epoch_trans_label_time = 0;
    double epoch_train_time = 0;
    int batch_num = sampler->batch_nums;

    if (hosts > 1) {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
    } else {
      max_batch_num = batch_num;
    }

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }

    sampler->metis_batch_id = 0;
    int batch_id = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      if (graph->config->run_time > 0 && gcn_run_time >= graph->config->run_time) {
        break;
      }
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute

      epoch_sample_time -= get_time();
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
      epoch_sample_time += get_time();

      epoch_trans_graph_time -= get_time();
      ssg->trans_graph_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      // ssg->trans_graph_to_gpu_async(cuda_stream->stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      epoch_trans_graph_time += get_time();

      epoch_trans_feature_time -= get_time();
      // sampler->load_feature_gpu(cuda_stream, ssg, X[0], gnndatum->dev_local_feature);
      sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      epoch_trans_feature_time += get_time();

      epoch_trans_label_time -= get_time();
      // sampler->load_label_gpu(cuda_stream, ssg, target_lab, gnndatum->dev_local_label);
      sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      epoch_trans_label_time += get_time();

      epoch_train_time -= get_time();
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], cuda_stream);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->is_train()) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }
      epoch_train_time += get_time();

      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }

      sampler->reverse_sgs();
      batch_id++;
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
    loss_epoch /= sampler->batch_nums;

    if (graph->config->classes > 1) {
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

    double epoch_trans_time = epoch_trans_graph_time + epoch_trans_feature_time + epoch_trans_label_time;
    double epoch_all_train_time = epoch_sample_time + epoch_train_time + epoch_trans_time;
    gcn_run_time += epoch_all_train_time;

    // LOG_DEBUG("epoch %d time %.3f, sample_time %.3f trans_time %.3f train_time %.3f\n", graph->rtminfo->epoch,
    // epoch_all_train_time, epoch_sample_time, epoch_trans_time, epoch_train_time); if (hosts > 1) {
    //   if (type == 0 && graph->rtminfo->epoch >= 3) rpc_wait_time -= get_time();
    //   while (ctx->training && batch != max_batch_num) {
    //     UpdateZero();
    //     batch++;
    //   }
    //   rpc->stop_running();
    //   if (type == 0 && graph->rtminfo->epoch >= 3) rpc_wait_time += get_time();
    // }
    return {acc, epoch_all_train_time};
  }

  float EvalForward(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = true;
    correct = 0;
    train_nodes = 0;
    float f1_epoch = 0;
    batch = 0;
    loss_epoch = 0;
    double nn_cost = 0;
    double backward_cost = 0;

    // enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS};
    if (graph->config->batch_type == SHUFFLE || graph->config->batch_type == RANDOM ||
        graph->config->batch_type == DELLOW || graph->config->batch_type == DELHIGH) {
      shuffle_vec_seed(sampler->sample_nids);
    }

    // node sampling
    sampler->zero_debug_time();
    int batch_num = sampler->batch_nums;
    if (hosts > 1) {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
    } else {
      max_batch_num = batch_num;
    }

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }

    sampler->metis_batch_id = 0;
    int i = -1;
    while (sampler->work_offset < sampler->work_range[1]) {
      ++i;
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());

      ssg->trans_graph_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);

      if (ctx->is_train()) zero_grad();   // should zero grad after every mini batch compute
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], cuda_stream);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->is_train()) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }

      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      sampler->reverse_sgs();
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();

    if (hosts > 1) {
      while (ctx->training && batch != max_batch_num) {
        UpdateZero();
        batch++;
      }
      rpc->stop_running();
    }

    loss_epoch /= sampler->batch_nums;
    if (graph->config->classes > 1) {
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
    for (int i = 0; i < P.size(); i++) {
      P[i]->zero_grad();
    }
  }

  void saveW() {
    for (int i = 0; i < layers; ++i) {
      P[i]->save_W("/home/yuanh/neutron-sanzo/saved_modules", graph->config->dataset_name, i);
    }
  }

  void loadW() {
    for (int i = 0; i < layers; ++i) {
      P[i]->load_W("/home/yuanh/neutron-sanzo/saved_modules", graph->config->dataset_name, i);
    }
  }

void count_sample_hop_nodes(Sampler* sampler) {
    long tmp = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      sampler->reverse_sgs();
      for (auto sg : ssg->sampled_sgs) {
        tmp += sg->e_size;
      }
    }
    printf("all batch edges %ld", tmp);
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }

  float run() {
    double pre_time = -get_time();
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    BatchType batch_type = graph->config->batch_type;
    best_val_acc == 0;
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

    if (batch_type == DELLOW || batch_type == DELHIGH) {
      if (batch_type == DELHIGH) {
        std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] > graph->in_degree_for_backward[y];
        });
      } else if (batch_type == DELLOW) {
        std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] < graph->in_degree_for_backward[y];
        });
      }
      int sz = train_nids.size();
      train_nids.erase(train_nids.begin() + static_cast<int>(sz * (1 - graph->config->del_frac)), train_nids.end());
    }
    train_sampler = new Sampler(fully_rep_graph, train_nids);
    train_sampler->show_fanout("train sampler");
    // eval_sampler = new Sampler(fully_rep_graph, val_nids, true);  // true mean full batch
    eval_sampler = new Sampler(fully_rep_graph, val_nids);  // true mean full batch
    if (graph->config->val_batch_size == 0) graph->config->val_batch_size = graph->config->batch_size;
    eval_sampler->update_batch_size(graph->config->val_batch_size);  // true mean full batch
    // eval_sampler->update_fanout(-1);        ÷            // val not sample
    eval_sampler->update_fanout(graph->gnnctx->val_fanout);  // val not sample
    eval_sampler->show_fanout("val sampler");
    eval_sampler->subgraph->show_fanout("val subgraph sampler");
    test_sampler = new Sampler(fully_rep_graph, test_nids, true);  // true mean full batch

    // count_sample_hop_nodes(train_sampler);
    // assert(false);

    if (batch_type == METIS) {
      int partition_num = (train_nids.size() + graph->config->batch_size - 1) / graph->config->batch_size;
      std::vector<VertexId> metis_partition_id;
      std::vector<VertexId> metis_partition_offset;
      double metis_time = -get_time();
      MetisPartitionGraph(fully_rep_graph, partition_num, "cut", metis_partition_id, metis_partition_offset);
      metis_time += get_time();
      LOG_DEBUG("metis partition cost %.3f", metis_time);
      LOG_DEBUG("metis_partition_id %d, train_nids %d", metis_partition_id.size(), train_nids.size());
      assert(partition_num + 1 == metis_partition_offset.size());

      train_sampler->update_metis_data(metis_partition_id, metis_partition_offset);
      train_sampler->update_batch_nums();
    }

    LOG_INFO("label rate: %.3f, train/val/test: (%d/%d/%d) (%.3f/%.3f/%.3f)",
             1.0 * (train_nids.size() + val_nids.size() + test_nids.size()) / graph->vertices, train_nids.size(),
             val_nids.size(), test_nids.size(), train_nids.size() * 1.0 / graph->vertices,
             val_nids.size() * 1.0 / graph->vertices, test_nids.size() * 1.0 / graph->vertices);

    pre_time += get_time();

    float config_run_time = graph->config->run_time;
    if (config_run_time > 0) {
      start_time = get_time();
      iterations = INT_MAX;
      LOG_DEBUG("iterations %d config_run_time %.3f", iterations, config_run_time);
    }
    gcn_run_time = 0;

    for (int i_i = 0; i_i < iterations; i_i++) {
      if (config_run_time > 0 && gcn_run_time >= config_run_time) {
        iterations = i_i;
        break;
      }
      graph->rtminfo->epoch = i_i;

      // update batch size should before Forward()
      if (graph->config->batch_switch_time > 0) {
        bool ret = train_sampler->update_batch_size_from_time(gcn_run_time);
        if (ret) eval_sampler->update_batch_size(train_sampler->batch_size);


        // load best parameter
        if (ret && graph->config->best_parameter > 0) {
          loadW();
          // double tmp_val_acc = EvalForward(eval_sampler, 1);
          // LOG_DEBUG("after loadW val_acc %.3f best_val_acc %.3f", tmp_val_acc, best_val_acc);
        }
      }

      ctx->train();
      auto [train_acc, epoch_train_time] = Forward(train_sampler, 0);
      float train_loss = loss_epoch;

      ctx->eval();
      double val_train_cost = -get_time();
      float val_acc = EvalForward(eval_sampler, 1);
      val_train_cost += get_time();
      float val_loss = loss_epoch;

      if (graph->partition_id == 0) {
        LOG_INFO(
            "Epoch %03d train_loss %.3f train_acc %.3f val_loss %.3f val_acc %.3f (train_time %.3f val_time %.3f, "
            "gcn_run_time %.3f) batch_size (%d, %d)",
            i_i, train_loss, train_acc, val_loss, val_acc, epoch_train_time, val_train_cost, gcn_run_time, train_sampler->batch_size, eval_sampler->batch_size);
      }

      if (val_acc > best_val_acc) {
        best_val_acc = val_acc;

        // save best parameter
        if (graph->config->best_parameter > 0) {
          saveW();
          LOG_DEBUG("saveW: best_val_acc %.3f", best_val_acc);
        }
      }

      if (graph->config->sample_switch_time > 0) {
        bool ret = train_sampler->update_sample_rate_from_time(gcn_run_time);
        if (ret) eval_sampler->update_batch_size(train_sampler->batch_size);
      }
      if (graph->config->batch_switch_acc > 0) {
        bool ret = train_sampler->update_batch_size_from_acc(i_i, val_acc, gcn_run_time);
        if (ret) eval_sampler->update_batch_size(train_sampler->batch_size);
      }
    }


    delete active;
    return best_val_acc;
  }
};