#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"

class GCN_GPU_NEIGHBOR_EXP3_impl {
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
  double train_sample_time = 0;
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
  double gcn_gather_cost = 0;
  double gcn_trans_cost = 0;
  double gcn_trans_cost_cache = 0;
  double gcn_gather_cost_cache = 0;
  float* dev_cache_feature;
  VertexId *local_idx, *local_idx_cache, *dev_local_idx, *dev_local_idx_cache;

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
  std::vector<int> cache_node_idx_seq;
  // std::unordered_set<int> cache_node_hashmap;
  // std::vector<int> cache_node_hashmap;
  VertexId* cache_node_hashmap;
  VertexId* dev_cache_node_hashmap;
  int cache_node_num = 0;
  // std::unordered_map<std::string, std::vector<int>> batch_size_mp;
  // std::vector<int> batch_size_vec;
  ~GCN_GPU_NEIGHBOR_EXP3_impl() {
    delete active;
  }

  GCN_GPU_NEIGHBOR_EXP3_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
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
    best_val_acc = 0;

    // batch_size_mp["ppi"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 9716};
    // batch_size_mp["ppi-large"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44906};
    // batch_size_mp["flickr"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44625};
    // batch_size_mp["AmazonCoBuy_computers"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8250};
    // batch_size_mp["ogbn-arxiv"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    // 65536, 90941}; batch_size_mp["AmazonCoBuy_photo"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4590};

    // batch_size_switch_idx = 0;
    // batch_size_vec = graph->config->batch_size_vec;
    

    if (graph->config->cache_rate > 0) {
      // determine the num of cache nodes
      cache_node_num = graph->vertices * graph->config->cache_rate;
      if (cache_node_num > graph->vertices) cache_node_num = graph->vertices;
      LOG_DEBUG("cache_node_num %d (%.3f)", cache_node_num, 1.0 * cache_node_num / graph->vertices);

      cache_node_idx_seq.resize(graph->vertices);
      std::iota(cache_node_idx_seq.begin(), cache_node_idx_seq.end(), 0);
      // cache_node_hashmap.resize(graph->vertices);
      cache_node_hashmap = (VertexId *)cudaMallocPinned(graph->vertices * sizeof(VertexId));
      dev_cache_node_hashmap = (VertexId *)getDevicePointer(cache_node_hashmap);

      #pragma omp parallel for 
      for (int i = 0; i < graph->vertices; ++i) {
        cache_node_hashmap[i] = -1;
        assert(cache_node_hashmap[i] == -1);
      }

      if (graph->config->cache_policy == "sample") {
        LOG_DEBUG("cache_sample");
        cache_sample(cache_node_idx_seq);
      } else { // default cache high degree
        LOG_DEBUG("cache_high_degree");
        cache_high_degree(cache_node_idx_seq);
      }
      
      

      // // gather_cache_feature, prepare trans to gpu
      // LOG_DEBUG("start gather_cpu_cache_feature");
      // float* local_cache_feature_gather = new float[cache_node_num * feat_dim];
      // // #pragma omp parallel for
      // for (int i = 0; i < cache_node_num; ++i) {
      //   int node_id = cache_node_idx_seq[i];
      //   LOG_DEBUG("copy node_id %d to", node_id);
      //   LOG_DEBUG("local_id %d", cache_node_hashmap[node_id]);
        
      //   for (int j = 0; j < feat_dim; ++j) {
      //     assert(cache_node_hashmap[node_id] < cache_node_num);
      //     LOG_DEBUG("copy feat dim %d", j);
      //     assert(cache_node_hashmap[node_id] * feat_dim + j < cache_node_num * feat_dim);
      //     assert(node_id * feat_dim + j < graph->gnnctx->l_v_num * feat_dim);
      //     local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] = gnndatum->local_feature[node_id * feat_dim + j];
      //     // local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] = 0;
      //   }
      //   LOG_DEBUG("copy done");
      // }
      // LOG_DEBUG("start trans to gpu");
      // move_data_in(dev_cache_feature, local_cache_feature_gather, 0, cache_node_num, feat_dim);
      // local_idx = (VertexId *)cudaMallocPinned(graph->vertices * sizeof(VertexId));
      // local_idx_cache = (VertexId *)cudaMallocPinned(graph->vertices * sizeof(VertexId));
      // dev_local_idx = (VertexId *)getDevicePointer(local_idx);
      // dev_local_idx_cache = (VertexId *)getDevicePointer(local_idx_cache);
    }
  }


void gater_cpu_cache_feature_and_trans_to_gpu() {
  int feat_dim = graph->gnnctx->layer_size[0];
  LOG_DEBUG("feat_dim %d", feat_dim);
  dev_cache_feature = (float *)cudaMallocGPU(cache_node_num * sizeof(float) * feat_dim);
  // gather_cache_feature, prepare trans to gpu
  LOG_DEBUG("start gather_cpu_cache_feature");
  float* local_cache_feature_gather = new float[cache_node_num * feat_dim];
  #pragma omp parallel for
  for (int i = 0; i < cache_node_num; ++i) {
    int node_id = cache_node_idx_seq[i];
    // LOG_DEBUG("copy node_id %d to", node_id);
    // LOG_DEBUG("local_id %d", cache_node_hashmap[node_id]);
    
    for (int j = 0; j < feat_dim; ++j) {
      assert(cache_node_hashmap[node_id] < cache_node_num);
      // LOG_DEBUG("copy feat dim %d", j);
      assert(cache_node_hashmap[node_id] * feat_dim + j < cache_node_num * feat_dim);
      assert(node_id * feat_dim + j < graph->gnnctx->l_v_num * feat_dim);
      local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] = gnndatum->local_feature[node_id * feat_dim + j];
      // local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] = 0;
    }
    // LOG_DEBUG("copy done");
  }
  LOG_DEBUG("start trans to gpu");
  move_data_in(dev_cache_feature, local_cache_feature_gather, 0, cache_node_num, feat_dim);
  local_idx = (VertexId *)cudaMallocPinned(graph->vertices * sizeof(VertexId));
  local_idx_cache = (VertexId *)cudaMallocPinned(graph->vertices * sizeof(VertexId));
  dev_local_idx = (VertexId *)getDevicePointer(local_idx);
  dev_local_idx_cache = (VertexId *)getDevicePointer(local_idx_cache);
}

void mark_cache_node(std::vector<int> &cache_nodes) {
  // init mask
  #pragma omp parallel for 
  for (int i = 0; i < graph->vertices; ++i) {
    cache_node_hashmap[i] = -1;
    assert(cache_node_hashmap[i] == -1);
  }  
  
  // mark cache nodes
  int tmp_idx = 0;
  for (int i = 0; i < cache_node_num; ++i) {
    // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
    cache_node_hashmap[cache_nodes[i]] = tmp_idx++;
  }
  LOG_DEBUG("cache_node_num %d tmp_idx %d", cache_node_num, tmp_idx);
  assert(cache_node_num == tmp_idx);
  
  // debug
  int cache_node_hashmap_num = 0;
  for (int i = 0; i < graph->vertices; ++i) {
    // LOG_DEBUG("cache_node_hashmap[%d] = %d", i, cache_node_hashmap[i]);
    cache_node_hashmap_num += cache_node_hashmap[i] != -1; // unsigned
  }
  LOG_DEBUG("cache_node_hashmap_num %d", cache_node_hashmap_num);
  assert(cache_node_hashmap_num == cache_node_num);
}
  
void cache_high_degree(std::vector<int> &node_idx) {
  std::sort(node_idx.begin(), node_idx.end(), [&](const int x, const int y) {
    return graph->out_degree_for_backward[x] > graph->out_degree_for_backward[y];
  });
  #pragma omp parallel for 
  for (int i = 1; i < graph->vertices; ++i) {
    assert(graph->out_degree_for_backward[node_idx[i]] <= graph->out_degree_for_backward[node_idx[i - 1]]);
  }
  mark_cache_node(node_idx);
}

void cache_sample(std::vector<int>& node_idx) {

  mark_cache_node(node_idx);
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
    double used_gpu_mem, total_gpu_mem;
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn(): gcn_gpu_mem %.3fM", used_gpu_mem);

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
    LOG_DEBUG("after gnndataum");
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
    }
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn() after read_feature: gcn_gpu_mem %.3fM", used_gpu_mem);
    // creating tensor to save Label and Mask
    if (graph->config->classes > 1) {
      gnndatum->registLabel(L_GT_C, gnndatum->local_label, gnndatum->gnnctx->l_v_num, graph->config->classes);
    } else {
      gnndatum->registLabel(L_GT_C);
    }
    gnndatum->registMask(MASK);
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn() after registlabel: gcn_gpu_mem %.3fM", used_gpu_mem);

    MASK_gpu = MASK.cuda();
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn() after maks.cuda(): gcn_gpu_mem %.3fM", used_gpu_mem);
    gnndatum->generate_gpu_data();
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn() after generate_gpu_data(): gcn_gpu_mem %.3fM", used_gpu_mem);

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
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn() after P->to(GPU): gcn_gpu_mem %.3fM", used_gpu_mem);

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

#pragma omp parallel for
        for (int i = 0; i < vertexs.size(); i++) {
          result_vector[i].resize(feature_size);
          memcpy(result_vector[i].data(), ntsVarBuffer + (vertexs[i] - start) * feature_size,
                 feature_size * sizeof(ValueType));
        }
        return result_vector;
      });
    }
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("init_nn() done: gcn_gpu_mem %.3fM", used_gpu_mem);

    gater_cpu_cache_feature_and_trans_to_gpu();
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
  std::tuple<float, double> Forward(Sampler* sampler, int type = 0) {
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
      shuffle_vec(sampler->sample_nids);
    }

    // node sampling
    sampler->zero_debug_time();
    double sample_cost = 0;
    double trans_feature_cost = 0;
    double gather_gpu_cache_cost = 0;
    double trans_label_cost = 0;
    double trans_graph_cost = 0;
    double gather_feature_time = 0;
    double gather_label_time = 0;
    // double epoch_trans_time = 0;
    int epoch_cache_hit = 0;
    int epoch_cache_all = 0;

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
    double epoch_time = -get_time();
    int batch_num = sampler->batch_nums;
    uint64_t compute_cnt = 0;

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

    double used_gpu_mem, total_gpu_mem;
    sampler->metis_batch_id = 0;
    int batch_id = 0;
    double forward_cost = 0;
    
    while (sampler->work_offset < sampler->work_range[1]) {
      // for (VertexId i = 0; i < sampler->batch_nums; ++i) {
      ctx->train();
      if (graph->config->run_time > 0 && gcn_run_time >= graph->config->run_time) {
        break;
      }

      forward_other_cost -= get_time();
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
      forward_other_cost += get_time();

      double one_batch_cost = -get_time();
      // LOG_DEBUG("batch id %d", i);
      double sample_one_cost = -get_time();

      sample_cost -= get_time();
      sampler->sample_one(graph->config->batch_type, ctx->is_train());
      sample_cost += get_time();
      sample_one_cost += get_time();

      auto ssg = sampler->subgraph;

      if (graph->config->mini_pull > 0) {  // generate csr structure for backward of pull mode
        generate_csr_time -= get_time();
        for (auto p : ssg->sampled_sgs) {
          convert_time -= get_time();
          p->generate_csr_from_csc();
          convert_time += get_time();
          debug_time -= get_time();
          // p->debug_generate_csr_from_csc();
          debug_time += get_time();
        }
        generate_csr_time += get_time();
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d pull gemnerate csr done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
        // total_gpu_mem);
      }

      if (type == 0 && graph->rtminfo->epoch >= 3) train_sample_time += sample_cost;

      double trans_graph_gpu_cost = -get_time();
      ssg->trans_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      trans_graph_gpu_cost += get_time();
      // LOG_DEBUG("batch %d tarns_to_gpu done", i);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d tarns_to_gpu done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem);

      // rpc.keep_running();

      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d get feature done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem); LOG_DEBUG("load_feature done");

      //////////////////////////////////////////////gathter feawture/////////////////////////////////////////////////////
      
      if (graph->config->cache_rate <= 0) {
        trans_feature_cost -= get_time();
        sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
        trans_feature_cost += get_time();
      } else {
        // LOG_DEBUG("start load_farture_gpu_cache");
        // trans_feature_cost -= get_time();
        auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(X[0], gnndatum->dev_local_feature, dev_cache_feature, local_idx, local_idx_cache, cache_node_hashmap,
                                        dev_local_idx, dev_local_idx_cache, dev_cache_node_hashmap);
        // trans_feature_cost += get_time();
        trans_feature_cost += trans_feature_tmp;
        gather_gpu_cache_cost += gather_gpu_cache_tmp;
        // LOG_DEBUG("start laod feature");
        one_batch_cost += get_time();
        epoch_cache_all += ssg->sampled_sgs[0]->src().size(); 
        for (auto &it : ssg->sampled_sgs[0]->src()) {
          if (cache_node_hashmap[it] != -1) {
            epoch_cache_hit++;
          }
        }
        one_batch_cost -= get_time();
      }


      // double gather_feature_cost = -get_time();
      // if (hosts > 1) {
      //   X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= 3) rpc_comm_time += tmp_time;
      // } else {
      //   X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
      // }
      // gather_feature_cost += get_time();
      // double trans_feature_gpu_cost = -get_time();
      // X[0] = X[0].cuda().set_requires_grad(true);
      // trans_feature_gpu_cost += get_time();
      // gather_feature_time += gather_feature_cost;

      // LOG_DEBUG("after laod feature");
      ///////////////////////////////////////////////////////////////////////////////////////////////////

      /////////////////////////////////////////////////target_lab//////////////////////////////////////////////////
      trans_label_cost -= get_time();
      sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      trans_label_cost += get_time();

      // double gather_label_cost = -get_time();
      // if (hosts > 1) {
      //   target_lab = nts::op::get_label_from_global(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= 3) rpc_comm_time += tmp_time;
      // } else {
      //   target_lab = nts::op::get_label(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
      // }
      // gather_label_cost += get_time();

      // double trans_label_gpu_cost = -get_time();
      // target_lab = target_lab.cuda();
      // trans_label_gpu_cost += get_time();
      // gather_label_time += gather_label_cost;

      // LOG_DEBUG("after laod label");
      ///////////////////////////////////////////////////////////////////////////////////////////////////

      // epoch_trans_time += trans_feature_gpu_cost + trans_label_gpu_cost + trans_graph_gpu_cost;
      trans_graph_cost += trans_graph_gpu_cost;
      
      
      // LOG_DEBUG("load_label done");
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d get label done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem);

      // print label
      // long* local_label_buffer = nullptr;
      // auto classes = graph->config->classes;
      // if (classes > 1) {local_label_buffer = graph->Nts->getWritableBuffer2d<long>(target_lab,
      // torch::DeviceType::CUDA);} else {local_label_buffer = graph->Nts->getWritableBuffer1d<long>(target_lab,
      // torch::DeviceType::CUDA);} long* tmp = new long[classes * graph->config->batch_size]; cudaMemcpy ( tmp,
      // local_label_buffer, classes * graph->config->batch_size * sizeof(long), cudaMemcpyDeviceToHost); for (int i =
      // 0; i < 4; ++i) {
      //   for (int j = 0; j < classes; ++j) {
      //     printf("%d ", tmp[i * classes + j]);
      //   }printf("\n");
      // }
      train_cost -= get_time();

      for (int l = 0; l < layers; l++) {  // forward
        // LOG_DEBUG("start compute layer %d", l);
        graph->rtminfo->curr_layer = l;
        forward_graph_cost -= get_time();
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l]);
        // LOG_DEBUG("after return output_tensor ptr %p", Y_i.data_ptr());
        forward_graph_cost += get_time();
        // LOG_DEBUG("  batch %d layer %d graph compute done", i, l);
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d layer %d graph compute done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, l,
        // used_gpu_mem, total_gpu_mem);

        forward_nn_cost -= get_time();
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
        forward_nn_cost += get_time();
        // LOG_DEBUG("  batch %d layer %d nn compute done", i, l);
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d layer %d nn compute done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, l,
        // used_gpu_mem, total_gpu_mem);
      }

      forward_loss_cost -= get_time();
      // std::cout << "X.size " << X[layers].size(0) << " label size " << target_lab.size(0) << std::endl;
      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();
      forward_loss_cost += get_time();
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d loss done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem, total_gpu_mem);

      // LOG_DEBUG("loss done %.3f", loss_epoch);

      if (ctx->training == true) {
        forward_append_cost -= get_time();
        ctx->appendNNOp(X[layers], loss_);
        forward_append_cost += get_time();

        // LOG_DEBUG("start backward done");
        backward_cost -= get_time();
        ctx->b_nn_time = 0;
        ctx->self_backward(false);
        // c10::cuda::CUDACachingAllocator::emptyCache();
        backward_nn_time += ctx->b_nn_time;
        backward_cost += get_time();
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d backward done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
        // total_gpu_mem);

        update_cost -= get_time();
        Update();
        update_cost += get_time();
        // LOG_DEBUG("batch %d update done", i);
      }

      train_cost += get_time();
      one_batch_cost += get_time();

      forward_acc_cost -= get_time();
      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      forward_acc_cost += get_time();
      // LOG_DEBUG("batch %d acc compute done", i);
      // LOG_DEBUG("batch %d backward done", i);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d acc done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem, total_gpu_mem);

      sampler->reverse_sgs();
      // printf("gcn_run_time %.3f one_batfh_cost %.3f ", gcn_run_time, one_batch_cost);
      if (graph->rtminfo->epoch >= 3) {
        gcn_run_time += one_batch_cost;
      }
      forward_cost += one_batch_cost;

      // printf("gcn_run_time %.3f\n", gcn_run_time);

      // LOG_DEBUG("start evalforward");

      /////////////////////////////////////////////////////////////////////
      // do evaluation after one batch
      // ctx->eval();
      // double eval_cost = -get_time();
      // float val_acc = EvalForward(eval_sampler, 1);
      // eval_cost += get_time();
      /////////////////////////////////////////////////////////////////////

      // LOG_INFO("Epoch %03d batch %03d eval_acc %.3f train_time %.3lf eval_time %.3lf run_time %.3lf",
      //          graph->rtminfo->epoch, batch_id, val_acc, one_batch_cost, eval_cost, gcn_run_time);
      ///////////////////////////////////////////////////
      // if (val_acc > best_val_acc) {
      //   LOG_DEBUG("val_acc %.3f best_val_acc %.3f", val_acc, best_val_acc);
      //   best_val_acc = val_acc;
      //   LOG_DEBUG("save_W to /home/hdd/sanzo/neutron-sanzo/saved_modules");
      //   for (int i = 0; i < layers; ++i) {
      //     P[i]->save_W("/home/hdd/sanzo/neutron-sanzo/saved_modules", i);
      //     // P[i]->load_W("/home/hdd/sanzo/neutron-sanzo/saved_modules", i);
      //   }
      // }
      ///////////////////////////////////////////////////
      batch_id++;
    }
    // LOG_DEBUG("sampler worker_offset %d range %d", sampler->work_offset, sampler->work_range[1]);
    assert(sampler->work_offset == sampler->work_range[1]);
    loss_epoch /= sampler->batch_nums;

    // for (int i = 0; i < layers; ++i) {
    //   sampler->subgraph->sampled_sgs[i]->print_debug_time();
    // }
    // LOG_DEBUG("all train nodes: %d, all sample nodes: %d", sampler->sample_nids.size(), st.size());

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

    epoch_time += get_time();
    // LOG_INFO("sample_cost %.3f", sample_cost);
    // LOG_INFO("generate_csr %.3f, convert %.3f debug %.3f", generate_csr_time, convert_time, debug_time);
    // LOG_INFO("trans_graph_cost %.3f", trans_graph_cost);
    // if (type == 0) {
    //   LOG_INFO("trainning cost %.3f", train_cost);
    // } else {
    //   LOG_INFO("evaluation cost %.3f", train_cost);
    // }
    // LOG_INFO("  trans_feature/lable (%.3f/%.3f)", trans_feature_cost, trans_label_cost);
    // LOG_INFO("  forward: graph %.3f, nn %.3f, loss %.3f, acc %.3f, update %.3f", forward_graph_cost, forward_nn_cost,
    //          forward_loss_cost, forward_acc_cost, update_cost);
    // LOG_INFO("  backward cost %.3f, b_nn_time %.3f", backward_cost, backward_nn_time);
    // if (type == 0) {
    //   get_gpu_mem(used_gpu_mem, total_gpu_mem);
    //   LOG_INFO("train_epoch %d cost %.3f, compute_cnt %lld, gpu mem:  (%.0fM/%.0fM)", graph->rtminfo->epoch,
    //   epoch_time,
    //            compute_cnt, used_gpu_mem, total_gpu_mem);
    // } else {
    //   get_gpu_mem(used_gpu_mem, total_gpu_mem);
    //   LOG_INFO("eval_epoch %d cost %.3f, compute_cnt %lld, gpu mem: (%.0fM/%.0fM)", graph->rtminfo->epoch,
    //   epoch_time,
    //            compute_cnt, used_gpu_mem, total_gpu_mem);
    // }
    // printf("\n");

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

    // return acc;
    // LOG_DEBUG("trans_cost %.3f\n", trans_feature_cost + trans_label_cost + trans_graph_cost);
    // LOG_DEBUG("epoch_cache_hit %d", epoch_cache_hit);
    double epoch_trans_time = trans_feature_cost + trans_label_cost + trans_graph_cost;
    double epoch_trans_time_cache = (trans_feature_cost / epoch_cache_all * (epoch_cache_all - epoch_cache_hit)) + trans_label_cost + trans_graph_cost;
    double epoch_gather_time = gather_feature_time + gather_label_time;
    double epoch_gather_time_cache = (gather_feature_time / epoch_cache_all * (epoch_cache_all - epoch_cache_hit)) + gather_label_time;
    if (graph->rtminfo->epoch >= 3) {
        gcn_trans_cost += epoch_trans_time;
        gcn_trans_cost_cache += epoch_trans_time_cache;
        gcn_gather_cost += epoch_gather_time;
        gcn_gather_cost_cache += epoch_gather_time_cache;
    }

    // double used_gpu_mem, total_gpu_mem;
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_INFO("Epoch %03d epoch_train_time %.3f trans_cost %.3f(feat %.3f,label %.3f, graph %.3f) gather_gpu_cache_cost %.3f gather_cost %.3f gpu_mem %.3fM", 
                 graph->rtminfo->epoch, forward_cost, epoch_trans_time, trans_feature_cost, trans_label_cost, trans_graph_cost, gather_gpu_cache_cost ,epoch_gather_time, used_gpu_mem);
    // double used_gpu_mem, total_gpu_mem;
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_INFO("Epoch %03d epoch_train_time_cache %.3f trans_cost_cache %.3f gather_cost_cache %.3f gpu_mem_cache %.3fM cache_mem %.3fM", 
    //              graph->rtminfo->epoch, forward_cost - (epoch_trans_time - epoch_trans_time_cache) - (epoch_gather_time - epoch_gather_time_cache),
    //               epoch_trans_time_cache, epoch_gather_time_cache, used_gpu_mem, 1.0 * cache_node_num * graph->gnnctx->layer_size[0]*4/1024/1024);

      // epoch_trans_time += trans_feature_gpu_cost + trans_label_gpu_cost + trans_graph_gpu_cost;

    // LOG_DEBUG("forward_cost %.3f gather_cost %.3f trans_cost %.3f", forward_cost, epoch_gather_time, epoch_trans_time);
    // return {acc, forward_cost, trans_cost};
    return {acc, forward_cost};
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
      shuffle_vec(sampler->sample_nids);
    }

    // node sampling
    sampler->zero_debug_time();
    double sample_cost = 0;
    double trans_feature_cost = 0;
    double trans_label_cost = 0;
    double forward_nn_cost = 0;
    double forward_acc_cost = 0;
    double forward_graph_cost = 0;
    double forward_loss_cost = 0;
    double forward_other_cost = 0;
    double forward_append_cost = 0;
    double backward_nn_time = 0;
    double update_cost = 0;
    double trans_graph_cost = 0;

    double train_cost = 0;
    double inner_cost = 0;
    double generate_csr_time = 0;
    double convert_time = 0;
    double debug_time = 0;
    double epoch_time = -get_time();
    int batch_num = sampler->batch_nums;
    uint64_t compute_cnt = 0;

    if (hosts > 1) {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
    } else {
      max_batch_num = batch_num;
    }

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    // NtsVar target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes},
    // torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }

    // LOG_DEBUG("epoch %d start compute", graph->rtminfo->epoch);
    double used_gpu_mem, total_gpu_mem;

    // for (int i = 0; i < layers; ++i) {
    //   sampler->subgraph->sampled_sgs[i]->zero_debug_time();
    // }
    // std::unordered_set<VertexId> st;
    sampler->metis_batch_id = 0;
    // LOG_DEBUG("sampler has %d batchs", sampler->batch_nums);
    // for (VertexId i = 0; i < sampler->batch_nums; ++i) {
    int i = -1;
    // LOG_DEBUG("work_offset %d work_range %d", sampler->work_offset, sampler->work_range[1]);
    while (sampler->work_offset < sampler->work_range[1]) {
      ++i;
      // if (graph->config->run_time > 0 && get_time() - gcn_start_time >= graph->config->run_time) {
      //   break;
      // }

      // LOG_DEBUG("batch id %d", i);
      double sample_one_cost = -get_time();
      sample_cost -= get_time();
      sampler->sample_one(graph->config->batch_type, ctx->is_train());
      sample_cost += get_time();
      sample_one_cost += get_time();

      compute_cnt += sampler->get_compute_cnt();
      // LOG_DEBUG("sample one done");
      // continue;

      // LOG_DEBUG("epoch %d batch %d, train_nodes %d", graph->rtminfo->epoch, i,
      // sampler->subgraph->sampled_sgs.back()->v_size); sampler->print_batch_nodes();

      ////////////////// check sampler
      // sampler->insert_batch_nodes(st);

      // LOG_DEBUG("sample one cost %.3f", sample_one_cost);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d sample_one done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem);

      auto ssg = sampler->subgraph;

      if (graph->config->mini_pull > 0) {  // generate csr structure for backward of pull mode
        generate_csr_time -= get_time();
        for (auto p : ssg->sampled_sgs) {
          convert_time -= get_time();
          p->generate_csr_from_csc();
          convert_time += get_time();
          debug_time -= get_time();
          // p->debug_generate_csr_from_csc();
          debug_time += get_time();
        }
        generate_csr_time += get_time();
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d pull gemnerate csr done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
        // total_gpu_mem);
      }

      if (type == 0 && graph->rtminfo->epoch >= 3) train_sample_time += sample_cost;

      // std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
      trans_graph_cost -= get_time();
      ssg->trans_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      trans_graph_cost += get_time();
      // LOG_DEBUG("batch %d tarns_to_gpu done", i);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d tarns_to_gpu done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem);

      train_cost -= get_time();
      forward_other_cost -= get_time();
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute

      forward_other_cost += get_time();
      // rpc.keep_running();
      trans_feature_cost -= get_time();
      sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      trans_feature_cost += get_time();
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d get feature done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem); LOG_DEBUG("load_feature done");

      // if (hosts > 1) {
      //   X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src(), F, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= 3) rpc_comm_time += tmp_time;
      // } else {
      //   X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src(), ssg->sampled_sgs[0]->src_size, F, graph);
      // }

      trans_label_cost -= get_time();
      sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      trans_label_cost += get_time();
      // LOG_DEBUG("load_label done");
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d get label done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
      // total_gpu_mem);

      // print label
      // long* local_label_buffer = nullptr;
      // auto classes = graph->config->classes;
      // if (classes > 1) {local_label_buffer = graph->Nts->getWritableBuffer2d<long>(target_lab,
      // torch::DeviceType::CUDA);} else {local_label_buffer = graph->Nts->getWritableBuffer1d<long>(target_lab,
      // torch::DeviceType::CUDA);} long* tmp = new long[classes * graph->config->batch_size]; cudaMemcpy ( tmp,
      // local_label_buffer, classes * graph->config->batch_size * sizeof(long), cudaMemcpyDeviceToHost); for (int i =
      // 0; i < 4; ++i) {
      //   for (int j = 0; j < classes; ++j) {
      //     printf("%d ", tmp[i * classes + j]);
      //   }printf("\n");
      // }

      for (int l = 0; l < layers; l++) {  // forward
        // LOG_DEBUG("start compute layer %d", l);
        graph->rtminfo->curr_layer = l;
        forward_graph_cost -= get_time();
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l]);
        // LOG_DEBUG("after return output_tensor ptr %p", Y_i.data_ptr());
        forward_graph_cost += get_time();
        // LOG_DEBUG("  batch %d layer %d graph compute done", i, l);
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d layer %d graph compute done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, l,
        // used_gpu_mem, total_gpu_mem);

        forward_nn_cost -= get_time();
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
        forward_nn_cost += get_time();
        // LOG_DEBUG("  batch %d layer %d nn compute done", i, l);
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d layer %d nn compute done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, l,
        // used_gpu_mem, total_gpu_mem);
      }

      forward_loss_cost -= get_time();
      // std::cout << "X.size " << X[layers].size(0) << " label size " << target_lab.size(0) << std::endl;
      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();
      forward_loss_cost += get_time();
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d loss done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem, total_gpu_mem);

      // LOG_DEBUG("loss done %.3f", loss_epoch);

      if (ctx->training == true) {
        forward_append_cost -= get_time();
        ctx->appendNNOp(X[layers], loss_);
        forward_append_cost += get_time();

        // LOG_DEBUG("start backward done");
        backward_cost -= get_time();
        ctx->b_nn_time = 0;
        ctx->self_backward(false);
        // c10::cuda::CUDACachingAllocator::emptyCache();
        backward_nn_time += ctx->b_nn_time;
        backward_cost += get_time();
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("epoch %d batch %d backward done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem,
        // total_gpu_mem);

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
      // LOG_DEBUG("batch %d backward done", i);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("epoch %d batch %d acc done, (%.0fM/%.0fM)", graph->rtminfo->epoch, i, used_gpu_mem, total_gpu_mem);

      train_cost += get_time();
      // std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
      sampler->reverse_sgs();
    }
    // LOG_DEBUG("sampler worker_offset %d range %d", sampler->work_offset, sampler->work_range[1]);
    assert(sampler->work_offset == sampler->work_range[1]);
    loss_epoch /= sampler->batch_nums;

    // for (int i = 0; i < layers; ++i) {
    //   sampler->subgraph->sampled_sgs[i]->print_debug_time();
    // }

    // LOG_DEBUG("all train nodes: %d, all sample nodes: %d", sampler->sample_nids.size(), st.size());

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

    epoch_time += get_time();
    // LOG_INFO("sample_cost %.3f", sample_cost);
    // LOG_INFO("generate_csr %.3f, convert %.3f debug %.3f", generate_csr_time, convert_time, debug_time);
    // LOG_INFO("trans_graph_cost %.3f", trans_graph_cost);

    // if (type == 0) {
    //   LOG_INFO("trainning cost %.3f", train_cost);
    // } else {
    //   LOG_INFO("evaluation cost %.3f", train_cost);
    // }
    // LOG_INFO("  trans_feature/lable (%.3f/%.3f)", trans_feature_cost, trans_label_cost);
    // LOG_INFO("  forward: graph %.3f, nn %.3f, loss %.3f, acc %.3f, update %.3f", forward_graph_cost, forward_nn_cost,
    //          forward_loss_cost, forward_acc_cost, update_cost);
    // LOG_INFO("  backward cost %.3f, b_nn_time %.3f", backward_cost, backward_nn_time);

    // if (type == 0) {
    //   get_gpu_mem(used_gpu_mem, total_gpu_mem);
    //   LOG_INFO("train_epoch %d cost %.3f, compute_cnt %lld, gpu mem:  (%.0fM/%.0fM)", graph->rtminfo->epoch,
    //   epoch_time,
    //            compute_cnt, used_gpu_mem, total_gpu_mem);
    // } else {
    //   get_gpu_mem(used_gpu_mem, total_gpu_mem);
    //   LOG_INFO("eval_epoch %d cost %.3f, compute_cnt %lld, gpu mem: (%.0fM/%.0fM)\n", graph->rtminfo->epoch,
    //   epoch_time,
    //            compute_cnt, used_gpu_mem, total_gpu_mem);
    // }

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
    double used_gpu_mem, total_gpu_mem;
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("run(): gcn_gpu_mem %.3fM", used_gpu_mem);



    gcn_run_time = gcn_gather_cost = gcn_trans_cost = 0;
    // epoch_cache_hit = 0;
   

    double pre_time = -get_time();
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    BatchType batch_type = graph->config->batch_type;
    assert(best_val_acc == 0);
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
    eval_sampler = new Sampler(fully_rep_graph, val_nids, true);   // true mean full batch
    // eval_sampler->update_fanout(-1);                               // val not sample
    test_sampler = new Sampler(fully_rep_graph, test_nids, true);  // true mean full batch
    // LOG_DEBUG("samper done");

    if (batch_type == METIS) {
      // void ClusterGCNSample(int layers, int batch_size, int partition_num, std::string objtype = "cut");
      // void init_graph(std::vector<VertexId> &sample_nids);

      // if (metis_partition_id.empty()) {
      // LOG_DEBUG("start update_graph");
      // fully_rep_graph->update_graph(train_nids);
      // LOG_DEBUG("after update_graph");

      int partition_num = (train_nids.size() + graph->config->batch_size - 1) / graph->config->batch_size;
      std::vector<VertexId> metis_partition_id;
      std::vector<VertexId> metis_partition_offset;

      double metis_time = -get_time();
      MetisPartitionGraph(fully_rep_graph, partition_num, "cut", metis_partition_id, metis_partition_offset);
      metis_time += get_time();
      LOG_DEBUG("metis partition cost %.3f", metis_time);

      // fully_rep_graph->back_to_global();
      LOG_DEBUG("metis_partition_id %d, train_nids %d", metis_partition_id.size(), train_nids.size());
      // assert(metis_partition_id.size() == train_nids.size());
      assert(partition_num + 1 == metis_partition_offset.size());

      // LOG_DEBUG("metis partiton offset: ");
      // for (int i = 0; i < partition_num; ++i) {
      //   printf("part id: %d, %d\n", i, metis_partition_offset[i + 1] - metis_partition_offset[i]);
      // }
      // LOG_DEBUG("metis partiton id: ");
      // for (auto id : metis_partition_id) {
      //   printf("%d ", id);
      // }printf("\n");

      // std::sort(metis_partition_id.begin(), metis_partition_id.end());
      // std::sort(train_nids.begin(), train_nids.end());
      // for (int i = 0; i < train_nids.size(); ++i) {
      //   assert(metis_partition_id[i] == train_nids[i]);
      // }
      // assert(false);
      // }
      train_sampler->update_metis_data(metis_partition_id, metis_partition_offset);
      train_sampler->update_batch_nums();
    }

    LOG_INFO("label rate: %.3f, train/val/test: (%d/%d/%d) (%.3f/%.3f/%.3f)",
             1.0 * (train_nids.size() + val_nids.size() + test_nids.size()) / graph->vertices, train_nids.size(),
             val_nids.size(), test_nids.size(), train_nids.size() * 1.0 / graph->vertices,
             val_nids.size() * 1.0 / graph->vertices, test_nids.size() * 1.0 / graph->vertices);

    pre_time += get_time();

    double train_time = 0;
    double val_time = 0;
    double test_time = 0;
    double run_time = -get_time();
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

      ctx->train();
      // if (i_i >= graph->config->time_skip) train_time -= get_time();
      // float train_acc = Forward(train_sampler, 0);
      // auto [train_acc, epoch_time, trans_cost, cache_hit] = Forward(train_sampler, 0);
      auto [train_acc, epoch_time] = Forward(train_sampler, 0);
      float train_loss = loss_epoch;

      // update batch size after the whole epoch training
      if (graph->config->batch_switch_time > 0) {
        train_sampler->update_batch_size_from_time(gcn_run_time);
      }

      if (graph->config->sample_switch_time > 0) {
        train_sampler->update_sample_rate_from_time(gcn_run_time);
      }

      float val_acc = EvalForward(eval_sampler, 1);
      LOG_DEBUG("Epoch %03d val_acc %.3f", i_i, val_acc);
      float test_acc = 0;
      if (graph->partition_id == 0) {
        // double used_gpu_mem, total_gpu_mem;
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_INFO("Epoch %03d epoch_train_time %.3f trans_cost %.3f gpu_mem %.3fM", 
        //          i_i, epoch_time, trans_cost, used_gpu_mem);
      }
    }
    // double used_gpu_mem, total_gpu_mem;
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("div %d epoch", graph->config->epochs - 3);
    gcn_trans_cost /= (graph->config->epochs - 3);
    gcn_run_time /= (graph->config->epochs - 3);
    gcn_gather_cost /= (graph->config->epochs - 3);

    LOG_DEBUG("dataset %s gcn_cache_rate %.3f gcn_batch_size %d gcn_run_time %.3f gcn_gpu_run_mem %.3fM gcn_gather_cost %.3f (%.3f) gcn_trans_cost %.3f (%.3f), all_cost %.3f", 
              graph->config->dataset_name.c_str(), graph->config->cache_rate, graph->config->batch_size, gcn_run_time, used_gpu_mem, 
              gcn_gather_cost, gcn_gather_cost / gcn_run_time, 
              gcn_trans_cost, gcn_trans_cost / gcn_run_time,
              (gcn_gather_cost + gcn_trans_cost) / gcn_run_time);
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    gcn_run_time -= gcn_gather_cost - gcn_gather_cost_cache + gcn_trans_cost - gcn_trans_cost_cache;
    // LOG_DEBUG("dataset %s gcn_batch_size_cache %d gcn_run_time_cache %.3f gcn_gpu_mem_cache %.3fM gcn_cache_mem %.3fM gcn_gather_cost_cache %.3f (%.3f) gcn_trans_cost_cache %.3f (%.3f), all_cost %.3f", 
    //         graph->config->dataset_name.c_str(), graph->config->batch_size, gcn_run_time, used_gpu_mem, 1.0 * cache_node_num * graph->gnnctx->layer_size[0]*4/1024/1024, 
    //         gcn_gather_cost_cache, gcn_gather_cost_cache / gcn_run_time, 
    //         gcn_trans_cost_cache, gcn_trans_cost_cache / gcn_run_time,
    //         (gcn_gather_cost_cache + gcn_trans_cost_cache) / gcn_run_time);       

    // double comm_time = mpi_comm_time + rpc_comm_time + rpc_wait_time;
    // double compute_time = train_time - comm_time;
    // printf(
    //     "train TIME(%d) sample %.3f compute_time %.3f comm_time %.3f mpi_comm %.3f rpc_comm %.3f rpc_wait_time
    //     %.3f\n", graph->partition_id, train_sample_time, compute_time, comm_time, mpi_comm_time, rpc_comm_time,
    //     rpc_wait_time);
    // printf("train/val/test time: %.3f, %.3f, %.3f\n", train_time, val_time, test_time);

    // if (hosts > 1) rpc->exit();
    // run_time += get_time();
    // printf("run(): pre_time %.3fs runtime: %.3fs\n", pre_time, run_time);

    // if (graph->partition_id == 0)
    //   printf("Avg epoch train_time %.3f val_time %.3f test_time %.3f\n",
    //          train_time / (iterations - graph->config->time_skip), val_time / (iterations -
    //          graph->config->time_skip), test_time / (iterations - graph->config->time_skip));
    // delete active;
    return best_val_acc;
  }

  // void batch_size_mem_run_time() {
  //   std::vector<int> batch_size_list {512, 1024, 512, 1024};
  //   for (auto it : batch_size_list) {
  //     graph->config->batch_size = it;
  //     run();
  //   }
  // }
};

