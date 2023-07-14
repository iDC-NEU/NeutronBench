#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"
#include <c10/cuda/CUDACachingAllocator.h>

class GCN_GPU_NEIGHBOR_partition_impl {
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
  float f1_epoch = 0;
  Sampler* train_sampler = nullptr;
  Sampler* eval_sampler = nullptr;
  Sampler* test_sampler = nullptr;
  // double gcn_start_time = 0;
  double gcn_run_time;


  int threads;

  VertexId *local_idx, *local_idx_cache, *dev_local_idx, *dev_local_idx_cache;
  Cuda_Stream* cuda_stream;

  std::vector<at::cuda::CUDAStream> torch_stream;




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
  // std::unordered_map<std::string, std::vector<int>> batch_size_mp;
  // std::vector<int> batch_size_vec;

  std::vector<int> cache_node_idx_seq;
  // std::unordered_set<int> cache_node_hashmap;
  // std::vector<int> cache_node_hashmap;
  VertexId* cache_node_hashmap;
  VertexId* dev_cache_node_hashmap;
  int cache_node_num = 0;


  GCN_GPU_NEIGHBOR_partition_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
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

    pipelines = graph->config->pipelines;
    pipelines = std::max(1, pipelines);
    torch_stream.clear();

    gcn_run_time = 0;
    gcn_gather_time = 0;
    gcn_sample_time = 0;
    gcn_trans_time = 0;
    gcn_train_time = 0;
    gcn_cache_hit_rate = 0;
    gcn_trans_memory = 0;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);

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
      // P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], learn_rate, weight_decay));
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


  void test_depcache() {
    int partition_num = graph->config->part_num;
    assert (partition_num > 0);
    std::vector<std::unordered_set<VertexId>> partition_nodes;
    std::vector<std::vector<VertexId>> train_id;
    partition_nodes.resize(partition_num);
    train_id.resize(partition_num);

    std::string part_file = "/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result_part_info/" + graph->config->part_algo 
                            + "-" + graph->config->dataset_name + "-part" + to_string(graph->config->part_num) + "-info.txt";
                            
    std::ifstream fin;
    std::cout << "read_file " << part_file << std::endl;
    fin.open(part_file.c_str(), std::ios::in);
    if (!fin)
    {
        std::cout << "cannot open file" << std::endl;
        exit(1);
        return;
    }
    int partition_id = 0;
    int vertices = 0;
    std::string line;
    int flag = 0;
    while(getline(fin, line))
    {
        std::stringstream ss(line);
        ss >> partition_id;
        ss >> vertices;
        VertexId vertex_id;
        while(ss >> vertex_id)
        {
          // std::cout << vertex_id << " ";
          partition_nodes[partition_id].insert(vertex_id);

          int type = gnndatum->local_mask[vertex_id];
          if(type == 0)
          {
            train_id[partition_id].push_back(vertex_id);
          }
        }
    }

    std::cout << "partition_nodes: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << partition_nodes[i].size() << " ";;
    } std::cout << std::endl;

    std::cout << "train_nodes: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << train_id[i].size() << " ";;
    } std::cout << std::endl;

    std::cout << "read " << part_file << " done." << std::endl;


    std::vector<Sampler*> train_sampler;
    for(int i = 0; i < partition_num; i++)
    {
      train_sampler.push_back(new Sampler(fully_rep_graph, train_id[i], pipelines, false) );
      shuffle_vec_seed(train_sampler[i]->sample_nids);
    }

    //统计
    std::vector<std::set<VertexId>> vertex_local;
    std::vector<std::set<VertexId>> vertex_remote;
    vertex_local.resize(partition_num);
    vertex_remote.resize(partition_num);
    
    

    for(int part = 0; part < partition_num; part++)
    {
      long local_node_num = 0, remote_node_num = 0;
      long local_edge_num = 0, remote_edge_num = 0;

      int batch_id = 0;
      while (train_sampler[part]->work_offset < train_sampler[part]->work_range[1]) {
        vertex_local[part].clear();
        vertex_remote[part].clear();
        auto ssg = train_sampler[part]->subgraph;
        train_sampler[part]->sample_one(ssg, graph->config->batch_type, ctx->is_train());
        //////// local/remote node_num
        for(int l = 0; l < layers; l++) {
        // std::cout << "batch_id: " << batch_id << " " << ssg->sampled_sgs[l]->src().size() << " " << ssg->sampled_sgs[l]->dst().size() << std::endl;
          for (auto read_v : ssg->sampled_sgs[l]->src()) {
            if(partition_nodes[part].find(read_v) != partition_nodes[part].end()) {
              vertex_local[part].insert(read_v);
            } else {
              vertex_remote[part].insert(read_v);
            }
          }
        }
        for (auto read_v : ssg->sampled_sgs[layers-1]->dst()) {
            if(partition_nodes[part].find(read_v) != partition_nodes[part].end()) {
              vertex_local[part].insert(read_v);
            } else {
              vertex_remote[part].insert(read_v);
            }
        }
        local_node_num += vertex_local[part].size();
        remote_node_num += vertex_remote[part].size();
        ////////////////////////////////

        //////// local/remote node_num
        // std::cout << "batch " <<  batch_id << " edges " << ssg->sampled_sgs[layers-1]->r_i().size() << std::endl;
        for (int l = 0; l < layers; ++l) {
          for (auto read_v : ssg->sampled_sgs[l]->r_i()) {
            // std::cout << read_v << " " << ssg->sampled_sgs[l]->src(read_v) << std::endl;
            read_v = ssg->sampled_sgs[l]->src(read_v);
            if(partition_nodes[part].find(read_v) != partition_nodes[part].end()) {
              local_edge_num++;
            } else {
              remote_edge_num++;
            }
          }
        }
        ////////////////////////////////

        train_sampler[part]->reverse_sgs();
        batch_id++;
      }
      assert(train_sampler[part]->work_offset == train_sampler[part]->work_range[1]);
      train_sampler[part]->restart();

      std::cout << "part[" << part << "] local_node_num: " << local_node_num << " remote_node_num: " << remote_node_num << " local/remote: " << 1.0 * local_node_num / remote_node_num << std::endl;
      std::cout << "part[" << part << "] local_edge_num: " << local_edge_num << " remote_edge_num: " << remote_edge_num << " local/remote: " << 1.0 * local_edge_num / remote_edge_num << std::endl;
      std::cout << std::endl;
    }
  }
};