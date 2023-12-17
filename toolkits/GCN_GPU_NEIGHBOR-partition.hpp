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
  int pipelines;

  int threads;

  VertexId *local_idx, *local_idx_cache, *dev_local_idx, *dev_local_idx_cache;
  Cuda_Stream* cuda_stream;

  std::vector<at::cuda::CUDAStream> torch_stream;
  // int batch_size_switch_idx = 0;

  std::vector<std::unordered_set<VertexId>> partition_nodes;
  std::vector<int> belongs;
  std::vector<std::vector<VertexId>> train_id;

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
  int partition_num = 0;

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
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    partition_num = graph->config->part_num;
    belongs.resize(graph->config->vertices, -1);
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

  
  
  void read_partition_result() {
    assert (partition_num > 0);
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
    }
    int partition_id = 0;
    VertexId vertices = 0;
    VertexId vertex_id;
    std::string line;
    while(getline(fin, line))
    {
        std::stringstream ss(line);
        ss >> partition_id >> vertices;
        while(ss >> vertex_id)
        {
          partition_nodes[partition_id].insert(vertex_id);
          belongs[vertex_id] = partition_id;
          // std::cout << partition_id << " " << vertex_id << std::endl;
          if(gnndatum->local_mask[vertex_id] == 0)
          {
            train_id[partition_id].push_back(vertex_id);
          }
        }
        assert(partition_nodes[partition_id].size() == vertices);
    }
    if (graph->config->part_algo != "pagraph") {
      for (int i = 0; i < graph->config->vertices; ++i) {
        if (belongs[i] < 0 || belongs[i] > partition_num) {
          std::cout << "assert false " << i << " " << belongs[i] << std::endl;
        }
        assert(belongs[i] >= 0 && belongs[i] < partition_num);
      }
    }
    std::cout << "read " << part_file << " done." << std::endl;
    fin.close();

    // if (graph->config->part_algo == "pagraph") {
    //   for (int i = 0; i < partition_num; ++i) {
    //     train_id.clear();
    //   }
    //   part_file = "/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result_part_info/" + graph->config->part_algo 
    //                           + "-" + graph->config->dataset_name + "-part" + to_string(graph->config->part_num) + "-traininfo.txt";
    //   std::cout << "read_file " << part_file << std::endl;
    //   std::ifstream fin2;
    //   fin2.open(part_file.c_str(), std::ios::in);
    //   if (!fin2)
    //   {
    //       std::cout << "cannot open file" << std::endl;
    //       exit(1);
    //   }
    //   int partition_id = 0;
    //   VertexId vertices = 0;
    //   VertexId vertex_id;
    //   std::string line;
    //   while(getline(fin2, line))
    //   {
    //       std::stringstream ss(line);
    //       ss >> partition_id >> vertices;
    //       while(ss >> vertex_id)
    //       {
    //         assert(vertex_id < graph->config->vertices);
    //         assert(gnndatum->local_mask[vertex_id] == 0);
    //         train_id[partition_id].push_back(vertex_id);
    //       }
    //   }
    //   fin2.close();
    //   std::cout << "read " << part_file << " done." << std::endl;
    // }
  
  }


  void read_pagraph_train_partition_result() {
    assert (partition_num > 0);
    // partition_nodes.resize(partition_num);
    // train_id.resize(partition_num);
    for (int i = 0; i < partition_num; ++i) {
      train_id[i].clear();
    }
    std::string part_file2 = "/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result_part_info/" + graph->config->part_algo 
                            + "-" + graph->config->dataset_name + "-part" + to_string(graph->config->part_num) + "-traininfo.txt";
    std::cout << "read_file " << part_file2 << std::endl;
    std::ifstream fin2(part_file2.c_str(), std::ios::in);
    if (!fin2)
    {
        std::cout << "cannot open file" << std::endl;
        exit(1);
    }
    int partition_id = 0;
    VertexId vertices = 0;
    VertexId vertex_id;
    std::string line;
    while(getline(fin2, line))
    {
        std::stringstream ss(line);
        ss >> partition_id >> vertices;
        // std::cout << "line " << line << std::endl;
        while(ss >> vertex_id)
        {
          assert(vertex_id < graph->config->vertices);
          assert(gnndatum->local_mask[vertex_id] == 0);
          train_id[partition_id].push_back(vertex_id);
        }
        // std::cout << "read part " << partition_id << ' ' << train_id.size() << " " << vertices << std::endl;
        assert(train_id[partition_id].size() == vertices);
    }
    fin2.close();
    std::cout << "read " << part_file2 << " done." << std::endl;
  }


  // struct pair_hash {
  //   inline VertexId operator()( std::pair<VertexId, VertexId> & v)  {
  //       return v.first*31+v.second;
  //   }
  // };


  void print_partition_info() {
    std::cout << "\n########################" << std::endl;
    std::cout << "partition_nodes: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << partition_nodes[i].size() << " ";;
    } std::cout << std::endl;

    std::cout << "train_nodes: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << train_id[i].size() << " ";;
    } std::cout << std::endl;

    std::vector<int> partition_edge_num;
    for (int i = 0; i < partition_num; ++i) {
      // std::unordered_set<std::pair<VertexId, VertexId>, pair_hash> edges;
      // std::set<std::pair<VertexId, VertexId>, pair_hash> edges;
      // edges.clear();
      int edge_num = 0;
      for (auto u : partition_nodes[i]) {
        edge_num += fully_rep_graph->column_offset[u + 1] - fully_rep_graph->column_offset[u];
        // for (int j = fully_rep_graph->column_offset[u]; j < fully_rep_graph->column_offset[u + 1]; ++j) {
        //   edges.insert({u, fully_rep_graph->row_indices[j]});
        // }
      }
      partition_edge_num.push_back(edge_num);
    }
    std::cout << "partition_edges: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << partition_edge_num[i] << " ";;
    } std::cout << std::endl;
   
    std::vector<int> partition_train_edge_num;
    for (int i = 0; i < partition_num; ++i) {
      int edge_num = 0;
      for (auto u : train_id[i]) {
        edge_num += fully_rep_graph->column_offset[u + 1] - fully_rep_graph->column_offset[u];
      }
      partition_train_edge_num.push_back(edge_num);
    }
    std::cout << "partition_train_edges: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << partition_train_edge_num[i] << " ";;
    } std::cout << std::endl;

    std::vector<int> partition_train_node_num;
    for (int i = 0; i < partition_num; ++i) {
      std::set<VertexId> nodes;
      int edge_num = 0;
      for (auto u : train_id[i]) {
        edge_num += fully_rep_graph->column_offset[u + 1] - fully_rep_graph->column_offset[u];
        for (int j = fully_rep_graph->column_offset[u]; j < fully_rep_graph->column_offset[u + 1]; ++j) {
          nodes.insert(fully_rep_graph->row_indices[j]);
        }
      }
      partition_train_node_num.push_back(nodes.size());
    }
    std::cout << "partition_train_nodes: ";
    for (int i = 0; i < partition_num; ++i) {
       std::cout << partition_train_node_num[i] << " ";;
    } std::cout << std::endl;
    std::cout << "########################\n" << std::endl;
  }

  void test_depcache() {
    partition_nodes.clear();
    train_id.clear();
    read_partition_result();
    if (graph->config->part_algo == "pagraph") {
      read_pagraph_train_partition_result();
    }

    print_partition_info();


    std::vector<Sampler*> train_sampler;
    for(int i = 0; i < partition_num; i++)
    {
      train_sampler.push_back(new Sampler(fully_rep_graph, train_id[i], pipelines, false) );
      printf("part %d batch %d %d\n", i, train_sampler.back()->batch_nums, int(train_id[i].size() + graph->config->batch_size - 1) / graph->config->batch_size);
      shuffle_vec_seed(train_sampler[i]->sample_nids);
    }

    assert(layers == graph->gnnctx->fanout.size());
    std::vector<std::unordered_set<VertexId>> local_nodes(partition_num);
    std::vector<std::unordered_set<VertexId>> remote_nodes(partition_num);
    std::vector<std::unordered_map<VertexId, int>> node_sample_num(layers);
    node_sample_num.clear();
    std::vector<std::vector<std::vector<VertexId>>> partition_sample_nodes(partition_num, std::vector<std::vector<VertexId>>(layers, std::vector<VertexId>()));
    std::vector<std::vector<std::vector<VertexId>>> recv_sample_nodes(partition_num, std::vector<std::vector<VertexId>>(layers, std::vector<VertexId>()));
    

    for(int part = 0; part < partition_num; part++)
    {
      long local_node_num = 0, remote_node_num = 0;
      long local_edge_num = 0, remote_edge_num = 0;
      int batch_id = 0;
      while (train_sampler[part]->work_offset < train_sampler[part]->work_range[1]) {
        local_nodes[part].clear();
        remote_nodes[part].clear();
        auto ssg = train_sampler[part]->subgraph;
        train_sampler[part]->sample_one(ssg, graph->config->batch_type, ctx->is_train());
        //////// local/remote node_num
        for(int l = 0; l < layers; l++) {
        // std::cout << "batch_id: " << batch_id << " " << ssg->sampled_sgs[l]->src().size() << " " << ssg->sampled_sgs[l]->dst().size() << std::endl;
          for (auto read_v : ssg->sampled_sgs[l]->src()) {
            if(partition_nodes[part].find(read_v) != partition_nodes[part].end()) {
              local_nodes[part].insert(read_v);
            } else {
              remote_nodes[part].insert(read_v);
            }
          }
        }
        for (auto read_v : ssg->sampled_sgs[layers-1]->dst()) {
            if(partition_nodes[part].find(read_v) != partition_nodes[part].end()) {
              local_nodes[part].insert(read_v);
            } else {
              remote_nodes[part].insert(read_v);
            }
        }
        local_node_num += local_nodes[part].size();
        remote_node_num += remote_nodes[part].size();
        ////////////////////////////////

         // 统计每个点的采样的数量
         for(int l = 0; l < layers; l++) {
          auto one_layer = ssg->sampled_sgs[l];
          long debug_edges = 0;
          for (int i = 0; i < one_layer->dst().size(); ++i) {
            VertexId u = one_layer->dst(i);
            int sample_nums = one_layer->c_o(i + 1) - one_layer->c_o(i);
            debug_edges += sample_nums;
            if(node_sample_num[l].find(u) == node_sample_num[l].end()) {
              node_sample_num[l][u] = sample_nums;
            } else {
              assert(sample_nums == node_sample_num[l][u]);
            }
          }
          assert(debug_edges == one_layer->r_i().size());
        }


 
        // 统计每个分区需要执行采样的顶点，local+remote on each layer
         for(int l = 0; l < layers; l++) {
          for (auto u : ssg->sampled_sgs[l]->dst()) {
            if (graph->config->part_algo == "pagraph") {
              partition_sample_nodes[part][l].push_back(u);
            } else {
              VertexId tmp_part = belongs[u];
              partition_sample_nodes[tmp_part][l].push_back(u);
              if (tmp_part != part) {
                recv_sample_nodes[tmp_part][l].push_back(u);
              }
            }
          }
        }


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
      std::cout << "part[" << part << "] compute_load: " << local_edge_num + remote_edge_num << std::endl;
      std::cout << std::endl;
    }

    for (int i = 0; i < partition_num; ++i) {
      // sample load
      long sample_load = 0;
      for (int l = 0; l < layers; ++l) {
        for (auto u : partition_sample_nodes[i][l]) {
          sample_load += node_sample_num[l][u];
        }
      }

      // send sample edges
      long send_sample_edges = 0;
      for (int l = 0; l < layers; ++l) {
        for (auto u : recv_sample_nodes[i][l]) {
          send_sample_edges += node_sample_num[l][u];
        }
      }
      std::cout << "part[" << i << "] sample_load: " << sample_load << " send_sample_edges: " << send_sample_edges << std::endl;
    }

    
  }
};
