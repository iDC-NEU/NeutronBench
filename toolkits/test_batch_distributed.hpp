#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"

class TEST_BATCH_DIST_impl {
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

  TEST_BATCH_DIST_impl(Graph<Empty> *graph_, int iterations_,
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
  
  void init_active() {
    active = graph->alloc_vertex_subset();
    active->fill();
  }

  void init_graph() {
    fully_rep_graph=new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");
       
    //cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx=new nts::ctx::NtsContext();
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
      gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                       graph->config->label_file,
                                       graph->config->mask_file);
    }

    if (graph->config->classes > 1) {
      gnndatum->registLabel(L_GT_C, gnndatum->local_label, gnndatum->gnnctx->l_v_num, graph->config->classes);
    } else {
      gnndatum->registLabel(L_GT_C);
    }
    // std::cout << L_GT_C << std::endl;
    gnndatum->registMask(MASK);

  }

  template<typename T>
  void print(std::vector<T> &vec) {
    int n = vec.size();
    for (int i = 0; i < n; ++i) {
      if (i) std::cout << "-";
      std::cout << vec[i];
    }
  }

  void layer_nodes_distributed(Sampler* sampler, int type=0) {
    if (graph->config->batch_type != SEQUENCE) { // shuffle
      shuffle_vec(sampler->sample_nids);
    }  
    double sample_cost = -get_time();
    while(sampler->sample_not_finished()) {
      sampler->sample_one(layers,
                                graph->config->batch_size,
                                graph->gnnctx->fanout, graph->config->batch_type, ctx->is_train());
    }
    sample_cost += get_time();
    LOG_DEBUG("sample cost %.3f", sample_cost);
    std::vector<std::vector<int>> batch_nodes;
    std::vector<std::vector<int>> layer_nodes(layers + 1);

    SampledSubgraph *sg; 
    int batch_idx = 0;
    while (sampler->has_rest()) {
      sg = sampler->get_one();
      std::vector<int> tmp;
      for (int i = 0; i < layers; ++i) {
        auto *p = sg->sampled_sgs[i];
        if (i == 0) {
          tmp.push_back(p->src().size());
          layer_nodes[i].push_back(p->src().size());
        }
        tmp.push_back(p->dst().size());
        layer_nodes[i + 1].push_back(p->dst().size());
        // LOG_DEBUG("batch %d layers %d blocks %d %d", batch_idx, i, p->src().size(), p->dst().size());
      }
      batch_nodes.push_back(tmp);
      batch_idx++;
    }

    assert(batch_nodes.size() == batch_idx);
    for (int i = 0; i < layers + 1; ++i) {
      assert(layer_nodes[i].size() == batch_idx);
    }

    for (int i = 0; i < batch_idx; ++i) {
      for (int j = 0; j < layers + 1; ++j) {
        assert(layer_nodes[j][i] == batch_nodes[i][j]);
      }
    }

    std::sort(batch_nodes.begin(), batch_nodes.end(), [&](auto &X, auto &Y) {
      return X[0] - X[1] > Y[0] - Y[1];
    });
    std::cout << "batch num " << batch_idx << std::endl;
    std::cout << "max batch ";
    print(batch_nodes[0]); std::cout << endl;
    std::cout << "min batch ";
    print(batch_nodes.back()); std::cout << endl;

    
    std::vector<float> layer_node_distributed;
    for (int i = 0; i < layers + 1; ++i) {
      auto ret = get_mean_var(layer_nodes[i]);
      layer_node_distributed.push_back(ret.first);
      layer_node_distributed.push_back(ret.second);
    }
    std::cout << "layer nodes distirbuted (mean var):" << layer_node_distributed << std::endl;
    sampler->clear_queue();
    sampler->restart();
  }


  void shuffle_vec(std::vector<VertexId>& vec) {
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
    std::shuffle (vec.begin(), vec.end(), std::default_random_engine(seed));
  }

  void run() {
    std::vector<VertexId> train_nids, val_nids, test_nids;
    BatchType batch_type = graph->config->batch_type;
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

    shuffle_vec(val_nids);
    shuffle_vec(test_nids);
    
    if (batch_type == SHUFFLE) shuffle_vec(train_nids);

    if (batch_type == RANDOM || batch_type == DELLOW || batch_type == DELHIGH) { // random
      if (batch_type == DELHIGH) {
         std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] > graph->in_degree_for_backward[y];
        });
      } else if (batch_type == DELLOW){
        std::sort(train_nids.begin(), train_nids.end(), [&](const auto& x, const auto& y) {
          return graph->in_degree_for_backward[x] < graph->in_degree_for_backward[y];
        });
      } else { // random del nodes
        shuffle_vec(train_nids);
      }
      int sz = train_nids.size();
      train_nids.erase(train_nids.begin() + static_cast<int>(sz * (1 - graph->config->del_frac)), train_nids.end());
      shuffle_vec(train_nids);
    }
    std::cout << "train/val/test: " << train_nids.size() << " " << val_nids.size() << " " << test_nids.size() << std::endl;
    Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids);
    Sampler* eval_sampler = new Sampler(fully_rep_graph, val_nids);
    Sampler* test_sampler = new Sampler(fully_rep_graph, test_nids);

    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      // zero_grad();
      LOG_DEBUG("EPOCH %d", i_i);
      layer_nodes_distributed(train_sampler, 0);

    }
    
    delete active;
  }
};