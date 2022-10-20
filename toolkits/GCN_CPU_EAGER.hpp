#include "core/neutronstar.hpp"
class GCN_CPU_EAGER_impl {
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
  Graph<Empty> *graph;
  // std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  // GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  nts::ctx::NtsContext *ctx;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  torch::nn::Dropout drpmodel;
  double exec_time = 0;
  double all_sync_time = 0;
  double sync_time = 0;
  double all_graph_sync_time = 0;
  double graph_sync_time = 0;
  double all_compute_time = 0;
  double compute_time = 0;
  double all_copy_time = 0;
  double copy_time = 0;
  double graph_time = 0;
  double all_graph_time = 0;

  GCN_CPU_EAGER_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
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
    // std::vector<CSC_segment_pinned *> csc_segment;
    //    graph->generate_COO();
    //    graph->reorder_COO_W2W();
    //    // generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
    //    gt = new GraphOperation(graph, active);
    //    gt->GenerateGraphSegment(subgraphs, CPU_T, [&](VertexId src, VertexId dst) {
    //      return gt->norm_degree(src, dst);
    //    });
    //    gt->GenerateMessageBitmap_multisokects(subgraphs);
    partitioned_graph = new PartitionedGraph(graph, active);
    partitioned_graph->GenerateAll([&](VertexId src, VertexId dst) { return 1; }, CPU_T);
    graph->init_communicatior();
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

    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx, graph);
    // gnndatum->random_generate();
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      // gnndatum->readFtrFrom1(graph->config->feature_file,graph->config->label_file);
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
    }
    // target = torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      //        bias.push_back(new Parameter(1,graph->gnnctx->layer_size[i+1]));
    }
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      //        bias[i]->init_parameter();
    }
    drpmodel = torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(false));

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);

    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    X[0] = F.set_requires_grad(true);
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
      if (s == 0)
        std::cout << "Train ACC: " << acc_train << " " << g_train << " " << g_correct << std::endl;
      else if (s == 1)
        std::cout << "Eval  ACC: " << acc_train << " " << g_train << " " << g_correct << " " << std::endl;
      else if (s == 2)
        std::cout << "Test  ACC: " << acc_train << " " << g_train << " " << g_correct << " " << std::endl;
    }
  }

  void Loss() {
    //  return torch::nll_loss(a,L_GT_C);        y = y.log_softmax(1);
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(1);
    torch::Tensor mask_train = MASK.eq(0);
    loss = torch::nll_loss(a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)})).view({-1, a.size(1)}),
                           L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }

  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    if (layer == 0) {
      y = P[layer]->forward(a);
    } else if (layer == 1) {
      y = P[layer]->forward(torch::relu(drpmodel(a)));
    }
    return y;
  }
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;

      NtsVar Y_i = ctx->runVertexForward([&](NtsVar n_i, NtsVar v_i) { return vertexForward(n_i, v_i); }, X[i], X[i]);
      X[i + 1] = ctx->runGraphOp<nts::op::ForwardCPUfuseOp>(partitioned_graph, active, Y_i);
    }
  }

  void run() {
    if (graph->partition_id == 0) printf("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epochs\n", iterations);
    // graph->print_info();

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      Forward();
      Test(0);
      Test(1);
      Test(2);
      Loss();
      ctx->self_backward(true);
      Update();

      if (graph->partition_id == 0) std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss << std::endl;
    }
    exec_time += get_time();

    delete active;
  }
};
