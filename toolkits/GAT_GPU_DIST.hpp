#include "core/neutronstar.hpp"
class GAT_GPU_DIST_impl {
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
  NtsVar MASK_gpu;
  // GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext *ctx;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;

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

  GAT_GPU_DIST_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false) {
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
    graph->rtminfo->with_cuda = true;
    graph->rtminfo->lock_free = graph->config->lock_free;
  }
  void init_graph() {
    partitioned_graph = new PartitionedGraph(graph, active);
    partitioned_graph->GenerateAll(
        [&](VertexId src, VertexId dst) { return nts::op::nts_norm_degree(graph, src, dst); }, CPU_T, true);
    graph->init_communicatior();
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
    torch::manual_seed(0);
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx, graph);
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
    L_GT_G = L_GT_C.cuda();
    MASK_gpu = MASK.cuda();

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      P.push_back(new Parameter(graph->gnnctx->layer_size[i + 1] * 2, 1, alpha, beta1, beta2, epsilon, weight_decay));
    }

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    torch::Device GPU(torch::kCUDA, 0);
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      P[i]->to(GPU);
      P[i]->Adam_to_GPU();
    }

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);

    NtsVar d;
    X.resize(graph->gnnctx->layer_size.size(), d);
    // X[0] is the initial vertex representation. We created it from
    // local_feature
    X[0] = F.cuda();
  }

  void Test(long s) {  // 0 train, //1 eval //2 test
    NtsVar mask_train = MASK_gpu.eq(s);
    NtsVar all_train = X[graph->gnnctx->layer_size.size() - 1]
                           .argmax(1)
                           .to(torch::kLong)
                           .eq(L_GT_G)
                           .to(torch::kLong)
                           .masked_select(mask_train.view({mask_train.size(0)}));
    NtsVar all = all_train.sum(0).cpu();
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
  void Loss() {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(1);
    torch::Tensor mask_train = MASK_gpu.eq(0);
    loss = torch::nll_loss(a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)})).view({-1, a.size(1)}),
                           L_GT_G.masked_select(mask_train.view({mask_train.size(0)})));
    ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      P[i]->learnC2G_with_decay_Adam();
      P[i]->next();
    }
  }
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      NtsVar X_trans = ctx->runVertexForward(
          [&](NtsVar x_i_) {
            int layer = graph->rtminfo->curr_layer;
            return P[2 * layer]->forward(x_i_);
          },
          X[i]);
      NtsVar mirror = ctx->runGraphOp<nts::op::DistGPUGetDepNbrOp>(partitioned_graph, active, X_trans);
      NtsVar edge_src = ctx->runGraphOp<nts::op::DistGPUScatterSrc>(partitioned_graph, active, mirror);
      NtsVar edge_dst = ctx->runGraphOp<nts::op::DistGPUScatterDst>(partitioned_graph, active, X_trans);
      NtsVar e_msg = torch::cat({edge_src, edge_dst}, 1);
      NtsVar m = ctx->runEdgeForward(
          [&](NtsVar e_msg_) {
            int layer = graph->rtminfo->curr_layer;
            return torch::leaky_relu(P[2 * layer + 1]->forward(e_msg_), 0.2);
          },
          e_msg);  // edge NN
      //  partitioned_graph->SyncAndLog("e_msg_in");
      NtsVar a = ctx->runGraphOp<nts::op::DistGPUEdgeSoftMax>(partitioned_graph, active, m);  // edge NN
      NtsVar e_msg_out = ctx->runEdgeForward([&](NtsVar a_) { return edge_src * a_; },
                                             a);  // Edge NN
      //            partitioned_graph->SyncAndLog("e_msg_out");
      NtsVar nbr = ctx->runGraphOp<nts::op::DistGPUAggregateDst>(partitioned_graph, active, e_msg_out);
      X[i + 1] = ctx->runVertexForward([&](NtsVar nbr_) { return torch::relu(nbr_); }, nbr);
      partitioned_graph->SyncAndLog("hello 2");
    }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GATimpl] running [%d] Epoches\n", iterations);
    }

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      Forward();

      //      printf("sizeof %d",sizeof(__m256i));
      //      printf("sizeof %d",sizeof(int));
      Test(0);
      Test(1);
      Test(2);
      Loss();

      ctx->self_backward(true);
      Update();
      // ctx->debug();
      if (graph->partition_id == 0) std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss << std::endl;
    }

    delete active;
  }
};
