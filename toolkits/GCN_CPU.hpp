#include "core/neutronstar.hpp"

class GCN_CPU_impl {
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

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  torch::nn::Dropout drpmodel;
  std::vector<torch::nn::BatchNorm1d> bn1d;

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

  GCN_CPU_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false) {
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
    partitioned_graph = new PartitionedGraph(graph, active);
    partitioned_graph->GenerateAll(
        [&](VertexId src, VertexId dst) { return nts::op::nts_norm_degree(graph, src, dst); }, CPU_T,
        (graph->partitions) > 1);
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
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx, graph);
    // gnndatum->random_generate();
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
      // gnndatum->readFeature_Label_Mask_OGB(graph->config->feature_file,
      //                                  graph->config->label_file,
      //                                  graph->config->mask_file);
    }

    // creating tensor to save Label and Mask
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      if (i < graph->gnnctx->layer_size.size() - 2)
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i]));
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
    // X[0] is the initial vertex representation. We created it from
    // local_feature
    X[0] = F.set_requires_grad(true);
  }

  float Test(long s) {  // 0 train, //1 eval //2 test
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
    // if (graph->partition_id == 0) {
    //   if (s == 0) {
    //     LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
    //   } else if (s == 1) {
    //     LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
    //   } else if (s == 2) {
    //     LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
    //   }
    // }
    return acc_train;
  }
  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    // nn operation. Here is just a simple matmul. i.e. y = activate(a * w)
    if (layer == 0) {
      a = this->bn1d[layer](a);
      y = torch::relu(P[layer]->forward(a)).set_requires_grad(true);
    } else if (layer == 1) {
      y = P[layer]->forward(a);
      y = y.log_softmax(1);
    }
    // save the intermediate result for backward propagation
    //   ctx->op_push(a, y, nts::ctx::NNOP);
    return y;
  }
  void Loss() {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(1);
    torch::Tensor mask_train = MASK.eq(0);
    loss = torch::nll_loss(a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)})).view({-1, a.size(1)}),
                           L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      //      if (i != 0) {
      //        X[i] = drpmodel(X[i]);
      //      }

      NtsVar Y_i = ctx->runGraphOp<nts::op::ForwardCPUfuseOp>(partitioned_graph, active, X[i]);
      X[i + 1] = ctx->runVertexForward(
          [&](NtsVar n_i, NtsVar v_i) {
            if (i < (graph->gnnctx->layer_size.size() - 2)) {
              n_i = this->bn1d[i](n_i);
              return drpmodel(torch::relu(P[i]->forward(n_i)));

            } else {
              return P[i]->forward(n_i);
            }

            // return vertexForward(n_i, v_i);
          },
          Y_i, X[i]);
    }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      double epoch_train_time = -get_time();
      Forward();
      Loss();
      ctx->self_backward();
      Update();
      epoch_train_time += get_time();

      float train_acc = Test(0);
      float val_acc = Test(1);
      // float test_acc = Test(2);

      //       ctx->debug();
      if (graph->partition_id == 0)
        LOG_INFO("Epoch %03d epoch_train_time %.3f train_loss %.3f train_acc %.3f val_acc %.3f", i_i, epoch_train_time,
                 loss.item<float>(), train_acc, val_acc);
    }
    exec_time += get_time();
    //    std::string str="a10";
    //    at::ArrayRef<at::Dimname>names({at::Dimname::fromSymbol(at::Symbol::dimname(str)),at::Dimname::fromSymbol(at::Symbol::dimname("b10"))});
    //    at::ArrayRef<at::Dimname>names({at::Dimname::fromSymbol(at::Symbol::dimname(str)),at::Dimname::fromSymbol(at::Symbol::dimname("b10"))});
    //    NtsVar s=torch::ones({3,3},names,at::TensorOptions().requires_grad(true));
    //    std::vector<at::Dimname> dim;

    //    dim.push_back(at::Dimname::fromSymbol(at::Symbol::aten("0")));
    //    dim.push_back(at::Dimname::fromSymbol(at::Symbol::aten("1")));
    //    s.get_named_tensor_meta()->set_names(at::NamedTensorMeta::HasNonWildcard,dim);
    //    at::DimnameList::
    //    std::cout<<" "<<s.names()[0]<<" " <<s.names()[1]<<std::endl;
    delete active;
  }
};
