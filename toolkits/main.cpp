/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "GAT_CPU.hpp"
#include "GCN_CPU_NEIGHBOR.hpp"
#include "GCN_CPU_LAYER.hpp"
#include "GCN_CPU_CLUSTER.hpp"
#include "GCN_CPU.hpp"
#include "GCN_CPU_EAGER.hpp"
#include "GAT_CPU_DIST.hpp"
#include "GIN_CPU.hpp"
#include "test_getdepneighbor_cpu.hpp"
#if CUDA_ENABLE
#include "test_getdepneighbor_gpu.hpp"
#include "GAT_GPU_DIST.hpp"
#include "COMMNET_GPU.hpp"
#include "GCN.hpp"
#include "GCN_EAGER.hpp"
#include "GCN_EAGER_single.hpp"
#include "GIN_GPU.hpp"
#endif

int main(int argc, char **argv) {
  MPI_Instance mpi(&argc, &argv);
  if (argc < 2) {
    printf("configuration file missed \n");
    exit(-1);
  }

  double exec_time = 0;
  exec_time -= get_time();

  Graph<Empty> *graph;
  graph = new Graph<Empty>();
  graph->config->readFromCfgFile(argv[1]);
  if (graph->partition_id == 0)
    graph->config->print();

  int iterations = graph->config->epochs;
  graph->replication_threshold = graph->config->repthreshold;

  if (graph->config->algorithm == std::string("GCNCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_CPU_impl *ntsGCN = new GCN_CPU_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GINCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GIN_CPU_impl *ntsGIN = new GIN_CPU_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    ntsGIN->run();
  } else if (graph->config->algorithm == std::string("GCNNEIGHBOR")) {
    double load_graph_time = -get_time();
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    load_graph_time += get_time();
    LOG_DEBUG("load graph cost %.3f", load_graph_time);

    double generate_backward_time = -get_time();
    graph->generate_backward_structure();
    generate_backward_time += get_time();
    LOG_DEBUG("generate backward cost %.3f", generate_backward_time);

    GCN_CPU_NEIGHBOR_impl *ntsGCN = new GCN_CPU_NEIGHBOR_impl(graph, iterations);
    
    double init_graph_time = -get_time();
    ntsGCN->init_graph();
    init_graph_time += get_time();
    LOG_DEBUG("init graph cost %.3f", init_graph_time);

    double init_nn_time = -get_time();
    ntsGCN->init_nn();
    init_nn_time += get_time();
    LOG_DEBUG("init nn cost %.3f", init_nn_time);

    vector<float> best_val_accs;
    for (int i = 0; i < graph->config->runs; ++i) {
      if (i > 0) ntsGCN->init_active();
      float acc = ntsGCN->run();
      best_val_accs.push_back(acc);
    }
    std::cout << "accs: ";
    for (auto &it : best_val_accs) {
      std::cout << it << " ";
    }std::cout << std::endl;
    float mean, var;
    tie(mean, var) = get_mean_var(best_val_accs);
    printf("val acc %d runs: %.4f(%.4f)\n", graph->config->runs, mean, var);
    std::cout << "edge_file: " << graph->config->edge_file << std::endl;
  } else if (graph->config->algorithm == std::string("GCNLAYER")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_CPU_LAYER_impl *ntsGCN = new GCN_CPU_LAYER_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GCNCLUSTER")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_CPU_CLUSTER_impl *ntsGCN = new GCN_CPU_CLUSTER_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GCNCPUEAGER")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_CPU_EAGER_impl *ntsGCN = new GCN_CPU_EAGER_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
//  } else if (graph->config->algorithm == std::string("GGNNCPU")) {
//    graph->load_directed(graph->config->edge_file, graph->config->vertices);
//    graph->generate_backward_structure();
//    GGNN_CPU_impl *ntsGGNN = new GGNN_CPU_impl(graph, iterations);
//    ntsGGNN->init_graph();
//    ntsGGNN->init_nn();
//    ntsGGNN->run();
  } else if (graph->config->algorithm == std::string("GATCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GAT_CPU_impl *ntsGAT = new GAT_CPU_impl(graph, iterations);
    ntsGAT->init_graph();
    ntsGAT->init_nn();
    ntsGAT->run();
  }else if (graph->config->algorithm == std::string("GATCPUDIST")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GAT_CPU_DIST_impl *ntsGAT = new GAT_CPU_DIST_impl(graph, iterations);
    ntsGAT->init_graph();
    ntsGAT->init_nn();
    ntsGAT->run();
//  }  else if (graph->config->algorithm == std::string("GGCNCPU")) {
//    graph->load_directed(graph->config->edge_file, graph->config->vertices);
//    graph->generate_backward_structure();
//    GGCN_CPU_impl *ntsGAT = new GGCN_CPU_impl(graph, iterations);
//    ntsGAT->init_graph();
//    ntsGAT->init_nn();
//    ntsGAT->run();
  }
  else if (graph->config->algorithm == std::string("test_getdep1")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    test_get_neighbor *ntsGAT = new test_get_neighbor(graph, iterations);
    ntsGAT->init_graph();
    ntsGAT->init_nn();
    ntsGAT->run();
  }
 
#if CUDA_ENABLE
  else if (graph->config->algorithm == std::string("test_getdep")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    test_get_neighbor_gpu *ntsGAT = new test_get_neighbor_gpu(graph, iterations);
    ntsGAT->init_graph();
    ntsGAT->init_nn();
    ntsGAT->run();
  }
    else if (graph->config->algorithm == std::string("GATGPUDIST")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GAT_GPU_DIST_impl *ntsGAT = new GAT_GPU_DIST_impl(graph, iterations);
    ntsGAT->init_graph();
    ntsGAT->init_nn();
    ntsGAT->run();
    } else if (graph->config->algorithm == std::string("COMMNETGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    COMMNET_impl *ntsCOMM = new COMMNET_impl(graph, iterations);
    ntsCOMM->init_graph();
    ntsCOMM->init_nn();
    ntsCOMM->run();
  } else if (graph->config->algorithm == std::string("GINGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GIN_impl *ntsGIN = new GIN_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    ntsGIN->run();
  } else if (graph->config->algorithm == std::string("GCN")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_impl *ntsGCN = new GCN_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
    // GCN(graph, iterations);
  } else if (graph->config->algorithm == std::string("GCNEAGER")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_EAGER_impl *ntsGCN = new GCN_EAGER_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
    // GCN(graph, iterations);
  } else if (graph->config->algorithm == std::string("GCNEAGERSINGLE")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_EAGER_single_impl *ntsGCN =
        new GCN_EAGER_single_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("COMMNETGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    COMMNET_impl *ntsCOMM = new COMMNET_impl(graph, iterations);
    ntsCOMM->init_graph();
    ntsCOMM->init_nn();
    ntsCOMM->run();
  } else if (graph->config->algorithm == std::string("GINGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GIN_impl *ntsGIN = new GIN_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    ntsGIN->run();
  }
#endif
  exec_time += get_time();
  if (graph->partition_id == 0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  delete graph;

  //    ResetDevice();

  return 0;
}
