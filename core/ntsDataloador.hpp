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
#ifndef NTSDATALODOR_HPP
#define NTSDATALODOR_HPP
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <map>
#include <sstream>
#include <stack>
#include <vector>

#include "core/graph.hpp"
#include "utils/utils.hpp"

class GNNDatum {
 public:
  GNNContext *gnnctx;
  Graph<Empty> *graph;
  ValueType *local_feature;  // features of local partition
  ValueType *dev_local_feature;
  // ValueType *local_embedding;
  // ValueType *dev_local_embedding;
  long *dev_local_label;

  long *local_label;   // labels of local partition
  long *global_label;  // labels of global partition
  int *local_mask;     // mask(indicate whether data is for train, eval or test) of
  int *global_mask;    // mask(indicate whether data is for train, eval or test) of
                       // local partition

  // GNN datum world

  // train:    0
  // val:     1
  // test:     2
  /**
   * @brief Construct a new GNNDatum::GNNDatum object.
   * initialize GNN Data using GNNContext and Graph.
   * Allocating space to save data. e.g. local feature, local label.
   * @param _gnnctx pointer to GNN Context
   * @param graph_ pointer to Graph
   */
  GNNDatum(GNNContext *_gnnctx, Graph<Empty> *graph_) {
    gnnctx = _gnnctx;
    // local_feature = new ValueType[gnnctx->l_v_num * gnnctx->layer_size[0]];
    local_feature =
        (ValueType *)cudaMallocPinned((long)(gnnctx->l_v_num) * (gnnctx->layer_size[0]) * sizeof(ValueType));
    // local_embedding =
    // (ValueType*)cudaMallocPinned((long)(gnnctx->l_v_num)*(gnnctx->layer_size[0])*sizeof(ValueType));
    local_label = new long[gnnctx->l_v_num * graph_->config->classes];
    global_label = new long[graph_->partition_offset[graph_->partitions] * graph_->config->classes];
    memset(local_label, 0, sizeof(long) * gnnctx->l_v_num * graph_->config->classes);
    memset(global_label, 0, sizeof(long) * graph_->partition_offset[graph_->partitions] * graph_->config->classes);
    local_mask = new int[gnnctx->l_v_num];
    global_mask = new int[graph_->partition_offset[graph_->partitions]];
    // std::cout << "fuck " << graph->partitions << std::endl;
    // std::cout << "parts " << graph_->partitions<< " " << graph_->partition_offset[graph_->partitions] << std::endl;
    memset(local_mask, 0, sizeof(int) * gnnctx->l_v_num);
    memset(global_mask, 0, sizeof(int) * graph_->partition_offset[graph_->partitions]);
    graph = graph_;
  }

  void generate_gpu_data() {
    dev_local_label = (long *)cudaMallocGPU(gnnctx->l_v_num * sizeof(long) * graph->config->classes);
    dev_local_feature = (ValueType *)getDevicePointer(local_feature);
    // dev_local_embedding = (ValueType*) getDevicePointer(local_embedding);
    move_bytes_in(dev_local_label, local_label, gnnctx->l_v_num * sizeof(long) * graph->config->classes);
  }

  /**
   * @brief
   * generate random data for feature, label and mask
   */
  void random_generate() {
    // LOG_DEBUG("start random generate l_v_num %d layer %d lable_num %d", gnnctx->l_v_num, gnnctx->layer_size[0],
    // gnnctx->label_num); LOG_DEBUG("sizeof local_feature %d", sizeof(local_feature)); local_feature = new
    // float[gnnctx->l_v_num * gnnctx->layer_size[0]]; local_feature = (float*)malloc(gnnctx->l_v_num *
    // gnnctx->layer_size[0] * sizeof(float)); LOG_DEBUG("sizeof local_feature %d", sizeof(local_feature));

    for (long i = 0; i < gnnctx->l_v_num; i++) {
      // if (i % 500000 == 0 && i) printf("node id %d\n", i);
      for (long j = 0; j < gnnctx->layer_size[0]; j++) {
        local_feature[i * gnnctx->layer_size[0] + j] = 1.0;
      }
      local_label[i] = rand() % gnnctx->label_num;
      // local_mask[i] = i % 3;
    }

    // LOG_DEBUG("start random generate local_mask");
    std::vector<int> shuffle_nodes(gnnctx->l_v_num);
    std::iota(shuffle_nodes.begin(), shuffle_nodes.end(), 0);
    assert(shuffle_nodes.back() == gnnctx->l_v_num - 1);
    shuffle_vec(shuffle_nodes);
    // std::cout << "after shuffle ";
    // for (int i = 0; i < 10; ++i) {
    //   std::cout << shuffle_nodes[i] << " ";
    // } std::cout << std::endl;
    int tmpN = gnnctx->l_v_num;
    for (int i = 0; i < tmpN / 10 * 6; ++i) {
      local_mask[shuffle_nodes[i]] = 0;
    }
    for (int i = tmpN / 10 * 6; i < tmpN / 10 * 8; ++i) {
      local_mask[shuffle_nodes[i]] = 1;
    }
    for (int i = tmpN / 10 * 8; i < shuffle_nodes.size(); ++i) {
      local_mask[shuffle_nodes[i]] = 2;
    }

    // debug
    int cnt0 = 0, cnt1 = 0, cnt2 = 0;
    assert(tmpN == gnnctx->l_v_num);
    for (int i = 0; i < gnnctx->l_v_num; ++i) {
      if (local_mask[i] == 0) cnt0++;
      if (local_mask[i] == 1) cnt1++;
      if (local_mask[i] == 2) cnt2++;
    }
    assert(cnt0 + cnt1 + cnt2 == tmpN);
    LOG_DEBUG("local_mask: (%d %d %d) (%.1f %.1f %.1f)", cnt0, cnt1, cnt2, cnt0 * 1.0 / cnt2, cnt1 * 1.0 / cnt2, 1.0);
  }
  /**
   * @brief
   * Create tensor corresponding to local label
   * @param target target tensor where we should place local label
   */
  void registLabel(NtsVar &target) {
    target = graph->Nts->NewLeafKLongTensor(local_label, {gnnctx->l_v_num});
    // torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
  }

  void registLabel(NtsVar &target, long *data, int w, int h) {
    target = graph->Nts->NewLeafKLongTensor(data, {w, h});
    // target = graph->Nts->NewLeafKLongTensor(local_label, {w, h});
    // torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
    //  for (int i = 0; i < 121; ++i) {
    //     std::cout << local_label[100 * 121 + i] << " ";
    //   }std::cout << std::endl;
    // target = torch::from_blob(local_label, {gnnctx->l_v_num,121}, torch::kLong);
    // std::cout << target[100] << std::endl;
    // std::cout << gnnctx->l_v_num << std::endl;
    // assert(false);
  }

  void registGlobalLabel(NtsVar &target) {
    target = graph->Nts->NewLeafKLongTensor(global_label, {graph->partition_offset[graph->partitions]});
    // torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
  }

  /**
   * @brief
   * Create tensor corresponding to local mask
   * @param mask target tensor where we should place local mask
   */
  void registMask(NtsVar &mask) {
    mask = graph->Nts->NewLeafKIntTensor(local_mask, {gnnctx->l_v_num, 1});
    // torch::from_blob(local_mask, {gnnctx->l_v_num,1}, torch::kInt32);
  }

  void registGlobalMask(NtsVar &mask) {
    // std::cout << graph->partition_offset[graph->partitions] << std::endl;
    mask = graph->Nts->NewLeafKIntTensor(global_mask, {graph->partition_offset[graph->partitions]});
    // torch::from_blob(local_mask, {gnnctx->l_v_num,1}, torch::kInt32);
  }

  /**
   * @brief
   * read feature and label from file.
   * file format should be  ID Feature * (feature size) Label
   * @param inputF path to input feature
   * @param inputL path to input label
   */
  void readFtrFrom1(std::string inputF, std::string inputL) {
    std::string str;
    std::ifstream input_ftr(inputF.c_str(), std::ios::in);
    std::ifstream input_lbl(inputL.c_str(), std::ios::in);
    // std::ofstream outputl("cora.labeltable",std::ios::out);
    // ID    F   F   F   F   F   F   F   L
    if (!input_ftr.is_open()) {
      std::cout << "open feature file fail!" << std::endl;
      return;
    }
    if (!input_lbl.is_open()) {
      std::cout << "open label file fail!" << std::endl;
      return;
    }
    ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
    // TODO: figure out what is la
    std::string la;
    // std::cout<<"finish1"<<std::endl;
    VertexId id = 0;
    while (input_ftr >> id) {
      // feature size
      VertexId size_0 = gnnctx->layer_size[0];
      // translate vertex id to local vertex id
      VertexId id_trans = id - gnnctx->p_v_s;
      if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        // read feature
        for (int i = 0; i < size_0; i++) {
          input_ftr >> local_feature[size_0 * id_trans + i];
        }
        input_lbl >> la;
        // read label
        input_lbl >> local_label[id_trans];
        // partition data set based on id
        local_mask[id_trans] = id % 3;
      } else {
        // dump the data which doesn't belong to local partition
        for (int i = 0; i < size_0; i++) {
          input_ftr >> con_tmp[i];
        }
        input_lbl >> la;
        input_lbl >> la;
      }
    }
    delete[] con_tmp;
    input_ftr.close();
    input_lbl.close();
  }

  /**
   * @brief
   * read feature, label and mask from file.
   * @param inputF path to feature file
   * @param inputL path to label file
   * @param inputM path to mask file
   */
  void readFeature_Label_Mask(std::string inputF, std::string inputL, std::string inputM) {
    // logic here is exactly the same as read feature and label from file
    std::string str;
    std::ifstream input_ftr(inputF.c_str(), std::ios::in);
    std::ifstream input_lbl(inputL.c_str(), std::ios::in);
    std::ifstream input_msk(inputM.c_str(), std::ios::in);
    // std::ofstream outputl("cora.labeltable",std::ios::out);
    // ID    F   F   F   F   F   F   F   L
    if (!input_ftr.is_open()) {
      std::cout << "open feature file fail!" << std::endl;
      return;
    }
    if (!input_lbl.is_open()) {
      std::cout << "open label file fail!" << std::endl;
      return;
    }
    if (!input_msk.is_open()) {
      std::cout << "open mask file fail!" << std::endl;
      return;
    }
    // std::cout << inputF << " " << inputM << " " << inputL << std::endl;
    // assert (false);

    ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
    // std::cout << "layer size" << " " << gnnctx->layer_size[0] << std::endl;
    // assert (false);

    std::string la;
    // std::cout<<"finish1"<<std::endl;
    VertexId id = 0;
    while (input_ftr >> id) {
      VertexId size_0 = gnnctx->layer_size[0];
      VertexId id_trans = id - gnnctx->p_v_s;
      // std::cout << id << " " << id_trans << " " << size_0 <<  std::endl;

      if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        for (int i = 0; i < size_0; i++) {
          input_ftr >> local_feature[size_0 * id_trans + i];
        }
        input_lbl >> la;
        if (graph->config->classes > 1) {
          int sz, idx;
          input_lbl >> sz;
          // std::cout << la << " " << sz << std::endl;
          while (sz--) {
            input_lbl >> idx;
            local_label[id_trans * graph->config->classes + idx] = 1;
          }
        } else {
          input_lbl >> local_label[id_trans];
        }

        input_msk >> la;
        std::string msk;
        input_msk >> msk;
        // std::cout<<la<<" "<<msk<<std::endl;
        // std::cout << id << " " << id_trans << std::endl;
        if (msk.compare("train") == 0) {
          local_mask[id_trans] = 0;
        } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
          local_mask[id_trans] = 1;
        } else if (msk.compare("test") == 0) {
          local_mask[id_trans] = 2;
        } else {
          local_mask[id_trans] = 3;
        }

      } else {
        for (int i = 0; i < size_0; i++) {
          input_ftr >> con_tmp[i];
        }

        input_lbl >> la;
        input_lbl >> la;

        input_msk >> la;
        input_msk >> la;
      }
    }
    delete[] con_tmp;
    input_ftr.close();
    // std::cout << "here1" << std::endl;
    // read global label
    input_lbl.seekg(0, input_lbl.beg);
    while (input_lbl >> id) {
      if (graph->config->classes > 1) {
        int sz, idx;
        input_lbl >> sz;
        // std::cout << la << " " << sz << std::endl;
        while (sz--) {
          input_lbl >> idx;
          global_label[id * graph->config->classes + idx] = 1;
        }
      } else {
        input_lbl >> global_label[id];
      }
    }

    // std::cout << "read label" << std::endl;

    input_msk.seekg(0, input_msk.beg);
    while (input_msk >> id) {
      // if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
      std::string msk;
      input_msk >> msk;
      // std::cout << id << " " << msk << std::endl;
      if (msk.compare("train") == 0) {
        global_mask[id] = 0;
      } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
        global_mask[id] = 1;
      } else if (msk.compare("test") == 0) {
        global_mask[id] = 2;
      } else {
        global_mask[id] = 3;
      }
      // }
    }
    // std::cout << "read mask" << std::endl;
    // std::cout << global_label << std::endl << global_mask << std::endl;
    input_msk.close();
    input_lbl.close();
    // std::cout << "read all done" << std::endl;
  }

  void readFeature_Label_Mask_OGB(std::string inputF, std::string inputL, std::string inputM) {
    // logic here is exactly the same as read feature and label from file
    std::string str;
    std::ifstream input_ftr(inputF.c_str(), std::ios::in);
    std::ifstream input_lbl(inputL.c_str(), std::ios::in);
    // ID    F   F   F   F   F   F   F   L
    std::cout << inputF << std::endl;
    if (!input_ftr.is_open()) {
      std::cout << "open feature file fail!" << std::endl;
      return;
    }
    if (!input_lbl.is_open()) {
      std::cout << "open label file fail!" << std::endl;
      return;
    }
    ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
    std::string la;
    std::string featStr;
    for (VertexId id = 0; id < graph->vertices; id++) {
      VertexId size_0 = gnnctx->layer_size[0];
      VertexId id_trans = id - gnnctx->p_v_s;
      if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        getline(input_ftr, featStr);
        std::stringstream ss(featStr);
        std::string feat_u;
        int i = 0;
        while (getline(ss, feat_u, ',')) {
          local_feature[size_0 * id_trans + i] = std::atof(feat_u.c_str());
          //            if(id==0){
          //                std::cout<<std::atof(feat_u.c_str())<<std::endl;
          //            }
          i++;
        }
        assert(i == size_0);
        // input_lbl >> la;
        input_lbl >> local_label[id_trans];

      } else {
        getline(input_ftr, featStr);
        input_lbl >> la;
      }
    }

    std::string inputM_train = inputM;
    inputM_train.append("/train.csv");
    std::string inputM_val = inputM;
    inputM_val.append("/valid.csv");
    std::string inputM_test = inputM;
    inputM_test.append("/test.csv");
    std::ifstream input_msk_train(inputM_train.c_str(), std::ios::in);
    if (!input_msk_train.is_open()) {
      std::cout << "open input_msk_train file fail!" << std::endl;
      return;
    }
    std::ifstream input_msk_val(inputM_val.c_str(), std::ios::in);
    if (!input_msk_val.is_open()) {
      std::cout << inputM_val << "open input_msk_val file fail!" << std::endl;
      return;
    }
    std::ifstream input_msk_test(inputM_test.c_str(), std::ios::in);
    if (!input_msk_test.is_open()) {
      std::cout << "open input_msk_test file fail!" << std::endl;
      return;
    }
    VertexId vtx = 0;
    while (input_msk_train >> vtx) {  // train
      local_mask[vtx] = 0;
    }
    while (input_msk_val >> vtx) {  // val
      local_mask[vtx] = 1;
    }
    while (input_msk_test >> vtx) {  // test
      local_mask[vtx] = 2;
    }

    delete[] con_tmp;
    input_ftr.close();
    input_lbl.close();
  }
};

#endif
