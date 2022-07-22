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
#ifndef NTSSAMPLER_HPP
#define NTSSAMPLER_HPP
#include <mutex>
#include <cmath>
#include <stdlib.h>
#include <random>
#include "FullyRepGraph.hpp"
#include "core/MetisPartition.hpp"

class Sampler{
public:
    std::vector<SampledSubgraph*> work_queue;// excepted to be single write multi read
    std::mutex queue_start_lock;
    int queue_start;
    std::mutex queue_end_lock;
    int queue_end;
    FullyRepGraph* whole_graph;
    VertexId start_vid,end_vid;
    VertexId work_range[2];
    VertexId work_offset;
    std::vector<VertexId> sample_nids;
    std::vector<VertexId> metis_partition_offset;
    std::vector<VertexId> metis_partition_id;

    // template<typename T>
    // T RandInt(T lower, T upper) {
    //     std::uniform_int_distribution<T> dist(lower, upper - 1);
    //     return dist(rng_);
    // }
    // std::default_random_engine rng_;

    Sampler(FullyRepGraph* whole_graph_, VertexId work_start,VertexId work_end){
        whole_graph=whole_graph_;
        queue_start=-1;
        queue_end=0;
        work_range[0]=work_start;
        work_range[1]=work_end;
        work_offset=work_start;
    }
    Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index){
        assert(index.size() > 0);
        sample_nids.assign(index.begin(), index.end());
        assert(sample_nids.size() == index.size());
        whole_graph=whole_graph_;
        queue_start=-1;
        queue_end=0;
        work_range[0]=0;
        work_range[1]=sample_nids.size();
        work_offset=0;
    }
    ~Sampler(){
        clear_queue();
    }
    bool has_rest(){
        bool condition=false;
        int cond_start=0;
        queue_start_lock.lock();
        cond_start=queue_start;
        queue_start_lock.unlock();
        
        int cond_end=0;
        queue_end_lock.lock();
        cond_end=queue_end;
        queue_end_lock.unlock();
       
        condition=cond_start<cond_end&&cond_start>=0;
        return condition;
    }
//    bool has_rest(){
//        bool condition=false;
//        condition=queue_start<queue_end&&queue_start>=0;
//        return condition;
//    }
    SampledSubgraph* get_one(){
//        while(true){
//            bool condition=queue_start<queue_end;
//            if(condition){
//                break;
//            }
//         __asm volatile("pause" ::: "memory");  
//        }
        queue_start_lock.lock();
        VertexId id=queue_start++;
        queue_start_lock.unlock();
        assert(id<work_queue.size());
        return work_queue[id];
    }

    void push_one(SampledSubgraph* ssg) {
        work_queue.push_back(ssg);
        queue_end_lock.lock();
        queue_end++;
        queue_end_lock.unlock();
        if(work_queue.size()==1){
            queue_start_lock.lock();
            queue_start=0;
            queue_start_lock.unlock();
        }
    }

    void clear_queue(){
        for(VertexId i=0;i<work_queue.size();i++){
            delete work_queue[i];
        }
        work_queue.clear();
    } 
    bool sample_not_finished(){
        return work_offset<work_range[1];
    }
    void restart(){
        work_offset=work_range[0];
        queue_start=-1;
        queue_end=0;
    }
    void reservoir_sample(int layers_, int batch_size_,std::vector<int> fanout_){
        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);
        SampledSubgraph* ssg=new SampledSubgraph(layers_,fanout_);  
        
        for(int i=0;i<layers_;i++){
            ssg->sample_preprocessing(i);
            //whole_graph->SyncAndLog("preprocessing");
            if(i==0){
              ssg->sample_load_destination([&](std::vector<VertexId>& destination){
                  for(int j=0;j<actual_batch_size;j++){
                    //   destination.push_back(work_offset++);
                    destination.push_back(sample_nids[work_offset++]);
                  }
              },i);
              //whole_graph->SyncAndLog("sample_load_destination");
            }else{
               ssg->sample_load_destination(i); 
              //whole_graph->SyncAndLog("sample_load_destination2");
            }
            ssg->init_co([&](VertexId dst){
                VertexId nbrs=whole_graph->column_offset[dst+1]
                                 -whole_graph->column_offset[dst];
            return (nbrs>fanout_[i]) ? fanout_[i] : nbrs;
            },i);
            ssg->sample_processing([&](VertexId fanout_i,
                    VertexId dst,
                    std::vector<VertexId> &column_offset,
                        std::vector<VertexId> &row_indices,VertexId id){
                for(VertexId src_idx=whole_graph->column_offset[dst];
                        src_idx<whole_graph->column_offset[dst+1];src_idx++){
                    //ReservoirSampling
                    VertexId write_pos=(src_idx-whole_graph->column_offset[dst]);
                    if(write_pos<fanout_i){
                        write_pos+=column_offset[id];
                        row_indices[write_pos]=whole_graph->row_indices[src_idx];
                    }else{
                        VertexId random=rand()%write_pos;
                        if(random<fanout_i){
                          row_indices[random+column_offset[id]]=  
                                  whole_graph->row_indices[src_idx];
                        }
                    }
                }
            });
            //whole_graph->SyncAndLog("sample_processing");
            ssg->sample_postprocessing();
            //whole_graph->SyncAndLog("sample_postprocessing");
        }
        // generate csr for backward graph
        // for (auto p : ssg->sampled_sgs) {
            // p->generate_csr_from_csc();
            // p->debug_generate_csr_from_csc();
        // }
        std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
        push_one(ssg);
    }

    void LayerUniformSample(int layers, int batch_size, std::vector<int> fanout) {
        // construct layer
        assert(work_offset < work_range[1]);
        int actual_batch_size = std::min((VertexId)batch_size, work_range[1] - work_offset);
        // VertexId* indices = whole_graph->srcList;
        VertexId* indices = whole_graph->row_indices;
        VertexId* indptr = whole_graph->column_offset;

        // std::vector<VertexId> layer_offset;
        std::vector<VertexId> node_mapping;
        std::vector<VertexId> actl_layer_sizes;
        // std::vector<float> probabilities;

        std::copy(sample_nids.begin() + work_offset, sample_nids.begin() + work_offset + actual_batch_size, std::back_inserter(node_mapping));
        // LOG_DEBUG("copy %d %d len %d", work_offset, work_offset + actual_batch_size, sample_nids.size());
        work_offset += batch_size;
        actl_layer_sizes.push_back(node_mapping.size());
        VertexId curr = 0;
        VertexId next = node_mapping.size();
        // LOG_DEBUG("start construct layer");
        for (int i = layers - 1; i >= 0; --i) {
            // LOG_DEBUG("layer %d layer_size %d %d %d", i, layer_size, curr, next);
            std::unordered_set<VertexId> candidate_set;
            for (int j = curr; j < next; ++j) {
                VertexId dst = node_mapping[j];
                candidate_set.insert(indices + indptr[dst], indices + indptr[dst + 1]);
            }
            // LOG_DEBUG("candidate_set is done");
            std::vector<VertexId> candidate_vector;
            copy(candidate_set.begin(), candidate_set.end(), std::back_inserter(candidate_vector));
            // LOG_DEBUG("candidate_vector is done");
            int layer_size = std::min(fanout[i], (int)candidate_vector.size());

            std::unordered_map<VertexId, size_t> n_occurrences;
            size_t n_candidates = candidate_vector.size();
            for (int j = 0; j < layer_size; ++j) {
                // TODO(sanzo): thread safe
                // VertexId dst = candidate_vector[RandInt(0, n_candidates)];
                VertexId dst = candidate_vector[rand() % n_candidates];
                if (!n_occurrences.insert(std::make_pair(dst, 1)).second) {
                    ++n_occurrences[dst];
                }
            }
            // LOG_DEBUG("layer node is done");
            for (auto const &pair : n_occurrences) {
                node_mapping.push_back(pair.first);
            }
            // LOG_DEBUG("add node to node_mapping");

            actl_layer_sizes.push_back(node_mapping.size() - next);
            curr = next;
            next = node_mapping.size();
        }

        std::reverse(node_mapping.begin(), node_mapping.end());
        std::reverse(actl_layer_sizes.begin(), actl_layer_sizes.end());
        // std::vector<int64_t> layer_offset;
        // layer_offset.push_back(0);
        // for (auto const &size : actl_layer_sizes) {
        //     layer_offset.push_back(layer_offset.back() + size);
        // }
        // LOG_DEBUG("consruct layer done");
        
        SampledSubgraph* ssg = ConstructSampledSubgraph(layers, actl_layer_sizes, node_mapping);
        push_one(ssg);
    }

    // construct subgraph
    SampledSubgraph* ConstructSampledSubgraph(int layers, std::vector<VertexId>& actl_layer_sizes, std::vector<VertexId>& node_mapping) {
        auto indptr = whole_graph->column_offset;
        auto indices = whole_graph->row_indices;
        SampledSubgraph* ssg=new SampledSubgraph();
        VertexId curr = 0;
        // LOG_DEBUG("start construct subgraph");
        for (int i = 0; i < layers; ++i) {
            // LOG_DEBUG("layer %d", i);
            size_t src_size = actl_layer_sizes[i];
            std::unordered_map<VertexId, VertexId> source_map;
            std::vector<VertexId> sources;
            // TODO(sanzo): redundancy copy
            std::copy(node_mapping.begin() + curr, node_mapping.begin() + curr + src_size, std::back_inserter(sources));
            for (int j = 0; j < src_size; ++j) {
                source_map.insert(std::make_pair(node_mapping[curr + j], j));
            }
            // LOG_DEBUG("source_map is done");

            std::vector<VertexId> sub_edges;
            size_t dst_size = actl_layer_sizes[i + 1];
            std::vector<VertexId> destination;
            // TODO(sanzo): redundancy copy
            std::copy(node_mapping.begin() + curr + src_size, node_mapping.begin() + curr + src_size + dst_size, std::back_inserter(destination));
            std::vector<VertexId> column_offset;
            column_offset.push_back(0);
            // LOG_DEBUG("start select src node dst_size %d", dst_size);
            for (int j = 0; j < dst_size; ++j) {
                VertexId dst = node_mapping[curr + src_size + j];
                // LOG_DEBUG("j %d dst %d", j, dst);
                // std::pair<VertexId, VertexId> id_pair;
                // std::vertex<VertexId> neighbor_indices;
                for (int k = indptr[dst]; k < indptr[dst + 1]; ++k) {
                    if (source_map.find(indices[k]) != source_map.end()) {
                        // source_map.push_back(indices[k]);
                        sub_edges.push_back(source_map[indices[k]]);
                    }
                }
                column_offset.push_back(sub_edges.size());
            }

            sampCSC* sample_sg = new sampCSC(dst_size, sub_edges.size());
            sample_sg->init(column_offset, sub_edges, sources, destination);
            // sample_sg->generate_csr_from_csc();
            ssg->sampled_sgs.push_back(sample_sg);
            // LOG_DEBUG("subgraph is done");
            curr += src_size;
        }
        return ssg;
    }

    void ClusterGCNSample(int layers, int batch_size, int partition_num, std::string objtype = "cut") {
        if (metis_partition_id.empty()) {
            // (FIXME Sanzo) store metis partion result to file
            MetisPartitionGraph(whole_graph, partition_num, objtype, metis_partition_id, metis_partition_offset);
        }
        
        std::vector<int> random_partition(partition_num);
        std::iota(random_partition.begin(), random_partition.end(), 0);
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
        std::shuffle (random_partition.begin(), random_partition.end(), std::default_random_engine(seed));
        std::unordered_set<VertexId> node_set;
        // vector<VertexId> node_ids_check;
        auto& offset = metis_partition_offset;
        auto& ids = metis_partition_id;
        for (int i = 0; i < batch_size; ++i) {
            auto select_node = random_partition[i];
            node_set.insert(select_node);
            node_set.insert(ids.begin() + offset[select_node], ids.begin() + offset[select_node + 1]);
            // for (int j = offset[i]; j < offset[i + 1]; ++j) {
            //     node_ids_check.push_back(ids[j]);
            // }
        }
        // int compare_ret = std::equal(node_ids.begin(), node_ids.end(), node_ids_check.begin());
        // assert(compare_ret);
        // std::cout << "node_ids.size() " << node_ids.size() << std::endl;
        vector<VertexId> node_ids(node_set.begin(), node_set.end());
        // std::cout << "select nodes node_ids " << node_ids.size() << std::endl;

        std::vector<VertexId> column_offset;
        std::vector<VertexId> row_indices;
        column_offset.push_back(0);
        for (auto dst : node_ids) {
            for (int i = whole_graph->column_offset[dst]; i < whole_graph->column_offset[dst + 1]; ++i) {
                int src = whole_graph->row_indices[i];
                if (node_set.find(src) == node_set.end()) continue;
                row_indices.push_back(src);
            }
            column_offset.push_back(row_indices.size());
        }
        
        std::vector<VertexId> actl_layer_sizes(layers + 1, node_ids.size());
        std::vector<VertexId> node_mapping;
        for (int i = 0; i < layers + 1; ++i)
            std::copy(node_ids.begin(), node_ids.end(), std::back_inserter(node_mapping));

        // for (int i = 0; i < layers; ++i) {
        //     std::cout << "layer " << i << " " << actl_layer_sizes[i] << std::endl;
        // }
        // std::cout << "node mapping size " << node_mapping.size() << std::endl;

        SampledSubgraph* ssg = ConstructSampledSubgraph(layers, actl_layer_sizes, node_mapping);
        push_one(ssg);
    }
};


#endif