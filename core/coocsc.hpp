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
#if CUDA_ENABLE
#include "ntsCUDA.hpp"
#endif

#include <vector>
#include <map>
#include <algorithm>
#ifndef COOCSC_HPP
#define COOCSC_HPP

class sampCSC{
public:    
    sampCSC(){
        v_size=0;
        e_size=0;
        column_offset.clear();    
        row_indices.clear();
        row_offset.clear();    
        column_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    sampCSC(VertexId v_, VertexId e_){
        v_size=v_;
        e_size=e_;
        column_offset.resize(v_ + 1);
        row_indices.resize(e_);
        row_offset.resize(e_);    
        column_indices.resize(e_);
        destination.resize(v_);
        source.resize(e_);
        // src_index.clear();
    }

    void init(std::vector<VertexId> &column_offset, std::vector<VertexId> &row_indices,
              std::vector<VertexId> &source, std::vector<VertexId> &destination) {
        this->column_offset = column_offset;
        this->row_indices = row_indices;
        this->source = source;
        this->destination = destination;
    }

    sampCSC(VertexId v_){
        v_size=v_;
        e_size=0;
        // column_offset.clear();    
        column_offset.resize(v_ + 1, 0);
        row_indices.clear();
        row_offset.clear();    
        column_indices.clear();
        src_index.clear();
        // destination.clear();
        destination.resize(v_);
        source.clear();
    }
    // void alloc_index_table(VertexId size) {
    //     node_idx[]
    // }
    ~sampCSC(){
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
        row_offset.clear();    
        column_indices.clear();
    }
    
    void update_degree_of_csc(Graph<Empty> *graph) {
        VertexId* outs = graph->out_degree_for_backward;
        VertexId* ins = graph->in_degree_for_backward;
        #pragma omp parallel for
        for (int i = 0; i < graph->vertices; ++i) {
            outs[i] = 0;
            ins[i] = 0;
        }
        int dst_size = v_size;
        for (int i = 0; i < dst_size; ++i) {
            ins[destination[i]] += column_offset[i + 1] - column_offset[i];
            // #pragma omp parallel for
            for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
                int local_src = row_indices[j];
                outs[source[local_src]]++;
            }
        }
        // long sum_ins = 0, sum_outs = 0;
        // for (int i = 0; i < graph->vertices; ++i) {
        //     sum_ins += ins[i];
        //     sum_outs += outs[i];
        // }
        // assert(sum_ins == sum_outs);
    }

    void update_degree_of_csr(Graph<Empty> *graph) {
        // LOG_DEBUG("update_degree_of_csr");
        VertexId* outs = graph->out_degree_for_backward;
        VertexId* ins = graph->in_degree_for_backward;
        #pragma omp parallel for
        for (int i = 0; i < graph->vertices; ++i) {
            outs[i] = 0;
            ins[i] = 0;
        }
        for (int i = 0; i < src_size; ++i) {
            ins[source[i]] += row_offset[i + 1] - row_offset[i];
            // #pragma omp parallel for
            for (int j = row_offset[i]; j < row_offset[i + 1]; ++j) {
                int local_dst = column_indices[j];
                outs[destination[local_dst]]++;
            }
        }
        // long sum_ins = 0, sum_outs = 0;
        // for (int i = 0; i < graph->vertices; ++i) {
        //     sum_ins += ins[i];
        //     sum_outs += outs[i];
        // }
        // assert(sum_ins == sum_outs);
    }

    void generate_csr_from_csc() {
        // assert(source.size() == destination.size());

        int dst_size = v_size;
        int edge_size = e_size;
        assert(row_offset.size() >= src_size + 1);
        memset(row_offset.data(), 0, sizeof(VertexId) * (src_size + 1));
        #pragma omp parallel for
        for (int i = 0; i < src_size + 1; ++i) {
            assert(row_offset[i] == 0);
        }
        for (int i = 0; i < dst_size; ++i) {
            // #pragma omp parallel for
            for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
                int local_src = row_indices[j];
                row_offset[local_src + 1]++;
            }
        }
        for (int i = 1; i <= src_size; ++i) {
            row_offset[i] += row_offset[i - 1];
        }
        assert(row_offset[src_size] == column_offset[v_size]);
        assert(row_offset[src_size] == e_size);
        std::vector<int> tmp_row_offset(row_offset.begin(), row_offset.end());

        // #pragma omp parallel for
        for (int i = 0; i < dst_size; ++i) {
            // #pragma omp parallel for
            for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
                int local_src = row_indices[j];
                column_indices[tmp_row_offset[local_src]++] = i;
            }
        }
        // LOG_DEBUG("afrt dome");
        // for (int i= 0; i < src_size; ++i) {
        //     assert(tmp_row_offset[i] == row_offset[i + 1]);
        // }
    }

    void postprocessing(){
        src_size=0;
        row_indices_debug.resize(row_indices.size(),0);
        for(VertexId i_src=0;i_src<row_indices.size();i_src++){
          //  printf("debug %d\n",i_src);
            row_indices_debug[i_src]=row_indices[i_src];
            if(0xFFFFFFFF==row_indices[i_src]){
                continue;
            }
            auto iter = src_index.find(row_indices[i_src]);  
            //printf("%d\n",iter == src_index.end());
            if(iter == src_index.end()){   
            //    printf("debug %d\n",i_src);
                src_index.insert(std::make_pair(row_indices[i_src], src_size));
                src_size++;
           //     printf("debug %d\n",i_src);
                source.push_back(row_indices[i_src]);
                row_indices[i_src]=src_size-1;
                //reset src for computation
            }
            else{
                // redundant continue;
                assert(row_indices[i_src]==iter->first);
                row_indices[i_src]=iter->second; //reset src for computation
            }
        }
    }

    void postprocessing (Bitmap* bits){
        // std::unordered_set<int> st;
        // for (VertexId i = 0; i < e_size; ++i) {
        //     VertexId node_id = row_indices[i];
        //     st.insert(node_id);
        //     assert(bits->get_bit(node_id) > 0);
        // }

        // assert(st.size() == bits->get_ones());
        //////////////////////////////

        std::vector<int> node_idx(bits->size, -1);
        src_size = 0;
        for (int i = 0; i < bits->size; ++i) {
            if (bits->get_bit(i) > 0) {
                // source.push_back(i); // TODO(pre-alloc)
                source[src_size] = i;
                node_idx[i] = src_size++;
            }
        }


        #pragma omp parallel for
        // for (size_t i = 0; i < row_indices.size(); ++i) {
        for (VertexId i = 0; i < e_size; ++i) { 
            int src = row_indices[i];
            assert(node_idx[src] != -1);
            row_indices[i] = node_idx[src];
        }
        // LOG_DEBUG("v_size %d e_size %d unique %d src-size %d dst.size %d src.size %d", v_size, e_size, cnt, src_size, dst().size(), src().size());
    }

    void allocate_vertex(){
        destination.resize(v_size,0);       
        column_offset.resize(v_size+1,0);
    }
    void init_dst(VertexId* dst) {
        // LOG_DEBUG("copy to dst size %d", v_size);
        memcpy(destination.data(), dst, sizeof(VertexId) * v_size);
    }
    void allocate_co_from_dst(){
        v_size=destination.size();
        column_offset.resize(v_size+1,0);
    }
    void allocate_edge(){
        assert(0);
        row_indices.resize(e_size,0);
    }
    void allocate_edge(VertexId e_size_){
        e_size=e_size_;
        row_indices.resize(e_size,0);
    }
    void update_edges(VertexId e_size) {
        this->e_size = e_size;
    }
    void update_vertices(VertexId v_size) {
        this->v_size = v_size;
    }
    void allocate_all(){
        allocate_vertex();
        allocate_edge();
    }
    VertexId c_o(VertexId vid){
        return column_offset[vid];
    }
    VertexId r_i(VertexId vid){
        return row_indices[vid];
    }
    VertexId c_i(VertexId vid){
        return column_indices[vid];
    }
    VertexId r_o(VertexId vid){
        return row_offset[vid];
    }
    std::vector<VertexId>& dst(){
        return destination;
    }
    std::vector<VertexId>& src(){
        return source;
    }
    std::vector<VertexId>& c_o(){
        return column_offset;
    }
    std::vector<VertexId>& r_i(){
        return row_indices;
    }
    std::vector<VertexId>& r_o() {
        return row_offset;
    }
    std::vector<VertexId>& c_i() {
        return column_indices;
    }
    VertexId get_distinct_src_size(){
        return src_size;
    }
    void debug(){
        printf("print one layer:\ndst:\t");
        for(int i=0;i<destination.size();i++){
            printf("%d\t",destination[i]);
        }printf("\nc_o:\t");
        for(int i=0;i<column_offset.size();i++){
            printf("%d\t",column_offset[i]);
        }printf("\nr_i:\t");
        for(int i=0;i<row_indices.size();i++){
            printf("%d\t",row_indices[i]);
        }printf("\nrid:\t");
        for(int i=0;i<row_indices_debug.size();i++){
            printf("%d\t",row_indices_debug[i]);
        }printf("\nsrc:\t");
        for(int i=0;i<source.size();i++){
            printf("%d\t",source[i]);
        }printf("\n\n");
    }

    void debug_generate_csr_from_csc() {
        std::vector<std::pair<int, int>> edge_csc, edge_csr;
        int dst_size = v_size;
        for (int i = 0; i < src_size; ++i) {
            for (int j = row_offset[i]; j < row_offset[i + 1]; ++j) {
                edge_csr.push_back(std::make_pair(i, column_indices[j]));
            }
        }

        for (int i = 0; i < dst_size; ++i) {
            for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
                edge_csc.push_back(std::make_pair(row_indices[j], i));
            }
        }
        // LOG_DEBUG("csc.size() %d csr.size() %d", edge_csc.size(), edge_csr.size());
        assert(edge_csc.size() == edge_csr.size());

        auto cmp = [](const auto &l, const auto &r) {
            if (l.first != r.first) {
                return l.first < r.first;
            }
            return l.second < r.second;
        };

        sort(edge_csr.begin(), edge_csr.end(), cmp);
        sort(edge_csc.begin(), edge_csc.end(), cmp);
        int edge_size = edge_csc.size();
        for (int i = 0; i < edge_size; ++i) {
            // printf("(%d %d) - (%d %d)\n", edge_csr[i].first, edge_csr[i].second, edge_csc[i].first, edge_csc[i].second);
            assert(edge_csc[i].first == edge_csr[i].first);
            assert(edge_csc[i].second == edge_csr[i].second);
        }
        // assert(false);
    }
    
    void allocate_dev_array(VertexId vtx_size, VertexId edge_size) {
        // column_offset = (VertexId *)cudaMallocPinned((vtx_size + 1) * sizeof(VertexId));
        // dev_column_offset = (VertexId *)cudaMallocGPU((vtx_size + 1) * sizeof(VertexId));
        // row_offset = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
        // dev_row_offset = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));

        // row_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
        // edge_weight_forward = (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType));
        // dev_row_indices = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
        // dev_edge_weight_forward = (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));

        // column_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId)); 
        // edge_weight_backward = (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType)); 
        // dev_column_indices = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
        // dev_edge_weight_backward = (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));
    }


    // private:
    std::vector<VertexId> row_offset;
    std::vector<VertexId> column_offset;//local offset    
    std::vector<VertexId> row_indices;//local id
    std::vector<VertexId> column_indices;
    // VertexId* column_offset;
    // VertexId* row_offset;
    // VertexId* row_indices;
    // VertexId* column_indices;

    std::vector<VertexId> row_indices_debug;//local id

    std::vector<VertexId> source;//global id
    std::vector<VertexId> destination;//global_id
    // VertexId* source;
    // VertexId* destination;

    std::unordered_map<VertexId,VertexId> src_index;//set

    VertexId v_size; //dst_size
    VertexId e_size; // edge size
    VertexId src_size;//distinct src size

    ValueType* edge_weight_forward;//local id
    ValueType* edge_weight_backward;//local id
    ValueType* dev_edge_weight_forward;
    ValueType* dev_edge_weight_backward;

    VertexId* dev_source;
    VertexId* dev_destination;

    VertexId* dev_column_offset;
    VertexId* dev_row_indices;
    VertexId* dev_column_indices;
    VertexId* dev_row_offset;

    VertexId  size_dev_co, size_dev_ri, size_dev_ewf;
    VertexId  size_dev_ci, size_dev_ro, size_dev_ewb;
};



#endif