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
        column_offset.clear();    
        row_indices.clear();
        row_offset.clear();    
        column_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
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
        column_offset.clear();    
        row_indices.clear();
        row_offset.clear();    
        column_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    ~sampCSC(){
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
        row_offset.clear();    
        column_indices.clear();
    }
    void generate_csr_from_csc() {
        // assert(source.size() == destination.size());

        int dst_size = destination.size();
        int src_size = source.size();
        int edge_size = row_indices.size();
        // printf("column_offsete size %d dst_size %d src_size %d\n", column_offset.size(), dst_size, src_size);
        // assert(column_offset.size() == dst_size + 1);
        // assert(edge_size == row_indices.size());
        row_offset.resize(src_size + 1, 0);
        column_indices.resize(edge_size);
        
        for (int i = 0; i < dst_size; ++i) {
            for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
                int local_src = row_indices[j];
                // assert(local_src < source.size());
                row_offset[local_src + 1]++;
            }
        }
        for (int i = 1; i <= src_size; ++i) {
            row_offset[i] += row_offset[i - 1];
        }
        // int sum = std::accumulate(row_offset.begin(), row_offset.end(), 0);
        // std::cout << "edge size " << edge_size << " sum " << sum << std::endl;
        // assert(sum == edge_size);
        std::vector<int> tmp_row_offset(row_offset.begin(), row_offset.end());
        // sum = std::accumulate(tmp_row_offset.begin(), tmp_row_offset.end(), 0);
        // assert(sum == edge_size);
        for (int i = 0; i < dst_size; ++i) {
            for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
                int local_src = row_indices[j];
                column_indices[tmp_row_offset[local_src]++] = i;
            }
        }
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
            std::map<VertexId, VertexId>::iterator iter;
            iter = src_index.find(row_indices[i_src]);  
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
    void allocate_vertex(){
        destination.resize(v_size,0);       
        column_offset.resize(v_size+1,0);
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
        int src_size = source.size();
        int dst_size = destination.size();
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
        // std::cout << "edge_csc.size " << edge_csc.size() << " edge_csr.size " << edge_csr.size() << std::endl;
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
        assert(false);
    }
    
private:
std::vector<VertexId> column_offset;//local offset    
std::vector<VertexId> row_offset;
std::vector<VertexId> row_indices;//local id
std::vector<VertexId> column_indices;
std::vector<VertexId> row_indices_debug;//local id

std::vector<VertexId> source;//global id
std::vector<VertexId> destination;//global_id

std::map<VertexId,VertexId> src_index;//set

VertexId v_size; //dst_size
VertexId e_size; // edge size
VertexId src_size;//distinct src size
};



#endif