#!/bin/bash  

g++ -std=c++14 -pthread  bytegnn_partition.cpp  -o bytegnn


dataset_list=('cora' 'ogbn-arxiv' 'ogbn-products' 'reddit')

for dataset in ${dataset_list[*]}
do  
  ./bytegnn ~/neutron-sanzo/data/$dataset $dataset  4 2 ./partition_result/bytegnn
done



# ./bytegnn ~/neutron-sanzo/data/cora cora  4 2 ./partition_result/bytegnn
# ./bytegnn ~/neutron-sanzo/data/ogbn-arxiv ogbn-arxiv  4 2 ./partition_result/bytegnn
# ./bytegnn ~/neutron-sanzo/data/cora cora  4 2 ./partition_result/bytegnn
# ./bytegnn ~/neutron-sanzo/data/cora cora  4 2 ./partition_result/bytegnn
