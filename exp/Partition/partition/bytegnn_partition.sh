#!/bin/bash  
g++ -std=c++14 -pthread  bytegnn_partition.cpp  -o bytegnn

dataset_list=('ogbn-arxiv' 'ogbn-products' 'reddit' 'amazon')

data_dir=~/NeutronBench/data

for dataset in ${dataset_list[*]}
do  
  # Usage: ./bytegnn input_dir dataname num_parts num_hops out_dir\n
  ./bytegnn $data_dir/$dataset $dataset  4 2 ./partition_result
done