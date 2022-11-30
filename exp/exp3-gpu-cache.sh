#!/bin/bash

# read hostfile
# hosts=()
# while read line
# do
#   hosts[${#hosts[@]}]=$line
# done < hostfile

# sync func (filename, dest_dir)
# host_num=${#hosts[@]}
# echo "host_num ${host_num}"

function sync(){
  for((i=1;i<${host_num};i++));  
  do   
    echo "scp -r $1  ${USER}@${hosts[$i]}:$2/"
    scp -r $1  ${USER}@${hosts[$i]}:$2/
  done
}

if [ ! -d "build" ]; then
  mkdir build && cd build && cmake ../.. && cd ..
fi

if [ ! -d "log" ]; then
  mkdir log
fi

# sync /root/neutron-sanzo /root
# sync 'hostfile' $(pwd)
# sync data $(pwd)
# exit

function new_cfg() {
  echo -e "VERTICES:$1" > tmp.cfg
  echo -e "LAYERS:$2" >> tmp.cfg
  echo -e "EDGE_FILE:$3.edge" >> tmp.cfg
  echo -e "FEATURE_FILE:$3.feat" >> tmp.cfg
  echo -e "LABEL_FILE:$3.label" >> tmp.cfg
  echo -e "MASK_FILE:$3.mask" >> tmp.cfg
  echo -e "ALGORITHM:$4" >> tmp.cfg
  echo -e "FANOUT:$5" >> tmp.cfg
  echo -e "BATCH_SIZE:$6" >> tmp.cfg
  echo -e "EPOCHS:$7" >> tmp.cfg
  echo -e "BATCH_TYPE:$8" >> tmp.cfg
  echo -e "LEARN_RATE:$9" >> tmp.cfg
  echo -e "WEIGHT_DECAY:${10}" >> tmp.cfg
  echo -e "DROP_RATE:${11}" >> tmp.cfg
  echo -e "MINI_PULL:${12}" >> tmp.cfg
  echo -e "BATCH_NORM:${13}" >> tmp.cfg
  echo -e "TIME_SKIP:3" >> tmp.cfg
  echo -e "RUNS:${14}" >> tmp.cfg
  echo -e "CLASSES:${15}" >> tmp.cfg
  # echo -e "RUN_TIME:${16}" >> tmp.cfg
  echo -e "CACHE_RATE:${16}" >> tmp.cfg

  other_paras="PROC_OVERLAP:0\nPROC_LOCAL:0\nPROC_CUDA:0\nPROC_REP:0\nLOCK_FREE:1\nDECAY_EPOCH:100\nDECAY_RATE:0.97"
  echo -e ${other_paras} >> tmp.cfg
}

cd build && make -j $(nproc) && cd ..

log_path="./log/batch-size/epoch-10-25"

function batch_size() {
  if [ ! -d ${log_path} ]; then
    mkdir -p ${log_path}
  fi

  dataset=${4##*/}
  echo "dataset: ${dataset}"

  for bs in $1; do
    echo "run ${dataset} size $bs..."
    new_cfg $2 $3 $4 $5 $6 $bs $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16}
    # echo "mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "${log_path}/${dataset}_${bs}.log""
    # mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "${log_path}/${dataset}_${bs}.log"
    echo "mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg >> "${log_path}/${dataset}.log""
    mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg >> "${log_path}/${dataset}.log"
  done
}


# cora
# array=('32' '64' '140')
# batch_size "${array[*]}" 2708 1433-128-7 ../data/cora/cora GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# # citeseer
# array=('32' '64' '120')
# batch_size "${array[*]}" 3327 3703-128-6 ../data/citeseer/citeseer GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

function run_batch_size_mem() {
  # # Reddit
  # # 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, full
  array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '153431')
  # array=('153431' '512' '2048' '8192' '32768' '131072')
  batch_size "${array[*]}" 232965 602-128-41 ../data/reddit/reddit GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

  # Arxiv
  # array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '90941')
  array=('512' '3072' '6144' '12288' '24576' '49152' '90941')
  # array=('24576' '49152')
  # # array=('90941')
  batch_size "${array[*]}" 169343 128-128-40 ../data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

  # # Ogbn-products
  # array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '196615')
  # batch_size "${array[*]}" 2449029 100-128-47 ../data/ogbn-products/ogbn-products GCNNEIGHBORGPUEXP3 $1 200 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

   # pubmed
  # array=('128' '512' '1024' '2048' '4096' '8192' '11830')
  # array=('128' '512' '1024')
  # batch_size "${array[*]}" 19717 500-128-3 ../data/pubmed/pubmed GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

  # # AmazonCoBuyComputer
  # array=('512' '1024' '2048' '4096' '8250')
  # batch_size "${array[*]}" 13752 767-128-10 ../data/AmazonCoBuy_computers/AmazonCoBuy_computers GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

  # # AmazonCoBuyPhoto
  # array=('512' '1024' '2048' '4590')
  # batch_size "${array[*]}" 7650 745-128-8 ../data/AmazonCoBuy_photo/AmazonCoBuy_photo GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2
}


function cache_mem() {
  if [ ! -d ${log_path} ]; then
    mkdir -p ${log_path}
  fi

  dataset=${4##*/}
  echo "dataset: ${dataset}"

  for bs in $1; do
    echo "run ${dataset} cache $bs..."
    new_cfg $2 $3 $4 $5 $6 ${16} $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${bs}
    # echo "mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "${log_path}/${dataset}_${bs}.log""
    # mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "${log_path}/${dataset}_${bs}.log"
    echo "mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg >> "${log_path}/${dataset}.log""
    mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg >> "${log_path}/${dataset}.log"
  done
}

function run_cache_mem() {
  # # Reddit
  # # 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, full
  # array=('153431' '512' '2048' '8192' '32768' '131072')
  array=('0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0')
  cache_mem "${array[*]}" 232965 602-128-41 ../data/reddit/reddit GCNNEIGHBORGPUEXP3 $1 10 sequence 0.01 0.0001 0.5 1 0 1 1 512

  # Arxiv
  # array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '90941')
  array=('0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0')
  cache_mem "${array[*]}" 169343 128-128-40 ../data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBORGPUEXP3 $1 10 sequence 0.01 0.0001 0.5 1 0 1 1 512

  # # Ogbn-products
  # array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '196615')
  # cache_mem "${array[*]}" 2449029 100-128-47 ../data/ogbn-products/ogbn-products GCNNEIGHBORGPUEXP3 $1 200 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

   # pubmed
  # array=('128' '512' '1024' '2048' '4096' '8192' '11830')
  # array=('128' '512' '1024')
  # cache_mem "${array[*]}" 19717 500-128-3 ../data/pubmed/pubmed GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

  # # AmazonCoBuyComputer
  # array=('512' '1024' '2048' '4096' '8250')
  # cache_mem "${array[*]}" 13752 767-128-10 ../data/AmazonCoBuy_computers/AmazonCoBuy_computers GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2

  # # AmazonCoBuyPhoto
  # array=('512' '1024' '2048' '4590')
  # cache_mem "${array[*]}" 7650 745-128-8 ../data/AmazonCoBuy_photo/AmazonCoBuy_photo GCNNEIGHBORGPUEXP3 $1 10 shuffle 0.01 0.0001 0.5 1 0 1 1 $2
}

# log_path="./log/gpu-cache/batch-size-mem/epoch-4-8"
# run_batch_size_mem 4,8 0

log_path="./log/gpu-cache/cache-mem/4-8"
run_cache_mem 4,8

# log_path="./log/batch-size/dabian-val-full/epoch-2-2"
# run_batch_size 2,2

# log_path="./log/batch-size/dabian-val-mini/epoch-11"
# run_batch_size -1,-1

#  pre_time = {
#             # 'cora': 200, 'citeseer': 200, 'pubmed': 200, 
#             'AmazonCoBuy_computers': 7, 'AmazonCoBuy_photo': 7,
#             'ogbn-arxiv': 30, 'pubmed': 3, 'reddit': 50, 
#             'ogbn-products': 200
#             # 'reddit': 100, 'yelp': 100, 'ogbn-arxiv': 100,
#            }   

# # Yelp
# array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '262144' '537635')
# batch_size "${array[*]}" 716847 300-128-100 ../data/yelp/yelp GCNNEIGHBORGPU 15,25 100 shuffle 0.01 0.0001 0.5 1 0 3 100

# # ppi
# # 44906/6514/5524
# array=('512' '1024' '2048' '4096' '8192' '9716')
# batch_size "${array[*]}" 14755 50-128-121 ../data/ppi/ppi GCNNEIGHBORGPU 15,25 200 shuffle 0.01 0.0001 0.5 1 0 3 121

# # ppi-large
# array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '44906')
# batch_size "${array[*]}" 56944 50-128-121 ../data/ppi-large/ppi-large GCNNEIGHBORGPU 15,25 200 shuffle 0.01 0.0001 0.5 1 0 3 121


# # Flickr
# array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '44625')
# batch_size "${array[*]}" 89250 500-128-7 ../data/flickr/flickr GCNNEIGHBORGPU 15,25 200 shuffle 0.01 0.0001 0.5 1 0 3 1
