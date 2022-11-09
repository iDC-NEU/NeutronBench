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
  echo -e "RUN_TIME:${16}" >> tmp.cfg

  other_paras="PROC_OVERLAP:0\nPROC_LOCAL:0\nPROC_CUDA:0\nPROC_REP:0\nLOCK_FREE:1\nDECAY_EPOCH:100\nDECAY_RATE:0.97"
  echo -e ${other_paras} >> tmp.cfg
}

cd build && make -j $(nproc) && cd ..

log_path="./log/batch-size/full-batch-T50"

function batch_size() {
  if [ ! -d ${log_path} ]; then
    mkdir -p ${log_path}
  fi

  dataset=${4##*/}
  echo "dataset: ${dataset}"

  for bs in $1; do
    echo "run ${dataset} size $bs..."
    new_cfg $2 $3 $4 $5 $6 $bs $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16}
    echo "mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "${log_path}/${dataset}_${bs}.log""
    mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "${log_path}/${dataset}_${bs}.log"
  done
}

# cora
# array=('32' '64' '140')
# batch_size "${array[*]}" 2708 1433-128-7 ../data/cora/cora GCNNEIGHBOR -1,-1 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# # citeseer
# array=('32' '64' '120')
# batch_size "${array[*]}" 3327 3703-128-6 ../data/citeseer/citeseer GCNNEIGHBOR -1,-1 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# # pubmed
# array=('32' '60')
# batch_size "${array[*]}" 19717 500-128-3 ../data/pubmed/pubmed GCNNEIGHBOR -1,-1 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# ppi
# # 44906/6514/5524
# array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8192' '9716')
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '9716')
batch_size "${array[*]}" 14755 50-128-121 ../data/ppi/ppi GCNNEIGHBORGPU -1,-1 50 shuffle 0.01 0.0001 0.5 1 0 1 121 50


# ppi-large
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8192' '16384' '44906')
batch_size "${array[*]}" 56944 50-128-121 ../data/ppi-large/ppi-large GCNNEIGHBORGPU -1,-1 50 shuffle 0.01 0.0001 0.5 1 0 1 121 50


# Flickr
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8192' '16384' '44625')
batch_size "${array[*]}" 89250 500-128-7 ../data/flickr/flickr GCNNEIGHBORGPU -1,-1 50 shuffle 0.01 0.0001 0.5 1 0 1 1 50


# AmazonCoBuyComputer
# 8250 4125 1377
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8250')
batch_size "${array[*]}" 13752 767-128-10 ../data/AmazonCoBuy_computers/AmazonCoBuy_computers GCNNEIGHBORGPU -1,-1 50 shuffle 0.01 0.0001 0.5 1 0 1 1 50


# AmazonCoBuyPhoto
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4590')
batch_size "${array[*]}" 7650 745-128-8 ../data/AmazonCoBuy_photo/AmazonCoBuy_photo GCNNEIGHBORGPU -1,-1 50 shuffle 0.01 0.0001 0.5 1 0 1 1 50


# # Reddit
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '153431')
batch_size "${array[*]}" 232965 602-128-41 ../data/reddit/reddit GCNNEIGHBORGPU -1,-1 50 shuffle 0.01 0.0001 0.5 1 0 1 1 50

# # Arxiv
array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '90941')
batch_size "${array[*]}" 169343 128-128-40 ../data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBORGPU -1,-1 100 shuffle 0.01 0.0001 0.5 1 0 1 1 50

# Yelp
# array=('1' '2' '4' '8' '16' '32' '64' '128' '256' '512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '262144' '537635')
# batch_size "${array[*]}" 716847 300-128-100 ../data/yelp/yelp GCNNEIGHBORGPU -1,-1 100 shuffle 0.01 0.0001 0.5 1 0 3 100

