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

  other_paras="PROC_OVERLAP:0\nPROC_LOCAL:0\nPROC_CUDA:0\nPROC_REP:0\nLOCK_FREE:1\nDECAY_EPOCH:100\nDECAY_RATE:0.97"
  echo -e ${other_paras} >> tmp.cfg
}

cd build && make -j $(nproc) && cd ..

function batch_size() {
  if [ ! -d "./log/batch-size" ]; then
    mkdir -p ./log/batch-size
  fi

  dataset=${4##*/}
  echo "dataset: ${dataset}"

  for bs in $1; do
    echo "run ${dataset} size $bs..."
    new_cfg $2 $3 $4 $5 $6 $bs $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
    echo "mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "./log/batch-size/${dataset}_${bs}.log""
    mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > "./log/batch-size/${dataset}_${bs}.log"
  done
}

# cora
array=('32' '64' '140')
batch_size "${array[*]}" 2708 1433-128-7 ../data/cora/cora GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# citeseer
array=('32' '64' '120')
batch_size "${array[*]}" 3327 3703-128-6 ../data/citeseer/citeseer GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# pubmed
array=('32' '60')
batch_size "${array[*]}" 19717 500-128-3 ../data/pubmed/pubmed GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# ppi
# 44906/6514/5524
array=('512' '1024' '2048' '4096' '8192' '9716')
batch_size "${array[*]}" 14755 50-128-121 ../data/ppi/ppi GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 121

# ppi-large
array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '44906')
batch_size "${array[*]}" 56944 50-128-121 ../data/ppi-large/ppi-large GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 121

# Flickr
array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '44625')
batch_size "${array[*]}" 89250 500-128-7 ../data/flickr/flickr GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# AmazonCoBuyComputer
# 8250 4125 1377
array=('512' '1024' '2048' '4096' '8192' '8250')
batch_size "${array[*]}" 13752 767-128-10 ../data/AmazonCoBuy_computers/AmazonCoBuy_computers GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# AmazonCoBuyPhoto
# 4590
array=('512' '1024' '2048' '4096' '4590')
batch_size "${array[*]}" 7650 745-128-8 ../data/AmazonCoBuy_photo/AmazonCoBuy_photo GCNNEIGHBOR 15,25 200 shuffle 0.01 0.0001 0.5 0 0 3 1

# Reddit
# 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, full
array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '232965')
batch_size "${array[*]}" 232965 602-128-41 ../data/reddit/reddit GCNNEIGHBOR 15,25 100 shuffle 0.01 0.0001 0.5 0 0 3 1

# Yelp
array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '262144' '537635')
batch_size "${array[*]}" 716847 300-128-100 ../data/yelp/yelp GCNNEIGHBOR 15,25 100 shuffle 0.01 0.0001 0.5 0 0 3 100

# Arxiv
array=('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '90941')
batch_size "${array[*]}" 169343 128-128-40 ../data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBOR 15,25 100 shuffle 0.01 0.0001 0.5 0 1 3 1