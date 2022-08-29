#!/bin/bash

# read hostfile
hosts=()
while read line
do
  hosts[${#hosts[@]}]=$line
done < hostfile

# sync func (filename, dest_dir)
host_num=${#hosts[@]}
echo "host_num ${host_num}"
sync(){
  for((i=1;i<${host_num};i++));  
  do   
    echo "scp -r $1  ${USER}@${hosts[$i]}:$2/"
    scp -r $1  ${USER}@${hosts[$i]}:$2/
  done
}

if [ ! -d "build" ]; then
  mkdir build
  cd build && cmake .. && cd ..
fi

if [ ! -d "log" ]; then
  mkdir log
fi

# sync /root/neutron-sanzo /root
# sync 'hostfile' $(pwd)
# sync data $(pwd)
# exit

new_cfg() {
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
  other_paras="PROC_OVERLAP:0\nPROC_LOCAL:0\nPROC_CUDA:0\nPROC_REP:0\nLOCK_FREE:1\nDECAY_EPOCH:100\nDECAY_RATE:0.97"
  echo -e ${other_paras} >> tmp.cfg
}

cd build && make -j $(nproc) && cd ..

# paras: vertex layer dataset algo fanout batch_size epoch batch_type lr wd dropout
# h=1
# ## cora
# echo "run cora dataset..."
# new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 64 200 0 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/cora_seq.log
# new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 64 200 1 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/cora_shuffle.log
# new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 64 200 2 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/cora_rand.log
# new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 64 200 3 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/cora_low.log
# new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 64 200 4 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/cora_upper.log


# ## citeseer
# echo "run citeseer dataset..."
# new_cfg 3327 3703-128-6 ./data/citeseer/citeseer GCNNEIGHBOR 10-25 64 200 0 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/citeseer_seq.log
# new_cfg 3327 3703-128-6 ./data/citeseer/citeseer GCNNEIGHBOR 10-25 64 200 1 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/citeseer_shuffle.log
# new_cfg 3327 3703-128-6 ./data/citeseer/citeseer GCNNEIGHBOR 10-25 64 200 2 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/citeseer_rand.log
# new_cfg 3327 3703-128-6 ./data/citeseer/citeseer GCNNEIGHBOR 10-25 64 200 3 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/citeseer_low.log
# new_cfg 3327 3703-128-6 ./data/citeseer/citeseer GCNNEIGHBOR 10-25 64 200 4 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/citeseer_upper.log


## pubmed
# echo "run pubmed dataset..."
# new_cfg 19717 500-256-3 ./data/pubmed/pubmed GCNNEIGHBOR 10-25 64 200 0 0.01 0.0001 0.5
# mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/pubmed_seq.log
# new_cfg 19717 500-256-3 ./data/pubmed/pubmed GCNNEIGHBOR 10-25 64 200 1 0.01 0.0001 0.5
# mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/pubmed_shuffle.log
# new_cfg 19717 500-256-3 ./data/pubmed/pubmed GCNNEIGHBOR 10-25 64 200 2 0.01 0.0001 0.5
# mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/pubmed_rand.log
# new_cfg 19717 500-256-3 ./data/pubmed/pubmed GCNNEIGHBOR 10-25 64 200 3 0.01 0.0001 0.5
# mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/pubmed_low.log
# new_cfg 19717 500-256-3 ./data/pubmed/pubmed GCNNEIGHBOR 10-25 64 200 4 0.01 0.0001 0.5
# mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/pubmed_upper.log

h=8
sync './build/nts' $(pwd)/build
## reddit
echo "run reddit dataset..."
new_cfg 232965 602-128-41 ./data/reddit/reddit GCNNEIGHBOR 10-15-25 1024 100 0 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/reddit_seq.log 
new_cfg 232965 602-128-41 ./data/reddit/reddit GCNNEIGHBOR 10-15-25 1024 100 1 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/reddit_shuffle.log
new_cfg 232965 602-128-41 ./data/reddit/reddit GCNNEIGHBOR 10-15-25 1024 100 2 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/reddit_rand.log 
new_cfg 232965 602-128-41 ./data/reddit/reddit GCNNEIGHBOR 10-15-25 1024 100 3 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/reddit_low.log
new_cfg 232965 602-128-41 ./data/reddit/reddit GCNNEIGHBOR 10-15-25 1024 100 4 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/reddit_upper.log 

exit

h=8
sync './build/nts' $(pwd)/build
## arxiv
echo "run arxiv dataset..."
new_cfg 169343 128-256-256-40 ./data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBOR 10-15-25 1024 100 0 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/arxiv_seq.log 
new_cfg 169343 128-256-256-40 ./data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBOR 10-15-25 1024 100 1 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/arxiv_shuffle.log
new_cfg 169343 128-256-256-40 ./data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBOR 10-15-25 1024 100 2 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/arxiv_rand.log 
new_cfg 169343 128-256-256-40 ./data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBOR 10-15-25 1024 100 3 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/arxiv_low.log
new_cfg 169343 128-256-256-40 ./data/ogbn-arxiv/ogbn-arxiv GCNNEIGHBOR 10-15-25 1024 100 4 0.01 0.0001 0.5
sync tmp.cfg $(pwd)
mpiexec -hostfile hostfile -np $h ./build/nts tmp.cfg > ./log/arxiv_upper.log 


# python draw.py

# exit()

# h=8

# run() {
#   # epoch path dataset vertices layer
#   echo "$1 $2 $3 $4 $5"
#   # gcn
#   command="./build/nts GCNEAGER $4 $1 $5 $2.edge $2.feat $2.label $2.mask 0.01 0.0001 0.97 100 0.5"
#   mpirun -hostfile hostfile -np $h $command > ${h}.$3.log

#   command="./build/nts GCNCPUEAGER $4 $1 $5 $2.edge $2.feat $2.label $2.mask 0.01 0.0001 0.97 100 0.5"
#   mpirun -hostfile hostfile -np $h $command > ${h}.$3.log
  
#   # gat  
#   command="./build/nts GATCPUDIST $4 $1 $5 $2.edge $2.feat $2.label $2.mask 0.01 0.0001 0.97 100 0.5"
#   # mpirun -hostfile hostfile -np $h $command >> ${h}.$3.log
#   mpirun -hostfile hostfile -np $h $command > ${h}.$3.log
#   command="./build/nts GATGPUDIST $4 $1 $5 $2.edge $2.feat $2.label $2.mask 0.01 0.0001 0.97 100 0.5"
#   mpirun -hostfile hostfile -np $h $command > ${h}.$3.log
# }

# h=1
# # accacy
# run '300' '/root/dataset/cora/cora' 'cora' 2708 1433-128-7
# run '300' '/root/dataset/pubmed/pubmed' 'pubmed' 19717 500-128-3

# # run '300' '/root/dataset/citeseer/citeseer' 'citeseer' 3327 3703-128-6

# # speed
# h=1
# run '40' '/root/dataset/cora/cora' 'cora' 2708 1433-128-7
# run '40' '/root/dataset/citeseer/citeseer' 'citeseer' 3327 3703-128-6
# run '40' '/root/dataset/pubmed/pubmed' 'pubmed' 19717 500-128-3

# # h=8
# # run '300' '/root/dataset/ogbn-arxiv/ogbn-arxiv' 'ogbn-arxiv' 169343 128-64-40
# # run '300' '/root/dataset/reddit/reddit' 'reddit' 232965 602-256-41
# h=8
# run '40' '/root/dataset/ogbn-arxiv/ogbn-arxiv' 'ogbn-arxiv' 169343 128-64-40
# run '40' '/root/dataset/orkut/orkut' 'orkut' 3072626 320-160-20
# run '40' '/root/dataset/wiki/wiki' 'wiki' 12150976 256-128-16