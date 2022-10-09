#!/bin/bash

# read hostfile
# hosts=()
# while read line
# do
#   hosts[${#hosts[@]}]=$line
# done < hostfile

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

  h=1
  size_num=$1
  echo $1
  arg_arry=($*)
  i=1
  for((i=1;i<=$size_num;i++));
  do
    echo "传递参数位置: " $i
    echo "参数值：" ${arg_arry[$i]}
    new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 ${arg_arry[$i]} 10 0 0.01 0.0001 0.5
    mpiexec -np $h ./build/nts tmp.cfg >> ./log/cora_seq.log
  done
                                  
# bash demo.sh 4 64 128 256 512


# # sync './build/nts' $(pwd)/build
# ## cora
# echo "run cora dataset..."
# new_cfg 2708 1433-64-7 ./data/cora/cora GCNNEIGHBOR 10-25 64 200 0 0.01 0.0001 0.5
# mpiexec -np $h ./build/nts tmp.cfg > ./log/cora_seq.log



  # size_num=$1
  # arg_arry=($*)
  # i=1
  # for((i=1;i<=$size_num;i++));
  # do
  #   echo "传递参数位置: " $i
  #   echo "参数值：" ${arg_arry[$i]}
  # done

