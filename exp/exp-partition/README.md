## Computation and Communication load

```
python exp-partition.py
```


## Accuracy and Convergence speed

As described in Section 4, `Evaluating the System``, we use DistDGL as the experimental system for the accuracy and convergence speed experiments.


first you need to partition the graph to prepare the distdgl training, for the detail reference to 

> generate partition results

```bash
cd ~/NeutronBench/exp/Partition/partition

# make the partition_result directory
mkdir partition_result

# generate partition result for Stream-B
bash bytegnn_partition.sh
```


> partition the graph for DistDGL training


```bash
cd ~/NeutronBench/exp/Partition/partition

# Hash
python partition_graph.py --dataset ogbn-arxiv --num_parts 4 --mode hash

# Metis-V
python partition_graph.py --dataset ogbn-arxiv --num_parts 4 --mode metis --dim 1

# Metis-VE
python partition_graph.py --dataset ogbn-arxiv --num_parts 4 --mode metis --dim 2

# Metis-VET
python partition_graph.py --dataset ogbn-arxiv --num_parts 4 --mode metis --dim 4

# Stream-V
python partition_graph.py --dataset ogbn-arxiv --num_parts 4 --mode pagraph

# Stream-B
python partition_graph.py --dataset ogbn-arxiv --num_parts 4 --mode bytegnn
```


> start the training


The training code is in `~/NeutronBench/exp/Partition/partition/train_dist.py`, please follow the [DistDGL training documentation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist) to run the code.
