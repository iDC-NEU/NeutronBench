
# setup


> 0. some dependencies

python version 3.8

pip install -r requirements.txt


> 1. install metis

```bash
cd neutron-sanzo/exp/Partition

git clone git@github.com:MITIBMxGraph/METIS-GKlib.git
cd METIS-GKlib
make config shared=1 cc=gcc prefix=$(realpath ../pkgs) i64=1 r64=1 gklib_path=GKlib/
make install
cd ..
```


> 2. install torch-metis

```bash
cd neutron-sanzo/exp/Partition

git clone git@github.com:MITIBMxGraph/torch-metis.git
cd torch-metis
python setup.py install
cd ..
```


## run

```bash
cd partition
python python main.py --dataset ogbn-arxiv --num_parts 4 --fanout 10 25 --batch_size 6000

```