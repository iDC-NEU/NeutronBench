import sys
import os
import time
import utils
import numpy as np
import matplotlib.pyplot as plt

init_command = [
    "WEIGHT_DECAY:0.0001",
    "DROP_RATE:0.5",
    "DECAY_RATE:0.97",
    "DECAY_EPOCH:100",
    "PROC_OVERLAP:0",
    "PROC_LOCAL:0",
    "PROC_CUDA:0",
    "PROC_REP:0",
    "LOCK_FREE:1",
    "TIME_SKIP:3",
    "MINI_PULL:1",
    "BATCH_NORM:0",
    "PROC_REP:0",
    "LOCK_FREE:1",
    "CACHE_TYPE:none",
    "CACHE_POLICY:none",
    "CACHE_RATE:0",
]

graph_config = {
    'reddit': "VERTICES:232965\nEDGE_FILE:../data/reddit/reddit.edge\nFEATURE_FILE:../data/reddit/reddit.feat\nLABEL_FILE:../data/reddit/reddit.label\nMASK_FILE:../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
    'ogbn-arxiv': "VERTICES:169343\nEDGE_FILE:../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
    'ogbn-products': "VERTICES:2449029\nEDGE_FILE:../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../data/ogbn-products/ogbn-products.label\nMASK_FILE:../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
    'AmazonCoBuy_computers': "VERTICES:13752\nEDGE_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.edge\nFEATURE_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.feat\nLABEL_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.label\nMASK_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.mask\nLAYERS:767-128-10\n",
    'AmazonCoBuy_photo': "VERTICES:7650\nEDGE_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.edge\nFEATURE_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.feat\nLABEL_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.label\nMASK_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.mask\nLAYERS:745-128-8\n",
    'enwiki-links': "VERTICES:13593032\nEDGE_FILE:../data/enwiki-links/enwiki-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'livejournal': "VERTICES:4846609\nEDGE_FILE:../data/livejournal/livejournal.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-large': "VERTICES:7489073\nEDGE_FILE:../data/lj-large/lj-large.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-links': "VERTICES:5204175\nEDGE_FILE:../data/lj-links/lj-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'europe_osm': "VERTICES:50912018\nEDGE_FILE:../data/europe_osm/europe_osm.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dblp-2011': "VERTICES:933258\nEDGE_FILE:../data/dblp-2011/dblp-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'frwiki-2013': "VERTICES:1350986\nEDGE_FILE:../data/frwiki-2013/frwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dewiki-2013': "VERTICES:1510148\nEDGE_FILE:../data/dewiki-2013/dewiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'itwiki-2013': "VERTICES:1016179\nEDGE_FILE:../data/itwiki-2013/itwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'hollywood-2011': "VERTICES:1985306\nEDGE_FILE:../data/hollywood-2011/hollywood-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'enwiki-2016': "VERTICES:5088560\nEDGE_FILE:../data/enwiki-2016/enwiki-2016.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
}


def new_command(
    dataset,
    fanout='2,2',
    valfanout='-1,-1',
    batch_size='6000',
    algo='GCNNEIGHBORGPU',
    epochs='10',
    batch_type='random',
    lr='0.01',
    run='1',
    classes='1',
    **kw,
):

    other_config = init_command
    other_config.append(f'ALGORITHM:{algo}')
    other_config.append(f'FANOUT:{fanout}')
    other_config.append(f'VALFANOUT:{valfanout}')
    other_config.append(f'BATCH_SIZE:{batch_size}')
    other_config.append(f'EPOCHS:{epochs}')
    other_config.append(f'BATCH_TYPE:{batch_type}')
    other_config.append(f'LEARN_RATE:{lr}')
    other_config.append(f'RUNS:{lr}')
    other_config.append(f'CLASSES:{classes}')
    other_config.append(f'RUNS:{run}')
    for k, v in kw.items():
        other_config.append(f'{k}:{v}')
        print(k, v)
    # assert False
    ret = graph_config[dataset] + '\n'.join(init_command)
    return ret



def get_partition_statistic(dataset, num_parts, fanout, batch_size, dim, log_file):
    if not os.path.exists(os.path.dirname(log_file)):
       os.makedirs(os.path.dirname(log_file))

    run_command = f'python main.py --dataset {dataset} --num_parts {num_parts} --fanout {fanout} --batch_size {batch_size} --dim {dim} > {log_file}'
    print('start running: ', run_command)
    os.system(run_command)


def draw_partition_statistic(dataset, log_file, figpath, suffix):
    assert os.path.exists(log_file), log_file
    if not os.path.exists(figpath):
       os.makedirs(figpath)
    
    run_command = f'python draw.py --dataset {dataset}  --log {log_file} --figpath {figpath} --suffix {suffix}'
    print('start running: ', run_command)
    os.system(run_command)



if __name__ == '__main__':

    datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'computer']
    datasets = ['computer', 'ogbn-arxiv']
    batch_sizes = {
        # 'AmazonCoBuy_computers': (512, 1024, 2048, 4096, 8250),
        'computer': 1024,
        'ogbn-arxiv': 6000,
        'reddit': 6000,
        'ogbn-products': 6000,
    }

    num_parts = 4
    fanout = '10 25'
    # batch_size = 1024
    dim = 4

    for ds in datasets:
      for dim in [1, 2, 4]:
        log_file = f'./log/{ds}/{ds}-{dim}.log'
        get_partition_statistic(ds, num_parts, fanout, batch_sizes[ds], dim, log_file)
        draw_partition_statistic(ds, log_file, f'./log/{ds}', f'dim{dim}')
