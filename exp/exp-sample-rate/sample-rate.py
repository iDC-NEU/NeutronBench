import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import copy

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
    'reddit':
    "VERTICES:232965\nEDGE_FILE:../../data/reddit/reddit.edge\nFEATURE_FILE:../../data/reddit/reddit.feat\nLABEL_FILE:../../data/reddit/reddit.label\nMASK_FILE:../../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
    'ogbn-arxiv':
    "VERTICES:169343\nEDGE_FILE:../../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
    'ogbn-products':
    "VERTICES:2449029\nEDGE_FILE:../../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../../data/ogbn-products/ogbn-products.label\nMASK_FILE:../../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
    'AmazonCoBuy_computers':
    "VERTICES:13752\nEDGE_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.edge\nFEATURE_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.feat\nLABEL_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.label\nMASK_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.mask\nLAYERS:767-128-10\n",
    'AmazonCoBuy_photo':
    "VERTICES:7650\nEDGE_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.edge\nFEATURE_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.feat\nLABEL_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.label\nMASK_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.mask\nLAYERS:745-128-8\n",
    'enwiki-links':
    "VERTICES:13593032\nEDGE_FILE:../../data/enwiki-links/enwiki-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'livejournal':
    "VERTICES:4846609\nEDGE_FILE:../../data/livejournal/livejournal.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-large':
    "VERTICES:7489073\nEDGE_FILE:../../data/lj-large/lj-large.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-links':
    "VERTICES:5204175\nEDGE_FILE:../../data/lj-links/lj-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'europe_osm':
    "VERTICES:50912018\nEDGE_FILE:../../data/europe_osm/europe_osm.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dblp-2011':
    "VERTICES:933258\nEDGE_FILE:../../data/dblp-2011/dblp-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'frwiki-2013':
    "VERTICES:1350986\nEDGE_FILE:../../data/frwiki-2013/frwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dewiki-2013':
    "VERTICES:1510148\nEDGE_FILE:../../data/dewiki-2013/dewiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'itwiki-2013':
    "VERTICES:1016179\nEDGE_FILE:../../data/itwiki-2013/itwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'hollywood-2011':
    "VERTICES:1985306\nEDGE_FILE:../../data/hollywood-2011/hollywood-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'enwiki-2016':
    "VERTICES:5088560\nEDGE_FILE:../../data/enwiki-2016/enwiki-2016.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
}


def new_command(
    dataset,
    fanout='2,2',
    valfanout='-1,-1',
    batch_size='6000',
    val_batch_size='6000',
    algo='GCNNEIGHBORGPU',
    epochs='10',
    batch_type='random',
    lr='0.01',
    run='1',
    classes='1',
    lower_fanout=4,
    **kw,
):

    other_config = copy.copy(init_command)
    other_config.append(f'ALGORITHM:{algo}')
    other_config.append(f'FANOUT:{fanout}')
    other_config.append(f'VALFANOUT:{valfanout}')
    other_config.append(f'BATCH_SIZE:{batch_size}')
    other_config.append(f'VALBATCH_SIZE:{val_batch_size}')
    other_config.append(f'EPOCHS:{epochs}')
    other_config.append(f'BATCH_TYPE:{batch_type}')
    other_config.append(f'LEARN_RATE:{lr}')
    other_config.append(f'RUNS:{lr}')
    other_config.append(f'CLASSES:{classes}')
    other_config.append(f'RUNS:{run}')
    other_config.append(f'LOWER_FANOUT:{lower_fanout}')
    for k, v in kw.items():
        other_config.append(f'{k}:{v}')
        print(k, v)
    # assert False
    ret = graph_config[dataset] + '\n'.join(other_config)
    return ret


def create_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def run(dataset, cmd, log_path, suffix=''):
    if not os.path.exists(log_path):
        create_dir(log_path)

    run_time = time.time()
    with open('tmp.cfg', 'w') as f:
        f.write(cmd)

    run_command = f'../build/nts tmp.cfg > {log_path}/{dataset}{suffix}.log'
    print('running: ', run_command)
    os.system(run_command)

    run_time = time.time() - run_time
    print(f'done! cost {run_time:.2f}s')


if __name__ == '__main__':
    create_dir('./build')
    os.system('cd ../build && cmake ../../ && make -j $(nproc) && cd -')

    # datasets = ['AmazonCoBuy_computers', 'AmazonCoBuy_photo', 'reddit', 'ogbn-arxiv','ogbn-products']
    # datasets = ['reddit']
    # datasets = ['ogbn-products']
    # datasets = ['reddit', 'ogbn-arxiv', 'ogbn-products']
    # datasets = ['reddit', 'ogbn-arxiv']
    # datasets = ['AmazonCoBuy_computers', 'ogbn-products', 'AmazonCoBuy_photo']
    # datasets = ['reddit', 'ogbn-products']
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products']

    learn_rates = {
        'reddit': 0.001,
        'ogbn-arxiv': 0.001,
        'ogbn-products': 0.001,
    }

    run_times = {
        'ogbn-arxiv': 300,
        'reddit': 600,
        'ogbn-products': 300,
    }

    # datasets = ['reddit']
    # datasets = ['reddit', 'ogbn-arxiv']
    # datasets = ['reddit', 'ogbn-products']
    # datasets = ['ogbn-products']
    datasets = ['ogbn-arxiv']
    sample_rate = [
        '1.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'
    ]

    batch_sizes = {
        # 'reddit': 65536,
        # 'ogbn-arxiv': 512,
        # 'ogbn-arxiv': 2048,
        # 'ogbn-products': 2048,
        'ogbn-arxiv': 1024,
    }

    for ds in datasets:
        file_path = f'./log/{ds}'
        create_dir(file_path)
        for sr in sample_rate:
            print(sr)
            cmd = new_command(
                ds,
                fanout='10,25',
                batch_size=batch_sizes[ds],
                # valfanout='10,25',
                valfanout='-1,-1',
                val_batch_size=1000000000,
                # val_batch_size=batch_sizes[ds],
                # valfanout='10,25',
                # val_batch_size=batch_sizes[ds],
                algo='GCNNEIGHBORGPU',
                epochs=3,
                batch_type='shuffle',
                lr=0.001,
                RUN_TIME=run_times[ds],
                SAMPLE_RATE=sr,
                # CACHE_POLICY='sample',
                # CACHE_TYPE='gpu_memory',
                # MODE='pipeline',
                lower_fanout=15,
                MODE='zerocopy',
                CACHE_TYPE='gpu_memory',
                CACHE_POLICY='degree',
            )
            run(ds, cmd, file_path, suffix=f'-{batch_sizes[ds]}-{sr}')
