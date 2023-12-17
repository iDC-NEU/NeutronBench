import os
import time
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
]

graph_config = {
  'reddit': "VERTICES:232965\nEDGE_FILE:../../data/reddit/reddit.edge\nFEATURE_FILE:../../data/reddit/reddit.feat\nLABEL_FILE:../../data/reddit/reddit.label\nMASK_FILE:../../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
  'ogbn-arxiv': "VERTICES:169343\nEDGE_FILE:../../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
  'ogbn-products': "VERTICES:2449029\nEDGE_FILE:../../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../../data/ogbn-products/ogbn-products.label\nMASK_FILE:../../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
  'enwiki-links': "VERTICES:13593032\nEDGE_FILE:../../data/enwiki-links/enwiki-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'livejournal': "VERTICES:4846609\nEDGE_FILE:../../data/livejournal/livejournal.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'lj-large': "VERTICES:7489073\nEDGE_FILE:../../data/lj-large/lj-large.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'lj-links': "VERTICES:5204175\nEDGE_FILE:../../data/lj-links/lj-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'europe_osm': "VERTICES:50912018\nEDGE_FILE:../../data/europe_osm/europe_osm.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'dblp-2011': "VERTICES:933258\nEDGE_FILE:../../data/dblp-2011/dblp-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'frwiki-2013': "VERTICES:1350986\nEDGE_FILE:../../data/frwiki-2013/frwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'dewiki-2013': "VERTICES:1510148\nEDGE_FILE:../../data/dewiki-2013/dewiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'itwiki-2013': "VERTICES:1016179\nEDGE_FILE:../../data/itwiki-2013/itwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'hollywood-2011': "VERTICES:1985306\nEDGE_FILE:../../data/hollywood-2011/hollywood-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'enwiki-2016': "VERTICES:5088560\nEDGE_FILE:../../data/enwiki-2016/enwiki-2016.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'road-usa': "VERTICES:23947347\nEDGE_FILE:../../data/road-usa/road-usa.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'amazon': "VERTICES:1569960\nEDGE_FILE:../../data/amazon/amazon.edge\nFEATURE_FILE:../../data/amazon/amazon.feat\nLABEL_FILE:../../data/amazon/amazon.label\nMASK_FILE:../../data/amazon/amazon.mask\nLAYERS:200-128-107\n",
  'rmat': "VERTICES:992712\nEDGE_FILE:../../data/rmat/rmat.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'rmat': "VERTICES:1000000\nEDGE_FILE:../../data/rmat/rmat.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
  'ogbn-papers100M': "VERTICES:111059956\nEDGE_FILE:../../data/ogbn-papers100M/ogbn-papers100M.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:5-4-2\n",
}

# vertex: 992712 edges: 199489178
# 1000000


def new_command(dataset, cache_type, cache_policy, fanout='2,2', batch_size='6000', algo='GCNNEIGHBORGPUEXP3',  
            epochs='10', batch_type='random', lr='0.01', run='1', classes='1', cache_rate='0', **kw):

  other_config = copy.copy(init_command)
  other_config.append(f'ALGORITHM:{algo}')
  other_config.append(f'FANOUT:{fanout}')
  other_config.append(f'BATCH_SIZE:{batch_size}')
  other_config.append(f'EPOCHS:{epochs}')
  other_config.append(f'BATCH_TYPE:{batch_type}')
  other_config.append(f'LEARN_RATE:{lr}')
  other_config.append(f'RUNS:{lr}')
  other_config.append(f'CLASSES:{classes}')
  other_config.append(f'CACHE_TYPE:{cache_type}')
  other_config.append(f'CACHE_POLICY:{cache_policy}')
  other_config.append(f'CACHE_RATE:{cache_rate}')
  other_config.append(f'RUNS:{run}')
  for k,v in kw.items():
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
  



def different_optim(datasets):
  for ds in datasets:
    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='explicit')
    # run(ds, cmd, './log/explicit')

    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='explicit')
    # run(ds, cmd, './log/explicit-sequence')

    # zero_copy (gather)
    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='zerocopy')
    # run(ds, cmd, './log/zerocopy')
    
    # zero_copy (gather) pipeline1
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipeline')
    run(ds, cmd, './log/pipeline-ab1')
  
    # zero_copy (gather) pipeline2
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipelineS')
    run(ds, cmd, './log/pipeline-abS')

    # zero_copy (gather) pipeline3
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3, MODE='pipeline')
    run(ds, cmd, './log/pipeline-ab3')

    # # zero_copy (gather) pipeline1 + sample
    # cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipeline')
    # run(ds, cmd, './log/pipeline1-sample')
    
    # # zero_copy (gather) pipeline1 + degree
    # cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipeline')
    # run(ds, cmd, './log/pipeline1-degree')

    # # zero_copy (gather) pipeline2 + sample
    # cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=2, MODE='pipeline')
    # run(ds, cmd, './log/pipeline2-sample')
    
    # # zero_copy (gather) pipeline2 + degree
    # cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=2, MODE='pipeline')
    # run(ds, cmd, './log/pipeline2-degree')

    # # zero_copy (gather) pipeline3 + sample
    # cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3, MODE='pipeline')
    # run(ds, cmd, './log/pipeline3-sample')
    
    # # zero_copy (gather) pipeline3 + degree
    # cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3, MODE='pipeline')
    # run(ds, cmd, './log/pipeline3-degree')



def explicit_breakdown(datasets):
  for ds in datasets:
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='shuffle', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='explicit')
    run(ds, cmd, './log/explicit')




if __name__ == '__main__':
  create_dir('./build')
  os.system('cd ../build && cmake ../.. && make -j $(nproc) && cd -')
  datasets = ['livejournal', 'lj-large', 'lj-links']
  datasets = ['enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  datasets = ['enwiki-links']
  datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products']
  datasets = ['livejournal', 'lj-large', 'lj-links']
  datasets = ['dewiki-2013', 'frwiki-2013', 'itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'dblp-2011']
  datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  datasets = ['reddit', 'lj-links', 'enwiki-links']
  datasets = ['itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'reddit', 'lj-links', 'enwiki-links']
  datasets = ['itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'reddit', 'lj-links']
  datasets = ['reddit']
  datasets = ['hollywood-2011', 'lj-links', 'reddit', 'enwiki-2016', 'enwiki-links']
  datasets = ['enwiki-2016']
  datasets = ['hollywood-2011', 'reddit']
  datasets = ['ogbn-arxiv']
  datasets = ['ogbn-products']
  datasets = ['ogbn-arxiv', 'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links', 'ogbn-products']

  # explicit_breakdown(datasets)
  different_optim(datasets)