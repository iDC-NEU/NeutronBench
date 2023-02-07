import sys
import os
import time
import numpy as np
import utils

# datasets = ['ppi', 'ppi-large', 'reddit', 'flickr', 'yelp', 'amazon']
# batch_size = {'ppi':4096, 'ppi-large':4096, 'flickr':40960, 'yelp':40960, 'amazon':40960, 'reddit':40960}

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
  'reddit': "VERTICES:232965\nEDGE_FILE:../data/reddit/reddit.edge\nFEATURE_FILE:../data/reddit/reddit.feat\nLABEL_FILE:../data/reddit/reddit.label\nMASK_FILE:../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
  'ogbn-arxiv': "VERTICES:169343\nEDGE_FILE:../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
  'ogbn-products': "VERTICES:2449029\nEDGE_FILE:../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../data/ogbn-products/ogbn-products.label\nMASK_FILE:../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
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



def new_command(dataset, cache_type, cache_policy, fanout='2,2', batch_size='6000', algo='GCNNEIGHBORGPUEXP3',  
            epochs='10', batch_type='random', lr='0.01', run='1', classes='1', cache_rate='0', **kw):

  other_config = init_command
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
  ret = graph_config[dataset] + '\n'.join(init_command)
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
  
  run_command = f'mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > {log_path}/{dataset}{suffix}.log'
  print('running: ', run_command)
  os.system(run_command)

  run_time = time.time() - run_time
  print(f'done! cost {run_time:.2f}s')
  

def exp3(datasets, fanout):
  # # explicit
  # create_dir('./log/gpu-cache/explicit')
  # for ds in datasets:
  #   cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=2, epochs=3, PIPELINES=1)
  #   run(ds, cmd, './log/gpu-cache/explicit')

  
  # zero_copy (gather) pipeline3
  create_dir('./log/gpu-cache/pipeline1')
  for ds in datasets:
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1)
    run(ds, cmd, './log/gpu-cache/pipeline1')
  
  
  # zero_copy (gather) pipeline2
  create_dir('./log/gpu-cache/pipeline2')
  for ds in datasets:
    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', epochs=13)
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=2)
    run(ds, cmd, './log/gpu-cache/pipeline2')


  # zero_copy (gather) pipeline3
  create_dir('./log/gpu-cache/pipeline3')
  for ds in datasets:
    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', epochs=13)
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3)
    run(ds, cmd, './log/gpu-cache/pipeline3')


  # # zero_copy (gather) pipeline1 + cache
  create_dir('./log/gpu-cache/pipeline1-sample')
  for ds in datasets:
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1)
    run(ds, cmd, './log/gpu-cache/pipeline1-sample')

  # zero_copy (gather) pipeline1 + cache 
  create_dir('./log/gpu-cache/pipeline1-degree')
  for ds in datasets:
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1)
    run(ds, cmd, './log/gpu-cache/pipeline1-degree')


def different_optim():
  datasets = ['lj-links',]
  datasets = ['livejournal', 'lj-large', 'lj-links', 'enwiki-links']
  datasets = ['europe_osm', 'ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']

  
  for ds in datasets:
    # explicit
    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='explicit')
    # run(ds, cmd, './log/gpu-cache/explicit')

    # # zero_copy (gather)
    # cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='zerocopy')
    # run(ds, cmd, './log/gpu-cache/zerocopy')
    
    # zero_copy (gather) pipeline1
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline1')
  
    # zero_copy (gather) pipeline2
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=2, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline2')

    # zero_copy (gather) pipeline3
    cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline3')

    # zero_copy (gather) pipeline1 + sample
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline1-sample')
    
    # zero_copy (gather) pipeline1 + degree
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline1-degree')

    # zero_copy (gather) pipeline2 + sample
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=2, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline2-sample')
    
    # zero_copy (gather) pipeline2 + degree
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=2, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline2-degree')

    # zero_copy (gather) pipeline3 + sample
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='sample', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline3-sample')
    
    # zero_copy (gather) pipeline3 + degree
    cmd = new_command(ds, cache_type='gpu_memory', cache_policy='degree', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=3, MODE='pipeline')
    run(ds, cmd, './log/gpu-cache/pipeline3-degree')


def compare_cache_policy(datasets):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  # datasets = ['livejournal', 'lj-large', 'lj-links']
  # datasets = ['dewiki-2013', 'frwiki-2013']
  # datasets = ['itwiki-2013', 'enwiki-2016']
  

  # cache vary cache_rate (compare difference of cache policy)
  for ds in datasets:
    # for rate in np.linspace(0, 50, 11):
    # for rate in np.linspace(0, 0.4, 21):
    # for rate in np.linspace(0.42, 0.6, 10):
    # for rate in np.linspace(0.62, 0.8, 10):
    # for rate in np.linspace(0.56, 0.60, 3):
    for rate in np.linspace(0.82, 1, 10):
      cmd = new_command(ds, cache_type='rate', cache_policy='degree', cache_rate=f'{rate:.2f}', batch_type='shuffle', fanout='10,25', TIME_SKIP=0, epochs=1, PIPELINES=1, MODE='zerocopy')
      run(ds, cmd, './log/gpu-cache/vary-rate-degree', suffix=f'-{rate:.2f}')
    
      cmd = new_command(ds, cache_type='rate', cache_policy='sample', cache_rate=f'{rate:.2f}', batch_type='shuffle', fanout='10,25', TIME_SKIP=0, epochs=1, PIPELINES=1, MODE='zerocopy')
      run(ds, cmd, './log/gpu-cache/vary-rate-sample', suffix=f'-{rate:.2f}')

      cmd = new_command(ds, cache_type='rate', cache_policy='random', cache_rate=f'{rate:.2f}', batch_type='shuffle', fanout='10,25', TIME_SKIP=0, epochs=1, PIPELINES=1, MODE='zerocopy')
      run(ds, cmd, './log/gpu-cache/vary-rate-random', suffix=f'-{rate:.2f}')


def print_different_optim(mode, datasets):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']

  # explicit
  # for ds in datasets:
  #   cmd = new_command(ds, cache_type='none', cache_policy='none', batch_type='sequence', fanout='10,25', TIME_SKIP=1, epochs=3, PIPELINES=1)
  #   run(ds, cmd, './log/gpu-cache/explicit')
  ret = {}
  for optim in ['explicit', 'pipeline1', 'pipeline3', 'pipeline3-degree']:
    time_list = []
    for ds in datasets:
      log_file = f'./log/gpu-cache/{optim}/{ds}.log'
      time_list += utils.parse_num(log_file, mode)
    ret[optim] = time_list
  return ret


def print_explicit_time(datasets):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  ret = {}
  for optim in ['gcn_sample_time', 'gcn_gather_time', 'gcn_trans_time', 'gcn_train_time']:
    time_list = []
    for ds in datasets:
      log_file = f'./log/gpu-cache/explicit/{ds}.log'
      time_list += utils.parse_num(log_file, optim)
    ret[optim] = time_list
  return ret  


def print_compare_cache_policy(mode, datasets):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  # datasets = ['livejournal', 'lj-large', 'lj-links']
  # datasets = ['dewiki-2013', 'frwiki-2013']
  # datasets = ['itwiki-2013', 'enwiki-2016']

  # cache vary cache_rate (compare difference of cache policy)
  # log_path = './log/gpu-cache/vary-rate-degree'
  log_path = './log/gpu-cache/vary-rate'
  for ds in datasets:
    rate_list, run_time_list = [], []
    # for rate in np.linspace(0, 20, 6):
    for rate in np.linspace(0, 40, 21):
      rate /= 100
      log_file = f'{log_path}-degree/{ds}-{rate}.log'
      ts = utils.parse_num(log_file, mode)
      rate_list.append(rate)
      run_time_list += ts
    print(log_file, rate_list, run_time_list, sep='\n')

    rate_list, run_time_list = [], []
    for rate in np.linspace(0, 20, 6):
      rate /= 100
      log_file = f'{log_path}-sample/{ds}-{rate}.log'
      ts = utils.parse_num(log_file, mode)
      rate_list.append(rate)
      run_time_list += ts
    print(log_file, rate_list, run_time_list, sep='\n')


def draw_diff_optim(datasets):
  # datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links']
  ret = print_different_optim('one_epoch_time', datasets)
  # to numpy array
  for k,v in ret.items():
    ret[k] = np.array(v)

  # rename key
  for x, y in zip(['base', 'zerocopy', 'zerocopy+P', 'zerocopy+PC'], ['explicit', 'pipeline1', 'pipeline3', 'pipeline3-degree']):
    ret[x] = ret.pop(y)

  # normalized
  for k in ['zerocopy', 'zerocopy+P', 'zerocopy+PC']:
    ret[k] = ret['base'] / ret[k]
  ret['base'] = np.ones_like(ret['base'])

  x_name = ['arxiv', 'products', 'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki']
  y_name = 'Normalized Speedup'
  # labels = ['base', 'zerocopy', 'pipeline', 'pipeline+cache']
  labels = list(ret.keys())
  utils.plot_bar(x_name, y_name, list(ret.values()), labels, filename=f'{os.getcwd()}/overleaf-gnn-eval/exp3-gpu-cache/diff-optim.pdf')


def draw_explicit_time(datasets):
# datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links']
  ret = print_explicit_time(datasets)
  # normalized
  diff_stage_time = np.array(list(ret.values()))
  epoch_time = diff_stage_time.sum(axis=0)
  diff_stage_time /= epoch_time


  x_name = ['arxiv', 'products', 'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki']
  y_name = None
  labels = ['sample', 'gather', 'transfer', 'train']
  # labels = list(ret.keys())
  utils.plot_stack_bar(x_name, y_name, diff_stage_time, labels, filename=f'{os.getcwd()}/overleaf-gnn-eval/exp3-gpu-cache/explicit_breakdown.pdf')


def print_diff_cache_ratio(mode, datasets, ratios):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  ret = {}
  for ds in datasets:
    hit_rate_list_degree = []
    hit_rate_list_sample = []
    hit_rate_list_random = []
    for ratio in ratios[ds]:
      log_file = f'./log/gpu-cache/vary-rate-degree/{ds}-{ratio:.2f}.log'
      hit_rate = utils.parse_num(log_file, mode)
      hit_rate_list_degree += hit_rate

      log_file = f'./log/gpu-cache/vary-rate-sample/{ds}-{ratio:.2f}.log'
      hit_rate = utils.parse_num(log_file, mode)
      hit_rate_list_sample += hit_rate

      log_file = f'./log/gpu-cache/vary-rate-random/{ds}-{ratio:.2f}.log'
      hit_rate = utils.parse_num(log_file, mode)
      hit_rate_list_random += hit_rate
    ret[ds+'degree'] = hit_rate_list_degree
    ret[ds+'sample'] = hit_rate_list_sample
    ret[ds+'random'] = hit_rate_list_random
  return ret


def draw_diff_cache_ratio(datasets):
  # datasets = ['itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'reddit', 'lj-links', 'enwiki-links']
  # ret = print_diff_cache_ratio(') cache_hit_rate', datasets, np.linspace(start=0, stop=0.8, num=41))
  radio_dict = {'reddit': np.linspace(0, 1, 51), 'hollywood-2011': np.linspace(0, 1, 51),
                  'lj-links': np.linspace(0, 0.8, 41), 'enwiki-links': np.linspace(0, 0.26, 14)}
  ret = print_diff_cache_ratio(') cache_hit_rate', datasets, radio_dict)
  x_ticks_dict = {'reddit': np.linspace(0, 1, 6), 'hollywood-2011': np.linspace(0, 1, 6),
                  'lj-links': np.linspace(0, 0.6, 6), 'enwiki-links': np.arange(0, 0.26, 0.05)}
  # for k,v in ret.items():
    # print(k, v, sep=' ')
  # x_ticks = np.linspace(0, 0.4, 9)
  # X = np.linspace(start=0, stop=0.8, num=41).reshape(1, -1)
  
  # for ds in ['reddit', 'itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'reddit', 'lj-links', 'enwiki-links']:
  # for ds in ['reddit', 'hollywood-2011', 'reddit', 'lj-links', 'enwiki-links']:
  for ds in datasets:
    X = radio_dict[ds].reshape(1, -1)
    X = X.repeat(3, axis=0)

    Y = []
    for mode in ['random', 'degree', 'sample']:
      Y.append(ret[f'{ds}{mode}'])
    # print(Y)
    print(len(X[0]), len(Y[0]),len(Y[1]),len(Y[2]), min(len(Y[0]),len(Y[1]),len(Y[2])))
    tmp = X[:,:min(len(Y[0]),len(Y[1]),len(Y[2]))]
  
    utils.plot_line(tmp, Y, ['random', 'degree', 'sample'], savefile=f'{os.getcwd()}/overleaf-gnn-eval/exp3-gpu-cache/vary_cache_ratio_{ds}.pdf', x_ticks=x_ticks_dict[ds], x_label='Cache Ratio (%)', y_label='Cache Hit Ratio (%)')


if __name__ == '__main__':
  create_dir('./build')
  os.system('cd build && make -j $(nproc) && cd ..')

  datasets = ['livejournal', 'lj-large', 'lj-links']
  datasets = ['enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  datasets = ['enwiki-links']
  datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products']
  datasets = ['ogbn-arxiv']
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

  # different_optim()
  # draw_diff_optim(datasets)
  # degree_different_optim()

  # draw_explicit_time(datasets)
  # print_compare_cache_policy(') cache_hit_rate', datasets)
  # print_compare_cache_policy(') cache_hit_rate')

  # compare_cache_policy(datasets)
  datasets = ['reddit', 'hollywood-2011', 'lj-links', 'enwiki-links']
  draw_diff_cache_ratio(datasets)
  
  # datasets = ['enwiki-links']
  # datasets = ['itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'reddit', 'lj-links', 'enwiki-links']
  # for ds in datasets:
  #   # for ratio in np.linspace(0, 0.8, 41):
  #   # for ratio in np.linspace(0.42, 0.6, 10):
  #   for ratio in np.linspace(0.62, 0.8, 10):
  #     for mode in ['sample', 'degree', 'random']:
  #       log_file = f'./log/gpu-cache/vary-rate-{mode}/{ds}-{ratio}.log'
  #       if os.path.exists(log_file):
  #         dst_file = f'./log/gpu-cache/vary-rate-{mode}/{ds}-{ratio:.2f}.log'
  #         if log_file != dst_file:
  #           os.rename(log_file, dst_file)
  #           print(log_file, dst_file)
        

      # dst_file = f'./log/gpu-cache/vary-rate-degree/{ds}-{ratio:.2f}.log'
      # # utils.create_file(dst_file)
      # # print(dst_file)
      # with open(dst_file, 'w') as f:
      #   pass

  