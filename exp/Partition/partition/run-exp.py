import sys
import os
import time
import utils
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from PIL import Image
import argparse

import scipy.io
import numpy as np
import matplotlib.ticker as mtick

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



def get_partition_statistic(dataset, num_parts, fanout, batch_size, dim, log_file, algo):
    if not os.path.exists(os.path.dirname(log_file)):
       os.makedirs(os.path.dirname(log_file))

    run_command = f'python main.py --dataset {dataset} --num_parts {num_parts} --fanout {fanout} --batch_size {batch_size} --dim {dim} --algo {algo} > {log_file}'
    print('start running: ', run_command)
    os.system(run_command)


def draw_partition_statistic(dataset, log_file, figpath, suffix):
    assert os.path.exists(log_file), log_file
    if not os.path.exists(figpath):
       os.makedirs(figpath)
    
    run_command = f'python draw.py --dataset {dataset}  --log {log_file} --figpath {figpath} --suffix {suffix}'
    print('start running: ', run_command)
    os.system(run_command)


def plot_bar(plot_params, Y, labels, xlabel, ylabel, xticks, anchor=None, figpath=None):
  
  # plt.rcParams.update(plt.rcParamsDefault)
  # print(plt.rcParams.keys())
  
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # print(plt.style.available)
  plt.style.use('classic')
  # plt.style.use('"bmh"')
  # plt.style.use('"ggplot"')
  # plt.style.use('grayscale')
  # plt.style.use("seaborn-deep")
  # plt.style.use("seaborn-paper")
  # plt.style.use("seaborn-notebook")
  # plt.style.use("seaborn-poster")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  
  width = 0.15
  color_list = ['b', 'g', 'c', 'r', 'm']
  n = len(Y[0])
  ind = np.arange(n)                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + 0.5

  for i, y in enumerate(Y):
    plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i])  
  
  # plt.xticks(np.arange(n) + (len(labels)/2-0.5)*width, xticks)
  plt.xticks(np.arange(n), xticks)
  
  plt.legend(ncol=len(labels), bbox_to_anchor=anchor)

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  # Set the formatter
  axes = plt.gca()   # get current axes
  # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  # ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  axes.grid(axis='y')
  # axes.grid(axis='x')


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()




def plot_stack_multi_bar(plot_params, Y, labels, xlabel, ylabel, xticks, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
  # print(plt.rcParams.keys())
  
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # print(plt.style.available)
  plt.style.use('classic')
  # plt.style.use('"bmh"')
  # plt.style.use('"ggplot"')
  # plt.style.use('grayscale')
  # plt.style.use("seaborn-deep")
  # plt.style.use("seaborn-paper")
  # plt.style.use("seaborn-notebook")
  # plt.style.use("seaborn-poster")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  

  width = 0.25
  color_list = ['b', 'g', 'c', 'r', 'm']

  n = len(Y[0][0])
  ind = np.arange(n)                # the x locations for the groups
  m = len(Y)
  offset = np.arange(m) - m / 2 + 0.5

  for i, sta_y in enumerate(Y):
    pre_bottom = np.zeros(len(sta_y[0]))
    for j, y in enumerate(sta_y):
        # plt.bar(ind+width*i,y,width,color=color_list[i], label =labels[i])  
        # print(ind, y)
        plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], alpha=1-1/len(y)*j, label=labels[i][j], bottom=pre_bottom)
        pre_bottom += y  


    # plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i])  

#   if isinstance(Y, list):
#     n = len(Y[0])
#   else:
#     n = Y.shape[1]
#   ind = np.arange(n)                # the x locations for the groups

#   pre_bottom = np.zeros(len(Y[0]))
#   for i, y in enumerate(Y):
#     # plt.bar(ind+width*i,y,width,color=color_list[i], label =labels[i])  
#     # print(ind, y)
#     plt.bar(ind,y,width,color=color_list[i], label =labels[i], bottom=pre_bottom)
#     pre_bottom += y  

  
  plt.xticks(np.arange(n), xticks)
  # plt.ylim(0, 100)
  # plt.yticks(np.linspace(0, 100, 6), ('0%','20%','40%','60%','80%','100%'))
  # plt.yticks(np.arange(5), ('0%','20%','40%','60%','80%','100%'))

  if anchor:
    plt.legend(ncol=len(labels), bbox_to_anchor=anchor)
  else:
    plt.legend(ncol=len(labels))

  # num1, num2 = 1, 1.2
  # plt.legend(ncol=4, bbox_to_anchor=(num1, num2))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  # Set the formatter
  axes = plt.gca()   # get current axes
  fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  axes.grid(axis='y', linewidth=1.5)
  # axes.grid(axis='x')


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()




def plot_stack_bar(plot_params, Y, labels, xlabel, ylabel, xticks, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
  # print(plt.rcParams.keys())
  
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # print(plt.style.available)
  plt.style.use('classic')
  # plt.style.use('"bmh"')
  # plt.style.use('"ggplot"')
  # plt.style.use('grayscale')
  # plt.style.use("seaborn-deep")
  # plt.style.use("seaborn-paper")
  # plt.style.use("seaborn-notebook")
  # plt.style.use("seaborn-poster")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  

  width = 0.25
  color_list = ['b', 'g', 'c', 'r', 'm']
  n = Y.shape[1]
  ind = np.arange(n)                # the x locations for the groups
  pre_bottom = np.zeros(len(Y[0]))
  for i, y in enumerate(Y):
    # plt.bar(ind+width*i,y,width,color=color_list[i], label =labels[i])  
    # print(ind, y)
    plt.bar(ind,y,width,color=color_list[i], label =labels[i], bottom=pre_bottom)
    pre_bottom += y  

  
  plt.xticks(np.arange(n), xticks)
  # plt.ylim(0, 100)
  # plt.yticks(np.linspace(0, 100, 6), ('0%','20%','40%','60%','80%','100%'))
  # plt.yticks(np.arange(5), ('0%','20%','40%','60%','80%','100%'))

  if anchor:
    plt.legend(ncol=len(labels), bbox_to_anchor=anchor)
  else:
    plt.legend(ncol=len(labels))

  # num1, num2 = 1, 1.2
  # plt.legend(ncol=4, bbox_to_anchor=(num1, num2))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  # Set the formatter
  axes = plt.gca()   # get current axes
  fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  axes.grid(axis='y')
  # axes.grid(axis='x')


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()





def parse_line_num(filename, mode):
  if not os.path.exists(filename):
    print(f'{filename} not exist!')
    assert False
  if not os.path.isfile(filename):
    print(f'{filename} not a file!')
    assert False
  ret = []
  with open(filename) as f:
    for line in f.readlines():
      if line.find(mode) >= 0:
        nums = re.findall(r"\d+\.?\d*", line[line.find(mode):])
        ret.append(nums)
  print(filename, mode, ret)
  assert len(ret) == 1
  # if len(ret) != 1:
  #   print("!!!Warning:", filename, mode, ret)
  # else:
  #   ret = ret[0]
  # if len(ret) == 1:
  ret = [float(x) for x in ret[0]]
  return ret


def get_partition_result(dataset, log_file, node=False, edge=False, train=False, val=False, test=False):
  ret = []
  if node:
    nodes = parse_line_num(log_file, 'partition nodes')
    ret.append(nodes)
  if edge:
    edges = parse_line_num(log_file, 'partition edges')
    ret.append(edges)
  if train:
    train = parse_line_num(log_file, 'train distributed')
    ret.append(train)
  if val:
    val = parse_line_num(log_file, 'val distributed')
    ret.append(val)
  if test:
    test = parse_line_num(log_file, 'test distributed')
    ret.append(test)
  return ret
  


def get_depcomm_result(dataset, mode, log_file):
  local_edges = parse_line_num(log_file, f'{mode}_local_edges')
  cross_edges = parse_line_num(log_file, f'{mode}_cross_edges')
  all_sample_edges = parse_line_num(log_file, f'{mode}_all_sample_edges')
  receive_sample_edges = parse_line_num(log_file, f'{mode}_receive_sample_edges')
  assert np.equal(np.array(local_edges) + np.array(cross_edges), np.array(all_sample_edges)).all()
  return local_edges, cross_edges, all_sample_edges, receive_sample_edges


def get_depcache_result(dataset, mode, log_file):
  local_sample_edges = parse_line_num(log_file, f'{mode}_local_sample_edges')
  remote_sample_edges = parse_line_num(log_file, f'{mode}_remote_sample_edges')
  recv_sample_edges = parse_line_num(log_file, f'{mode}_recv_sample_edges')
  send_edges = parse_line_num(log_file, f'{mode}_send_edges')
  send_features = parse_line_num(log_file, f'{mode}_send_features')
  send_edges_bytes = parse_line_num(log_file, f'{mode}_sen_edges_bytes')
  send_features_bytes = parse_line_num(log_file, f'{mode}_sen_features_bytes')

  return local_sample_edges, recv_sample_edges, send_edges, send_features, send_edges_bytes, send_features_bytes, remote_sample_edges


if __name__ == '__main__':
    params={
      'axes.labelsize': '14',
      'xtick.labelsize':'14',
      'ytick.labelsize':'14',
      'lines.linewidth': 1,
      'legend.fontsize': '14.2',
      'figure.figsize' : '8, 4',
      'legend.loc': 'upper center', #[]"upper right", "upper left"]
      # 'legend.loc': 'best', #[]"upper right", "upper left"]
    }

    datasets = ['reddit', 'ogbn-products']
    datasets = ['computer', 'ogbn-arxiv']
    datasets = ['ogbn-products']
    datasets = ['ogbn-arxiv']
    datasets = ['reddit']
    datasets = ['computer']
    datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit', 'computer']
    
    batch_sizes = {
        # 'AmazonCoBuy_computers': (512, 1024, 2048, 4096, 8250),
        'computer': 600,
        'ogbn-arxiv': 5000,
        'reddit': 8000,
        'ogbn-products': 8000,
    }

    num_parts = 4
    fanout = '10 25'
    # batch_size = 1024

    for ds in datasets:
      for dim in [1, 2, 4]:
        log_file = f'./log/{ds}/{ds}-{dim}.log'
        get_partition_statistic(ds, num_parts, fanout, batch_sizes[ds], dim, log_file, 'metis')
      
      log_file = f'./log/{ds}/{ds}-hash.log'
      get_partition_statistic(ds, num_parts, fanout, batch_sizes[ds], 1, log_file, 'hash')
      
      log_file = f'./log/{ds}/{ds}-pagraph.log'
      get_partition_statistic(ds, num_parts, fanout, batch_sizes[ds], 1, log_file, 'pagraph')
      
      log_file = f'./log/{ds}/{ds}-bytegnn.log'
      get_partition_statistic(ds, num_parts, fanout, batch_sizes[ds], 1, log_file, 'bytegnn')
    exit(0)


    modes = ['1', '4', 'pagraph', 'hash', 'bytegnn']
    labels = ['metis1', 'metis4', 'pagraph', 'hash', 'bytegnn']
    
    # modes = ['1', '4', 'hash', 'bytegnn']
    # labels = ['metis1', 'metis4', 'hash', 'bytegnn']
    # partition graph result
    for ds in datasets:
      node_dist = []
      edge_dist = []
      # modes = ['metis1', 'metis2', 'metis4', 'hash', 'pagraph']
      # modes = ['1', '4', 'pagraph', 'hash', 'bytegnn']
      # modes = ['1', '4', 'hash', 'bytegnn']
      for mode in modes:
        log_file = f'./log/{ds}/{ds}-{mode}.log'

        node, edge, train = get_partition_result(ds, log_file, node=True, edge=True, train=True)
        node_dist.append(node)
        edge_dist.append(edge)

      print(modes, 'node number', [sum(x) for x in node_dist])
      print(modes, 'edge number', [sum(x) for x in edge_dist])
      xticks = [f'part {x}' for x in range(num_parts)]
      xlabel = ''
      ylabel = ''
      # ./log/{ds}/{ds}-node.pdf
      # plot_bar(params, node_dist, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'./overleaf/{ds}-node.pdf')
      # plot_bar(params, edge_dist, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'./overleaf/{ds}-edge.pdf')
    
  
    # dep cache statistics
    for ds in datasets:
      # if ds == 'reddit': # reddit pagraph msissed
      #    continue
      
      compute_Y = []
      comm_Y = []
      # modes = ['metis1', 'metis2', 'metis4', 'hash', 'pagraph']
      for mode in modes:
        log_file = f'./log/{ds}/{ds}-{mode}.log'
        # draw_partition_statistic(ds, log_file, f'./log/{ds}', f'dim{dim}')
        local_sample_edges, recv_sample_edges, send_edges, send_features, send_edges_bytes, send_features_bytes, remote_sample_edges = get_depcache_result(ds, 'sum', log_file)
        compute_Y.append(np.array(local_sample_edges) + np.array(recv_sample_edges))
        comm_Y.append((np.array(send_edges_bytes) + np.array(send_features_bytes)) / 1024/ 1024)
    
      print(modes, 'compute load', [sum(x) for x in compute_Y])
      print(modes, 'comm load', [sum(x) for x in comm_Y])

      xlabel = 'communication load'
      ylabel = 'MB'
      xticks = [f'part {x}' for x in range(num_parts)]
      plot_bar(params, comm_Y, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'./overleaf/{ds}-depcache-comm.pdf')

      ylabel = ''
      xlabel = 'compute load'
      plot_bar(params, compute_Y, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'./overleaf/{ds}-depcache-compute.pdf')



    
    # dep cache statistics (stack bar)
    for ds in datasets:
      local_edges = []
      remote_edges = []
      # modes = ['metis1', 'metis2', 'metis4', 'hash', 'pagraph']
      # modes = ['1', '4', 'pagraph', 'hash', 'bytegnn']
      # modes = ['1', '4', 'hash', 'bytegnn']
      for mode in modes:

        log_file = f'./log/{ds}/{ds}-{mode}.log'
        # draw_partition_statistic(ds, log_file, f'./log/{ds}', f'dim{dim}')
        local_sample_edges, recv_sample_edges, send_edges, send_features, send_edges_bytes, send_features_bytes, remote_sample_edges  = get_depcache_result(ds, 'sum', log_file)
        local_edges.append(np.array(local_sample_edges))
        remote_edges.append(np.array(remote_sample_edges))
    
      print(modes, 'local edges', [sum(x) for x in local_edges])
      print(modes, 'remote edges', [sum(x) for x in remote_edges])

      xlabel = '#partition ID'
      partiton_algo = ['metis1', 'metis4', 'pagraph', 'hash', 'bytegnn']
      # partiton_algo = ['metis1', 'metis4', 'hash', 'bytegnn']
      labels = ['Local', 'Remote']
      for i, algo in enumerate(partiton_algo):
        Y = np.array([local_edges[i], remote_edges[i]])
        Y /= 1e6
        print(algo, 'all data request', Y.sum())
        ylabel = '#Reqeust Number 1e6'
        plot_stack_bar(params, Y, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'./overleaf/{ds}-{algo}-datarequire.pdf')

