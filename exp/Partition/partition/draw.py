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
  
  
  width = 0.2
  color_list = ['b', 'g', 'c', 'r', 'm']
  n = len(Y[0])
  ind = np.arange(n)                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + 0.5

  for i, y in enumerate(Y):
    plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i])  
  
  # plt.xticks(np.arange(n) + (len(labels)/2-0.5)*width, xticks)
  plt.xticks(np.arange(n), xticks)
  
  # plt.legend(ncol=len(labels))
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
  if isinstance(Y, list):
    n = len(Y[0])
  else:
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
  axes.grid(axis='y', linewidth=1.5)
  # axes.grid(axis='x')


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()





def parse_num(filename, mode):
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
        ret.append(float(nums[0]))
  return ret


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
  assert len(ret) == 1
  # print(filename, mode, ret)
  # if len(ret) != 1:
  #   print("!!!Warning:", filename, mode, ret)
  # else:
  #   ret = ret[0]
  ret = [float(x) for x in ret[0]]
  return ret


def get_partition_result(dataset, log_file):
  nodes = parse_line_num(log_file, 'metis partition nodes')
  edges = parse_line_num(log_file, 'metis partition edges')
  train = parse_line_num(log_file, 'train distributed')
  val = parse_line_num(log_file, 'val distributed')
  test = parse_line_num(log_file, 'test distributed')
  return nodes, edges, train, val, test
  


def get_depcomm_result(dataset, mode, log_file):
  local_edges = parse_line_num(log_file, f'{mode}_local_edges')
  cross_edges = parse_line_num(log_file, f'{mode}_cross_edges')
  all_sample_edges = parse_line_num(log_file, f'{mode}_all_sample_edges')
  receive_sample_edges = parse_line_num(log_file, f'{mode}_receive_sample_edges')
  assert np.equal(np.array(local_edges) + np.array(cross_edges), np.array(all_sample_edges)).all()
  return local_edges, cross_edges, all_sample_edges, receive_sample_edges


def get_depcache_result(dataset, mode, log_file):
  local_sample_edges = parse_line_num(log_file, f'{mode}_local_sample_edges')
  recv_sample_edges = parse_line_num(log_file, f'{mode}_recv_sample_edges')
  send_edges = parse_line_num(log_file, f'{mode}_send_edges')
  send_features = parse_line_num(log_file, f'{mode}_send_features')
  send_edges_bytes = parse_line_num(log_file, f'{mode}_sen_edges_bytes')
  send_features_bytes = parse_line_num(log_file, f'{mode}_sen_features_bytes')

  return local_sample_edges, recv_sample_edges, send_edges, send_features, send_edges_bytes, send_features_bytes 


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Draw for graph partition statistic')
  parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (cora, citeseer, pubmed, reddit)")
  
  parser.add_argument("--log", type=str, required=True, help="your log file path.")
  parser.add_argument("--suffix", type=str, default='', help="save figure suffix.")
  parser.add_argument("--figpath", type=str, default='.', help="save figure path.")
  
  args = parser.parse_args()
  print(args)
  assert os.path.exists(args.log)
  if args.suffix != '':
    args.suffix = '-' + args.suffix
  
  if not os.path.exists(args.figpath):
    print(f'{args.figpath} not exist, create it now...')
    os.makedirs(args.figpath)
    print(f'{args.figpath} created done...')
  
  params={
    'axes.labelsize': '14',
    'xtick.labelsize':'14',
    'ytick.labelsize':'14',
    'lines.linewidth': 1,
    'legend.fontsize': '14.7',
    'figure.figsize' : '8, 4',
    'legend.loc': 'upper center', #[]"upper right", "upper left"]
    # 'legend.loc': 'best', #[]"upper right", "upper left"]
  }

  
  node, edge, train, val, test = get_partition_result(args.dataset, args.log)


  xticks = [f'part {x}' for x in range(len(node))]
  ylabel = ''
  xlabel = ''
  
  # node info
  labels = ['node', 'train', 'val', 'test']
  plot_bar(params, [node, train, val, test], labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'{args.figpath}/{args.dataset}-node{args.suffix}.pdf')


  # edge info
  labels = ['edge']
  plot_bar(params, [edge], labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'{args.figpath}/{args.dataset}-edge{args.suffix}.pdf')


  labels = ['local', 'cross']
  # labels = list(ret.keys())
  local_edges, cross_edges, all_sample_edges, receive_sample_edges = get_depcomm_result(args.dataset, 'sum', args.log)
  plot_stack_bar(params, [local_edges, cross_edges], labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'{args.figpath}/{args.dataset}-depcomm{args.suffix}.pdf')

  
  labels = ['local sample edges', 'recv sample edges']
  local_sample_edges, recv_sample_edges, send_edges, send_features, send_edges_bytes, send_features_bytes  = get_depcache_result(args.dataset, 'sum', args.log)
  plot_stack_bar(params, [local_sample_edges, recv_sample_edges], labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'{args.figpath}/{args.dataset}-depcache-compute{args.suffix}.pdf')
  
  labels = ['comm edges', 'comm features']
  ylabel = 'bytes'
  # print('sum_send_edges_bytes', np.sum(epoch_send_edges_bytes, axis=0).tolist())
  #   print('sum_send_features_bytes', np.sum(epoch_send_features_bytes, axis=0).tolist())                    
  plot_stack_bar(params, [send_edges_bytes, send_features_bytes], labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath=f'{args.figpath}/{args.dataset}-depcache-comm{args.suffix}.pdf')
