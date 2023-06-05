import sys
import os
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from PIL import Image

import scipy.io
import numpy as np
import matplotlib.ticker as mtick


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


def print_different_optim(mode, datasets):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  ret = {}
  for optim in ['explicit', 'zerocopy', 'pipeline1', 'pipeline3', 'pipeline3-degree', 'pipeline3-sample']:
    time_list = []
    for ds in datasets:
      log_file = f'../log/gpu-cache/{optim}/{ds}.log'
      time_list += parse_num(log_file, mode)
    ret[optim] = time_list
  return ret


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



if __name__ == '__main__':
  datasets = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links']
  ret = print_different_optim('one_epoch_time', datasets)
  # to numpy array
  for k,v in ret.items():
    print(k, v)
    ret[k] = np.array(v)

  # rename key
  # for x, y in zip(['base', 'zerocopy', 'zerocopy+P', 'zerocopy+PC'], ['explicit', 'pipeline1', 'pipeline3', 'pipeline3-degree']):
  tmp_ret = {}
  # for x, y in zip(['base', 'zerocopy', 'zerocopy+P', 'zerocopy+PC'], ['explicit', 'zerocopy', 'pipeline3', 'pipeline3-degree']):
  for x, y in zip(['base', 'zero', 'zero+P', 'zero+PC'], ['explicit', 'zerocopy', 'pipeline3', 'pipeline3-sample']):
    tmp_ret[x] = ret[y]
  ret = tmp_ret

  print('\nafter rename:')
  for k,v in ret.items():
    print(k, v)

  # normalized
  for k in ['zero', 'zero+P', 'zero+PC']:
    ret[k] = ret['base'] / ret[k]
  ret['base'] = np.ones_like(ret['base'])

  print('\nafter normalized:')
  for k,v in ret.items():
    print(k, v, f'avg: {np.average(v):.3f}', )

  params={
    'axes.labelsize': '14',
    'xtick.labelsize':'14',
    'ytick.labelsize':'14',
    'lines.linewidth': 2,
    'legend.fontsize': '14.7',
    'figure.figsize' : '8, 4',
    'legend.loc': 'upper center', #[]"upper right", "upper left"]
  }

  xticks = datasets
  ylabel = 'Normalized Speedup'
  xlabel = ''
  # labels = ['base', 'zerocopy', 'pipeline', 'pipeline+cache']
  labels = list(ret.keys())
  Y = list(ret.values())
  plot_bar(params, Y, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.15), figpath='diff-optim.pdf')
