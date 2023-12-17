import sys
import os
import time
import numpy as np
import re
# import utils
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.pylab as pylab

# datasets = ['ppi', 'ppi-large', 'reddit', 'flickr', 'yelp', 'amazon']
# batch_size = {'ppi':4096, 'ppi-large':4096, 'flickr':40960, 'yelp':40960, 'amazon':40960, 'reddit':40960}


def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)

# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, X, Y, labels, xlabel, ylabel, xticks, yticks, xlim, ylim, ds, markevery, anchor=None, figpath=None):

  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # plt.style.use("grayscale")
  # plt.style.use("classic")
  # plt.style.use("seaborn-paper")
  # plt.style.use("bmh")
  # plt.style.use("ggplot")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  makrer_list = ['D', 's', 'v', 'p', '*', 'd', 'X', 'D',    'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [5,5,5,5,5,5,10,10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
  # color_list = ['b', 'g', 'k', 'c', 'm', 'y', 'r'] 
  # color_list = ['#2a6ca6', '#f47a2d', '#419136', '#f47a2d', '#c4342b', '#7c4e44', '#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44'] 
  color_list = ['C0', 'C1'] 

  
  axes1 = plt.subplot(111)#figure1的子图1为axes1
  for i, (x, y) in enumerate(zip(X, Y)):
    # plt.plot(x, y, label = labels[i], color=color_list[i], marker=makrer_list[i], markersize=8,markevery=markevery[ds])
    # plt.plot(x, y, label = labels[i], color=color_list[i])
    plt.plot(x, y, label = labels[i], color=color_list[i], linewidth=myparams['lines.linewidth'])

    # plt.plot(x, y, label = labels[i], markersize=5)
  axes1.set_yticks(yticks, pad=0)
  axes1.set_xticks(xticks)
  axes1.tick_params(axis='both', which='major', pad=2)
  # axes1.set_xticks([0,125,250,375,500])  
  ############################
  # axes1.set_ylim(0.92, 0.94)
  # axes1.set_xlim(0, 500)
  axes1.set_ylim(ylim)
  axes1.set_xlim(xlim)
  plt.legend(ncol=2, bbox_to_anchor=anchor, columnspacing=1.5, handletextpad=.25 , handleheight=1, handlelength=.7)#

  # plt.legend(ncol=2, frameon=False)
  ############################

  # axes1 = plt.gca()
  # axes1.grid(True)  # add grid

  plt.ylabel(ylabel, labelpad=0) 
  plt.xlabel(xlabel) 


  # Set the formatter
  # axes = plt.gca()   # get current axes
  # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  # ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes.xaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes1.grid(axis='y', linestyle='-', )
  # axes1.grid(axis='y', linestyle='', )
  # axes1.spines['bottom'].set_linewidth(myparams['lines.linewidth']);###设置底部坐标轴的粗细
  # axes1.spines['left'].set_linewidth(myparams['lines.linewidth']);####设置左边坐标轴的粗细
  # axes1.spines['right'].set_linewidth(myparams['lines.linewidth']);###设置右边坐标轴的粗细
  # axes1.spines['top'].set_linewidth(myparams['lines.linewidth']);####设置上部坐标轴的粗细
  
  figpath = './line.pdf' if not figpath else figpath
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
                nums = re.findall(r"\d+\.?\d*", line[line.find(mode) :])
                ret.append(float(nums[0]))
    return ret




def print_diff_cache_ratio(datasets, log_path):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  ret = {}
  for ds in datasets:
    for block in ['256', '512', '1024', '2048', '4096']:
      log_file = f'{log_path}/block{block}/{ds}.log'
      print(log_file)
      if not os.path.exists(log_file):
         continue
      threshold_rate = parse_num(log_file, 'thrashold_rate')
      suit_explicit_rate = parse_num(log_file, 'suit explicit trans block rate')
      ret[ds+block+'rate'] = threshold_rate
      ret[ds+block+'explicit'] = suit_explicit_rate


      log_file = f'{log_path}/block{block}-degree-cache0.5/{ds}.log'
      print(log_file)
      if not os.path.exists(log_file):
         continue
      threshold_rate = parse_num(log_file, 'thrashold_rate')
      suit_explicit_rate = parse_num(log_file, 'suit explicit trans block rate')
      ret[ds+block+'rate_cache'] = threshold_rate
      ret[ds+block+'explicit_cache'] = suit_explicit_rate
  return ret



if __name__ == '__main__':

  myparams={
    'axes.labelsize': '10',
    'xtick.labelsize':'10',
    'ytick.labelsize':'10',
    'lines.linewidth': 1.5,
    # 'legend.fontsize': '14.7',
    'legend.fontsize': '10',
    'figure.figsize' : '2, 1.7',
    'legend.loc': 'upper center', #[]"upper right", "upper left"]
    # 'legend.loc': 'best', #[]"upper right", "upper left"]
    'legend.frameon': False,
    # 'font.family': 'Arial'
    'font.family': 'Arial',
    'font.serif': 'Arial',
  }

  datasets = ['hollywood-2011', 'lj-links', 'reddit', 'lj-links', 'enwiki-links','ogbn-arxiv', 'livejournal', 'ogbn-products']
  datasets = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links', 'ogbn-arxiv', 'ogbn-products']
  datasets = ['reddit', 'lj-links', 'enwiki-links', 'ogbn-arxiv', 'ogbn-products', 'hollywood-2011']
  
  # datasets = ['reddit', 'hollywood-2011', 'lj-links', 'enwiki-links']
  datasets = ['road-usa']
  datasets = ['amazon']
  datasets = ['ogbn-products', 'reddit', 'ogbn-arxiv']
  datasets = ['lj-links', 'ogbn-arxiv']
  datasets = ['lj-large']
  datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit', 'livejournal', 'lj-large', 'hollywood-2011', 'lj-links', 'enwiki-links']
  datasets = ['ogbn-arxiv']
  datasets = ['ogbn-arxiv','ogbn-products', 'reddit', 'livejournal', 'lj-large', 'hollywood-2011', 'lj-links', 'enwiki-links']

  x_lims = {
      'reddit': (0, 100),
      'ogbn-arxiv': (0, 100),
      'ogbn-products': (0, 50),
      'livejournal': (0, 50),
      'lj-large': (0, 50),
      'hollywood-2011': (0, 50),
      'enwiki-links': (0, 50),
      'lj-links': (0, 50),
  }
  
  mark_list = {
          'ogbn-arxiv': 4,
          'ogbn-products': 2,
          'reddit': 4,
          'livejournal': 2,
          'lj-large': 2,
          'hollywood-2011': 2,
      }



  ret = print_diff_cache_ratio(datasets, './log')
  # ret = print_diff_cache_ratio(datasets, '../log/gpu-cache-dgl2')


  # labels = ['256', '512', '1024']
  blcoks_list = ['256', '512', '1024', '2048', '4096']
  blcoks_list = ['256', '1024',  '4096']
  blcoks_list = ['256', '1024',]
  blcoks_list = ['512']
  blcoks_list = ['2048']
  blcoks_list = ['256']
  labels = []
  for ds in datasets:
    for block_size in blcoks_list:
      X, Y = [], []
      threshold_rate = ret[ds+block_size+'rate']
      suit_explicit_rate = ret[ds+block_size+'explicit']
      threshold_rate_cache = ret[ds+block_size+'rate_cache']
      suit_explicit_rate_cache = ret[ds+block_size+'explicit_cache']

      X.append(threshold_rate)
      Y.append(suit_explicit_rate)
      labels.append('w/o cache')
      X.append(threshold_rate_cache)
      Y.append(suit_explicit_rate_cache)
      labels.append('w/ cache')
       

      Y = np.array(Y) * 100
      X = np.array(X) * 100

      x_ticks = np.linspace(0, 100, 6)
      x_ticks = np.linspace(*x_lims[ds], 6)
      y_ticks = np.linspace(0, 100, 6)
      y_lim = (0, 100)

      create_dir('./pdf')
      pdf_file = f'./pdf/{ds}-{block_size}.pdf'

      xlabel = 'Threshold Ratio (%)'
      ylabel = 'Suit Explicit Block Ratio (%)'
      # print(X)
      # print(Y)
      plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks, y_ticks, (x_ticks[0], x_ticks[-1]), y_lim, ds, mark_list, (0.5, 1.25), pdf_file)

  

  