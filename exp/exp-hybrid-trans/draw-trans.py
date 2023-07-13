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


def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)

# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, X, Y, labels, xlabel, ylabel, xticks, yticks, xlim, ylim, ds, markevery, figpath=None):

  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # plt.style.use("seaborn-deep")
  # plt.style.use("grayscale")
  plt.style.use("classic")
  # plt.style.use("bmh")
  # plt.style.use("ggplot")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  makrer_list = ['D', 's', 'v', 'p', '*', 'd', 'X', 'D',    'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [5,5,5,5,5,5,10,10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
  color_list = ['b', 'g', 'k', 'c', 'm', 'y', 'r'] 
  color_list = ['#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44', '#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44'] 

  
  axes1 = plt.subplot(111)#figure1的子图1为axes1
  for i, (x, y) in enumerate(zip(X, Y)):
    
    plt.plot(x, y, label = labels[i], color=color_list[i], marker=makrer_list[i], markersize=8,markevery=markevery[ds])
    # plt.plot(x, y, label = labels[i], color=color_list[i])
    # plt.plot(x, y, label = labels[i], markersize=5)
  axes1.set_yticks(yticks)
  axes1.set_xticks(xticks)
  # axes1.set_xticks([0,125,250,375,500])  
  ############################
  # axes1.set_ylim(0.92, 0.94)
  # axes1.set_xlim(0, 500)
  axes1.set_ylim(ylim)
  axes1.set_xlim(xlim)
  plt.legend(ncol=1, frameon=False)
  ############################

  # axes1 = plt.gca()
  # axes1.grid(True)  # add grid

  plt.ylabel(ylabel) 
  plt.xlabel(xlabel) 

  # Set the formatter
  # axes = plt.gca()   # get current axes
  # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  # ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes.xaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes1.grid(axis='y', linestyle='-', )
  axes1.grid(axis='y', linestyle='', )
  
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

  myparams = {
        'axes.labelsize': '18',
        'xtick.labelsize': '18',
        'ytick.labelsize': '18',
        # 'font.family': 'Times New Roman',
        'figure.figsize': '5, 4',  #图片尺寸
        'lines.linewidth': 4,
        'legend.fontsize': '13',
        'legend.loc': 'best', #[]"upper right", "upper left"]
        'legend.numpoints': 1,
        # 'lines.ncol': 2,
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
  datasets = ['ogbn-arxiv','ogbn-products', 'reddit', 'livejournal', 'lj-large', 'hollywood-2011']
  datasets = ['ogbn-arxiv']

  x_lims = {
      'reddit': (0, 100),
      'ogbn-arxiv': (0, 100),
      'ogbn-products': (0, 50),
      'livejournal': (0, 50),
      'lj-large': (0, 50),
      'hollywood-2011': (0, 50),
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
    X, Y = [], []
    for block_size in blcoks_list:
      threshold_rate = ret[ds+block_size+'rate']
      suit_explicit_rate = ret[ds+block_size+'explicit']
      threshold_rate_cache = ret[ds+block_size+'rate_cache']
      suit_explicit_rate_cache = ret[ds+block_size+'explicit_cache']
      # print(ds, block_size)
      # print(threshold_rate)
      # print(suit_explicit_rate)
      # print(threshold_rate_cache)
      # print(suit_explicit_rate_cache)
      # print()
      X.append(threshold_rate)
      Y.append(suit_explicit_rate)
      labels.append(block_size)
      X.append(threshold_rate_cache)
      Y.append(suit_explicit_rate_cache)
      labels.append(block_size + ' w/cache')
       
    # print(Y)
    # print(len(X[0]), len(Y[0]),len(Y[1]),len(Y[2]), min(len(Y[0]),len(Y[1]),len(Y[2])))
    # X = X[:,:min(len(Y[0]),len(Y[1]),len(Y[2]))]

    Y = np.array(Y) * 100
    X = np.array(X) * 100

    x_ticks = np.linspace(0, 100, 6)
    x_ticks = np.linspace(*x_lims[ds], 6)
    y_ticks = np.linspace(0, 100, 6)
    y_lim = (0, 100)

    pdf_file = f'./trans_pdf/{ds}.pdf'

    xlabel = 'Threshold Ratio (%)'
    ylabel = 'Suit Explicit Block Ratio (%)'
    # print(X)
    # print(Y)
    plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks, y_ticks, (x_ticks[0], x_ticks[-1]), y_lim, ds, mark_list, pdf_file)

    # plot_line(tmp, Y, ['random', 'degree', 'sample'], savefile=, x_ticks=x_ticks, x_name=x_name, y_ticks=y_ticks, y_name=y_name, x_label=, y_label='')

  

  