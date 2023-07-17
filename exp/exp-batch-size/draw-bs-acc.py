import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import numpy as np
import matplotlib.ticker as mtick
import os
import re

def create_dir(path=None):
    if path and not os.path.exists(path):
        os.makedirs(path)

# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, X, Y, labels, xlabel, ylabel, xticks, yticks, xlim, ylim, figpath=None):

  pylab.rcParams.update(plot_params)  #更新自己的设置
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # plt.style.use("seaborn-deep")
  # ["seaborn-deep", "grayscale", "bmh", "ggplot"]

  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  makrer_list = [ 'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
  axes1 = plt.subplot(111)#figure1的子图1为axes1
  for i, (x, y) in enumerate(zip(X, Y)):
    # plt.plot(x, y, label = labels[i], marker=makrer_list[i], markersize=3,markevery=marker_every[i])
    plt.plot(x, y, label = labels[i], markersize=5)
  axes1.set_yticks(yticks)
  axes1.set_xticks(xticks)
  ############################
  axes1.set_ylim(ylim[0], ylim[1])
  axes1.set_xlim(xlim[0], xlim[1])
  plt.legend(ncol=2)
  ############################

  # axes1 = plt.gca()
  # axes1.grid(True)  # add grid

  plt.ylabel(ylabel)
  plt.xlabel(xlabel)

  figpath = './log/batch-size/reddit-exp5/plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()


# 每隔time_skip对acc取一个平均值
def split_list(X, Y, time_skip):
    retX, retY = [], []
    for arrx,arry in zip(X, Y):
        tmpx, tmpy = [], []
        pre, idx = 0, 0
        for i in range(len(arrx)):
            x, y = arrx[i], arry[i]
            if x >= idx * time_skip:
                tmpx.append(x)
                tmpy.append(np.average(arry[pre : i + 1]))
                pre = i + 1
                idx += 1
        if pre < len(arrx):
            tmpx.append(arrx[-1])
            tmpy.append(np.average(arry[pre : ]))

        retX.append(tmpx)
        retY.append(tmpy)
    return retX, retY


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



if __name__ == '__main__':
  myparams = {
    'axes.labelsize': '10',
    'xtick.labelsize': '10',
    'ytick.labelsize': '10',
    # 'font.family': 'Times New Roman',
    'figure.figsize': '4, 3',  #图片尺寸
    'lines.linewidth': 1,
    'legend.fontsize': '8',
    'legend.loc': 'best', #[]"upper right", "upper left"]
    'legend.numpoints': 1,
    # 'lines.ncol': 2,
  }
  # def print_val_acc(mode, datasets, batch_sizes, suffix=None):
  batch_sizes = {
        # 'reddit': (512, 1024,  8192, 16384, 32768,  153431),
        'reddit': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
        # 'ogbn-arxiv': (128,  1024, 2048, 4096,  16384,  90941,'mix6'),dddd
        # 'ogbn-arxiv': (128, 512, 3072, 6144, 12288, 24576, 49152, 90941),
        'ogbn-arxiv': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 90941),
        # 'ogbn-products': (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,'mix'),
        'ogbn-products': (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
    }
  
  acc_ylim = {
        'reddit': [0, .945],
        'ogbn-arxiv': [0, .72],
        'ogbn-products': [0, .92],

        'reddit': [.925, .945],
        'ogbn-arxiv': [.66, .72],
        'ogbn-products': [.76, .92],
    }
  
  # datasets = ['ogbn-arxiv']
  datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit']

  log_path = '../log'
  for ds in datasets:
    val_acc = []
    run_time = []
    for bs in batch_sizes[ds]:
        # log_file = f'{log_path}/batch-size-nts-dgl-sgd/{ds}-0.001/{ds}-{bs}.log'
        log_file = f'{log_path}/batch-size-nts-old-adam/{ds}-0.001/{ds}-{bs}.log'
        # print(ds, log_file)
        val_acc.append(parse_num(log_file, 'val_acc'))
        run_time.append(parse_num(log_file, 'gcn_run_time'))
    yticks = np.linspace(*acc_ylim[ds], 5)
    xticks = np.linspace(0, 300, 5)
    labels = batch_sizes[ds]
    xlabel = 'run time'
    ylabel = 'Accuracy'
    run_time_skip,val_acc_skip = split_list(run_time,val_acc,10)
  #   run_time_skip,val_acc_skip = run_time,val_acc
    create_dir('./nts-old-sample')
    plot_line(myparams, run_time_skip, val_acc_skip, labels, xlabel, ylabel, xticks, yticks, (0,300), acc_ylim[ds], f'./nts-old-sample/{ds}-acc.pdf')