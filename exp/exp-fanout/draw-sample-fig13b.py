import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from numpy import linalg as la
import re

sys.path.append(os.path.join(os.getcwd(), '../..'))
# import utils

def create_dir(path=None):
    if path and not os.path.exists(path):
        os.makedirs(path)


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


# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, X, Y, labels, xlabel, ylabel, xticks, yticks, xlim, ylim, figpath=None):

  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # plt.style.use("seaborn-deep")
#   plt.style.use("grayscale")
#   plt.style.use("classic")
  # plt.style.use("bmh")
  # plt.style.use("ggplot")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  makrer_list = ['o', 'v', 'v', 'p', '*', 'd', 'X', 'D',    'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [10,10,10,10,10,10,10,10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
#   color_list = ['b', 'g', 'k', 'c', 'm', 'y', 'r'] 
#   color_list = ['#fdd37a','#f0633a','#99093d','#e6f397','#73c49a','#415aa4',]
#   color_list = ['#4676b5','#b84746',]
  color_list = ['C1','C0']
  

  ax1 = plt.subplot(111)#figure1的子图1为ax1
  print(X, Y[0])
  ax1.plot(X, Y[0], label = labels[0], color=color_list[0], marker=makrer_list[0], markersize=3)
  ax1.set_yticks(yticks[0])
  ax1.set_xticks(X, xticks, rotation=50)
#   plt.xticks(np.arange(n), xticks, rotation=25)

  ax1.set_ylim(ylim[0])
  ax1.set_xlim(xlim)
  ax1.set_xlabel(xlabel) 
  ax1.set_ylabel(ylabel[0], labelpad=2) 


  ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

  ax2.plot(X, Y[1], label = labels[1], color=color_list[1], marker=makrer_list[1], markersize=3)
  ax2.set_yticks(yticks[1])
  ax2.set_ylabel(ylabel[1], rotation=90, labelpad=2) 
  ax2.set_ylim(ylim[1])

  lines, labels = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  plt.legend(lines + lines2, labels + labels2, ncol=2, bbox_to_anchor=(0.5, 1.2), handlelength=1, columnspacing=0.5, handletextpad=.2, labelspacing=.2)


  figpath = './line.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', pad_inches=0, format='pdf')#bbox_inches='tight'会裁掉多余的白边
#   plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()


def print_val_acc(mode, datasets, batch_sizes, suffix=None):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for ds in datasets:
        for bs in batch_sizes[ds]:
            val_acc_list = []
            if suffix:
                log_file = f'./log/{ds}/{ds}-{suffix}-{bs}.log'
            else:
                log_file = f'./log/{ds}/{ds}-{bs}.log'
            val_acc = parse_num(log_file, mode)
            print(log_file)
            val_acc_list += val_acc
            ret[ds + str(bs)] = val_acc_list
    # print(ret)
    return ret


def draw_all(myparams, datasets, sample_rates, pdf_dir, suffix=None):
    acc_ylim = {
        'reddit': [.91, .95],
        # 'reddit': [.1, .95],
        'ogbn-arxiv': [.45, .72],
        'ogbn-arxiv': [.68, .722],
        # 'ogbn-arxiv': [.1, .722],
        'ogbn-products': [.76, .92],
        # 'ogbn-products': [.1, .92],
        'computer': [.50, 1],
        'photo': [.50, 1],

        'reddit': [0, .945],
        'ogbn-arxiv': [0, .72],
        'ogbn-products': [0, .92],

        'reddit': [92.5, 94.5],
        'ogbn-arxiv': [70.5, 72.5],
        'ogbn-products': [76, 92],
    }

    speed_ylim = {
        'reddit': [0, 300],
        'ogbn-products': [0, 600],
        'ogbn-arxiv': [0, 200],
    }
    
    target_acc = {
        'ogbn-arxiv': 0.684,
        'reddit': 0.93,
        'ogbn-products': 0.89,
    }


    if suffix:
        val_acc_dict = print_val_acc('val_acc', datasets, sample_rates, str(suffix))
        train_time_dict = print_val_acc('train_time', datasets, sample_rates, str(suffix))
    else:
        val_acc_dict = print_val_acc('val_acc', datasets, sample_rates)
        train_time_dict = print_val_acc('train_time', datasets, sample_rates)

    for ds in datasets:
        Best_ACC_Y = []
        for bs in sample_rates[ds]:
            Best_ACC_Y.append(np.max(val_acc_dict[ds+str(bs)]))
        Time_X, Y = [], []
        all_avg_train_time = []
        for i, bs in enumerate(sample_rates[ds]):
            Y.append(val_acc_dict[ds + str(bs)])
            avg_train_time = np.average(train_time_dict[ds + str(bs)][3:])
            all_avg_train_time.append(round(avg_train_time, 3))
            Time_X.append(np.cumsum([avg_train_time] * len(Y[-1])))
            # print('batch_size', bs, 'avg_epoch_time', round(avg_train_time, 2))

        target_acc[ds] = np.max(Best_ACC_Y)* 0.98
        target_acc[ds] = np.min(Best_ACC_Y) 
        X_t, Y_t = [], []
        for x, y in zip(Time_X, Y):
            if np.max(y) < target_acc[ds]:
                Y_t.append(y)
                X_t.append(x)
            else:
                y_t = np.array(y)
                idx = np.where(y_t >= target_acc[ds])[0][0] + 1
                Y_t.append(y[:idx])
                X_t.append(x[:idx])
                # print('time', X_t[-1][-1], 'acc', Y_t[-1][-1])
        assert len(Y) == len(Y_t)
        Y = [round((x[-1]),2) for x in X_t]
        # print('time_to_acc time:', Y)
        
        Y = [np.array(Best_ACC_Y) * 100, np.array(Y)]
        # X = [np.log2(batch_sizes[ds]), np.log2(batch_sizes[ds])]
        # X = np.log2(sample_rates[ds])

        # def print_xlx(info, X):
        #     print(info, ' '.join([str(round(x, 3)) for x in X]))

        # print()
        # print(ds)
        # print_xlx('batch_size', X)
        # print_xlx('best_acc', Y[0])
        # print_xlx('speed', Y[1])
        # print_xlx('epoch_time', all_avg_train_time)
        # print('min', np.min(Best_ACC_Y), 'max', np.max(Best_ACC_Y), 'target', target_acc[ds])

        labels = ['Accuracy (%)', 'Convergence Time (s)']
        xlabel = 'Sample Rate'
        ylabel = ['Accuracy (%)', 'Convergence Time (s)']
        
        create_dir(pdf_dir)
        if suffix:
            pdf_file = f'{pdf_dir}/{ds}-sample-{suffix}.pdf'
        else:
            pdf_file = f'{pdf_dir}/{ds}-sample.pdf'

        X = np.arange(1, len(Y[0]) + 1)
        x_lim = (0.5, len(Y[0]) + .5)  
        x_ticks = sample_rates[ds]

        print(x_lim, x_ticks)
        if ds == 'reddit':
            y_ticks = [np.arange(92, 95, 1), np.linspace(speed_ylim[ds][0], speed_ylim[ds][1], 5)]
            y_lim = [[92,94.5], speed_ylim[ds]]
            print(y_ticks)
        elif ds == 'ogbn-arxiv':
            y_lim = [acc_ylim[ds], speed_ylim[ds]]
            y_lim = [[68,71.5], [0, 300]]
            y_ticks = [np.linspace(68, 71, 4), np.linspace(y_lim[1][0], y_lim[1][1], 5)]
        else:
            y_ticks = [np.linspace(acc_ylim[ds][0], acc_ylim[ds][1], 5), np.linspace(speed_ylim[ds][0], speed_ylim[ds][1], 5)]
            # y_lim = [acc_ylim[ds], speed_ylim[ds]]
            y_lim = [[75,92], [0, 500]]
        plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks, y_ticks, x_lim, y_lim, pdf_file)


if __name__ == '__main__':
    sample_rates = {
        'computer': (128, 512, 1024, 2048, 4096, 8250),
        'photo': (128, 512, 1024, 2048, 4590),
        'reddit': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
        # 'ogbn-arxiv': (128, 512, 3072, 6144, 12288, 24576, 49152, 90941),
        'ogbn-products': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196615),
        # 'ogbn-arxiv': ('4,4', '8,8', '16,16', '32,32', 'mode1-1'),
        'ogbn-arxiv': ('2,2','4,4', '8,8', '16,16', '32,32'),
        'ogbn-arxiv': ('4,4', '5,10', '8,8', '10,15',  '10,25', '16,16', '32,32'),
        'ogbn-arxiv': ('4,4', '5,10', '8,8', '10,15',  '10,25', '16,16', '32,32'),
        'ogbn-arxiv': [round(x, 2) for x in np.arange(0.1, 1.01, 0.1).tolist()],
    }

    datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit']
    datasets = ['ogbn-arxiv']

    params={
        'axes.labelsize': '9',
        'xtick.labelsize':'9',
        'ytick.labelsize':'9',
        'lines.linewidth': 1,
        'legend.fontsize': '9',
        'figure.figsize' : '2.5, 2',
        'legend.loc': 'upper center', #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        # Times New Roman
        'font.family': 'Arial',
        'font.serif': 'Arial',
      }

    # draw_all(myparams, datasets, sample_rates, pdf_dir=f'./nts-old-sample', suffix='0.01')
    create_dir('./pdf')
    draw_all(params, datasets, sample_rates, pdf_dir=f'./pdf', suffix='1024')
