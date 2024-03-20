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
  pylab.rcParams.update(plot_params)  #更新自己的设置
  plt.rcParams['pdf.fonttype'] = 42

  makrer_list = ['o', 'v', 'v', 'p', '*', 'd', 'X', 'D',    'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
  marker_every = [10,10,10,10,10,10,10,10,10,10,10,10,10,10]
  color_list = ['C1','C0',]
  ax1 = plt.subplot(111)#figure1的子图1为ax1

  ax1.plot(X, Y[0], label = labels[0], color=color_list[0], marker=makrer_list[0], markersize=3)
  ax1.set_yticks(yticks[0])
  ax1.set_xticks(xticks)
  ax1.set_ylim(ylim[0])
  ax1.set_xlim(xlim)
  
  ax1.set_xlabel(xlabel) 
  ax1.set_ylabel(ylabel[0], labelpad=1) 


  ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴
  ax2.plot(X, Y[1], label = labels[1], color=color_list[1], marker=makrer_list[1], markersize=3)
  ax2.set_yticks(yticks[1])
  ax2.set_ylabel(ylabel[1], rotation=90, labelpad=3) 
  ax2.set_ylim(ylim[1])

  lines, labels = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()

  plt.legend(lines + lines2, labels + labels2, ncol=2, bbox_to_anchor=(0.5, 1.2), handlelength=1, columnspacing=0.5, handletextpad=.2, labelspacing=.2)

  figpath = './line.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', pad_inches=0, format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()


def print_val_acc(mode, datasets, batch_sizes, suffix=None):
    ret = {}
    for ds in datasets:
        for bs in batch_sizes[ds]:
            val_acc_list = []
            if suffix:
                log_file = f'./log/{ds}-{suffix}/{ds}-{bs}.log'
            else:
                log_file = f'./log/{ds}/{ds}-{bs}.log'
            print(log_file)
            val_acc = parse_num(log_file, mode)
            # print('parsing:', log_file)
            val_acc_list += val_acc
            ret[ds + str(bs)] = val_acc_list
    return ret


def draw_all(myparams, datasets, batch_sizes, pdf_dir, suffix=None):
    acc_ylim = {
        'reddit': [92.5, 94.5],
        'ogbn-arxiv': [70.5, 72.5],
        'ogbn-products': [76, 92],
    }

    speed_ylim = {
        'reddit': [0, 300],
        'ogbn-arxiv': [0, 300],
        'ogbn-products': [0, 600],
    }
    
    target_acc = {
        'ogbn-arxiv': 0.684,
        'reddit': 0.93,
        'ogbn-products': 0.89,
    }


    if suffix:
        val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes, str(suffix))
        train_time_dict = print_val_acc('train_time', datasets, batch_sizes, str(suffix))
    else:
        val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
        train_time_dict = print_val_acc('train_time', datasets, batch_sizes)

    for ds in datasets:
        Best_ACC_Y = []
        for bs in batch_sizes[ds]:
            # print(ds+str(bs))
            # print(np.max(val_acc_dict[ds+str(bs)]))
            Best_ACC_Y.append(np.max(val_acc_dict[ds+str(bs)]))

        Time_X, Y = [], []
        all_avg_train_time = []
        for i, bs in enumerate(batch_sizes[ds]):
            Y.append(val_acc_dict[ds + str(bs)])
            avg_train_time = np.average(train_time_dict[ds + str(bs)][3:])
            all_avg_train_time.append(round(avg_train_time, 3))
            Time_X.append(np.cumsum([avg_train_time] * len(Y[-1])))
            # print('batch_size', bs, 'avg_epoch_time', round(avg_train_time, 2))

        # tiling X,Y along that large val_acc
        target_acc[ds] = np.max(Best_ACC_Y) * 0.98
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
        
        Y = [np.array(Best_ACC_Y) * 100, np.array(Y)]
        X = np.log2(batch_sizes[ds])

        def print_xlx(info, X):
            print(info, ' '.join([str(round(x, 3)) for x in X]))

        print()
        print(ds)
        print_xlx('batch_size', X)
        print_xlx('best_acc', Y[0])
        print_xlx('speed', Y[1])
        print_xlx('epoch_time', all_avg_train_time)
        print('min', np.min(Best_ACC_Y), 'max', np.max(Best_ACC_Y), 'target', target_acc[ds])


        labels = ['Accuracy (%)', 'Converge Time (s)']
        xlabel = 'Batch Size (log scale)'
        ylabel = ['Accuracy (%)', 'Converge Time (s)']
        
        create_dir(pdf_dir)
        if suffix:
            pdf_file = f'{pdf_dir}/batchsize-acc-{ds}-{suffix}.pdf'
        else:
            pdf_file = f'{pdf_dir}/batchsize-acc-{ds}.pdf'

        x_ticks = [round(x, 1) for x in X]
        x_lim = (x_ticks[0] - 1, x_ticks[-1] + 1)
        x_ticks = np.arange(x_ticks[0], x_ticks[-1], 1)
        x_ticks = x_ticks[::2]
        if ds == 'reddit':
            y_ticks = [np.arange(92, 95, 1), np.linspace(speed_ylim[ds][0], speed_ylim[ds][1], 5)]
            y_lim = [[92,94.5], speed_ylim[ds]]
            print(y_ticks)
        elif ds == 'ogbn-arxiv':
            y_ticks = [np.linspace(70, 72, 3), np.linspace(speed_ylim[ds][0], speed_ylim[ds][1], 5)]
            y_lim = [acc_ylim[ds], speed_ylim[ds]]
            y_lim = [[70,72.3], [0, 350]]
        else:
            y_ticks = [np.linspace(acc_ylim[ds][0], acc_ylim[ds][1], 5), np.linspace(speed_ylim[ds][0], speed_ylim[ds][1], 5)]
            # y_lim = [acc_ylim[ds], speed_ylim[ds]]
            y_lim = [[75,92], [0, 500]]

        # plot_line(myparams, X, Best_ACC_Y, labels, xlabel, ylabel, x_ticks[:-1], y_ticks, (x_ticks[0], x_ticks[-1]), (0.68, 0.72), pdf_file)
        plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks, y_ticks, x_lim, y_lim, pdf_file)


if __name__ == '__main__':
    batch_sizes = {
        # 'ogbn-arxiv': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 90941),
        'ogbn-products': (32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196615),
        'reddit': (32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
    }

    datasets = ['ogbn-products', 'reddit']

    params={
        'axes.labelsize': '9',
        'xtick.labelsize':'9',
        'ytick.labelsize':'9',
        'lines.linewidth': 1,
        # 'axes.linewidth': 10,
        # 'bars.linewidth': 100,
        'legend.fontsize': '9',
        'figure.figsize' : '2.5, 2',
        'legend.loc': 'upper center', #[]"upper right", "upper left"]
        'legend.frameon': False,
        'font.family': 'Arial',
        'font.serif': 'Arial',
      }

    create_dir('./pdf')
    draw_all(params, datasets, batch_sizes, pdf_dir=f'./pdf', suffix='0.001')
