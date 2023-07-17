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
  # plt.style.use("grayscale")
  plt.style.use("classic")
  # plt.style.use("bmh")
  # plt.style.use("ggplot")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  makrer_list = ['o', 's', 'v', 'p', '*', 'd', 'X', 'D',    'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [10,10,10,10,10,10,10,10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
  color_list = ['b', 'g', 'k', 'c', 'm', 'y', 'r'] 
  color_list = ['#fdd37a','#f0633a','#99093d','#e6f397','#73c49a','#415aa4',]

  ax1 = plt.subplot(111)#figure1的子图1为ax1
  # ax1.plot(X, Y[0], label = labels[0], color='b', marker=makrer_list[0], markersize=5,markevery=marker_every[0])
  ax1_color = '#415aa4'
  ax2_color = '#f0633a'
  ax1_color = 'b'
  ax2_color = 'r'
  ax1.plot(X, Y[0], label = labels[0], color=ax1_color, marker=makrer_list[0], markersize=5)
    # plt.plot(x, y, label = labels[i], markersize=5)
  ax1.tick_params(axis='y', labelcolor=ax1_color)
  ax1.set_yticks(yticks[0])
  ax1.set_xticks(xticks)
  # ax1.set_xticks([0,125,250,375,500])  
  ############################
  # ax1.set_ylim(0.92, 0.94)
  # ax1.set_xlim(0, 500)
  ax1.set_ylim(ylim[0])
  ax1.set_xlim(xlim)
  
  ############################

  # ax1 = plt.gca()
  # ax1.grid(True)  # add grid

  ax1.set_xlabel(xlabel) 
  ax1.set_ylabel(ylabel[0], color=ax1_color, labelpad=0) 


  ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴
  # color = 'tab:blue'
  # ax2.set_ylabel('sin', color=color)
  # ax2.plot(t, data2, color=color)
  # ax2.tick_params(axis='y', labelcolor=color) 


  ax2.plot(X, Y[1], label = labels[1], color=ax2_color, marker=makrer_list[0], markersize=5)
    # plt.plot(x, y, label = labels[i], markersize=5)
  ax2.tick_params(axis='y', labelcolor=ax2_color)
  ax2.set_yticks(yticks[1])
  ax2.set_ylabel(ylabel[1], rotation=270, color=ax2_color, labelpad=15) 
  ax2.set_ylim(ylim[1])


#   plt.legend(ncol=2)
  lines, labels = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()

#   ax1.legend(lines + lines2, labels + labels2,ncol=2)
  
#   ax1.set_anchor((0.2, 10.3))
  plt.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, 1.12), ncol=2)

  figpath = './line.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()


def print_val_acc(mode, datasets, batch_sizes, suffix=None):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for ds in datasets:
        for bs in batch_sizes[ds]:
            val_acc_list = []
            if suffix:
                log_file = f'../log/batch-size-nts-old-adam/{ds}-{suffix}/{ds}-{bs}.log'
            else:
                log_file = f'../log/batch-size-nts-old-adam/{ds}/{ds}-{bs}.log'
            print(log_file)
            val_acc = parse_num(log_file, mode)
            val_acc_list += val_acc

            ret[ds + str(bs)] = val_acc_list
    # print(ret)
    return ret


def draw_all(myparams, datasets, batch_sizes, pdf_dir, suffix=None):
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
        'ogbn-arxiv': [0, 300],
        'ogbn-products': [0, 600],
    }
    
    target_acc = {
        'ogbn-arxiv': 0.684,
        'reddit': 0.93,
        'ogbn-products': 0.89,
    }


    if suffix:
        # val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes, "stop"+str(suffix))
        # train_time_dict = print_val_acc('train_time', datasets, batch_sizes, "stop"+str(suffix))
        val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes, str(suffix))
        train_time_dict = print_val_acc('train_time', datasets, batch_sizes, str(suffix))
    else:
        val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
        train_time_dict = print_val_acc('train_time', datasets, batch_sizes)

    for ds in datasets:
        Best_ACC_Y = []
        for bs in batch_sizes[ds]:
            Best_ACC_Y.append(np.max(val_acc_dict[ds+str(bs)]))
        # print(ds, suffix, batch_sizes[ds])
        # print(Best_ACC_Y, '\n')

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
        # print('time_to_acc time:', Y)
        
        Y = [np.array(Best_ACC_Y) * 100, np.array(Y)]
        # X = [np.log2(batch_sizes[ds]), np.log2(batch_sizes[ds])]
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


        labels = ['Val Acc (%)', 'Converge Time (s)']
        xlabel = 'Batch Size (log scale)'
        ylabel = ['Val ACC (%)', 'Converge Time (s)']
        

        
        create_dir(pdf_dir)
        if suffix:
            pdf_file = f'{pdf_dir}/{ds}-bs-exp-{suffix}.pdf'
        else:
            pdf_file = f'{pdf_dir}/{ds}-bs-exp.pdf'

        x_ticks = [round(x, 1) for x in X]
        y_ticks = [np.linspace(acc_ylim[ds][0], acc_ylim[ds][1], 5), np.linspace(speed_ylim[ds][0], speed_ylim[ds][1], 5)]
        y_lim = [acc_ylim[ds], speed_ylim[ds]]
        
        
        # plot_line(myparams, X, Best_ACC_Y, labels, xlabel, ylabel, x_ticks[:-1], y_ticks, (x_ticks[0], x_ticks[-1]), (0.68, 0.72), pdf_file)

        plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks[:-1], y_ticks, (x_ticks[0], x_ticks[-1]), y_lim, pdf_file)


if __name__ == '__main__':
    batch_sizes = {
        'computer': (128, 512, 1024, 2048, 4096, 8250),
        'photo': (128, 512, 1024, 2048, 4590),
        'reddit': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
        'ogbn-arxiv': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 90941),
        # 'ogbn-arxiv': (128, 512, 3072, 6144, 12288, 24576, 49152, 90941),
        'ogbn-products': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196615),
    }

    # datasets = ['reddit', 'ogbn-arxiv', 'ogbn-products', 'computer', 'photo']
    # datasets = ['ogbn-products']
    # datasets = ['ogbn-arxiv', 'reddit']
    # datasets = ['ogbn-products', 'reddit']
    # datasets = ['ogbn-arxiv']
    datasets = ['ogbn-arxiv']
    datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit']

    myparams = {
        'axes.labelsize': '12',
        'xtick.labelsize': '12',
        'ytick.labelsize': '12',
        # 'font.family': 'Times New Roman',
        'figure.figsize': '5, 4',  #图片尺寸
        'lines.linewidth': 2,
        'legend.fontsize': '12',
        # 'legend.loc': 'best', #[]"upper right", "upper left"]
        'legend.loc': 'upper center', #[]"upper right", "upper left"]
        'legend.numpoints': 1,
        'legend.frameon': False,
        # 'lines.ncol': 2,
    }

    # draw_all(myparams, datasets, batch_sizes, pdf_dir=f'./nts-old-sample', suffix='0.01')
    create_dir('./nts-old-sample')
    draw_all(myparams, datasets, batch_sizes, pdf_dir=f'./nts-old-sample', suffix='0.001')

    # for x in [20]:
    #     # draw_best_val_acc(myparams, datasets, batch_sizes, x)
    #     draw_all(myparams, datasets, batch_sizes, x)