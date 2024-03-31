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
def plot_line(plot_params,
              X,
              Y,
              labels,
              xlabel,
              ylabel,
              xticks,
              yticks,
              xlim,
              ylim,
              figpath=None):

    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    makrer_list = ['o', 's', 'v', 'p', '*', 'd', 'X', 'D']
    marker_every = [10, 10, 10, 10, 10, 10, 10]
    color_list = [
        'C7',
        'C8',
        'C2',
        'C5',
        'C4',
        'C3',
    ]
    axes1 = plt.subplot(111)  #figure1的子图1为axes1
    for i, (x, y) in enumerate(zip(X, Y)):
        y = np.array(y) * 100
        # print(i, labels[i])
        plt.plot(x, y, label=labels[i], color=color_list[i])
    axes1.set_yticks(yticks)
    axes1.set_xticks(xticks)
    ############################
    print(y_lim, yticks)
    axes1.set_ylim(ylim[0], ylim[1])
    axes1.set_xlim(xlim[0], xlim[1])

    # axes1.spines['bottom'].set_linewidth(myparams['lines.linewidth']);###设置底部坐标轴的粗细
    # axes1.spines['left'].set_linewidth(myparams['lines.linewidth']);####设置左边坐标轴的粗细
    # axes1.spines['right'].set_linewidth(myparams['lines.linewidth']);###设置右边坐标轴的粗细
    # axes1.spines['top'].set_linewidth(myparams['lines.linewidth']);####设置上部坐标轴的粗细

    # plt.legend(ncol=2)
    # plt.legend(ncol=3, bbox_to_anchor=(1.08, 1.28), columnspacing=0.5, handletextpad=.1, labelspacing=.1, handlelength=1.5)
    plt.legend(ncol=3,
               columnspacing=1.2,
               handletextpad=.3,
               labelspacing=.1,
               handlelength=.8,
               bbox_to_anchor=(0.5, 1.3))

    plt.ylabel(ylabel, labelpad=2)
    plt.xlabel(xlabel)

    figpath = './log/batch-size/reddit-exp5/plot.pdf' if not figpath else figpath
    plt.savefig(figpath,
                dpi=1000,
                bbox_inches='tight',
                pad_inches=0,
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
    print(figpath, 'is plot.')
    plt.close()


# 每隔time_skip对acc取一个平均值
def split_list(X, Y, time_skip):
    retX, retY = [], []
    for arrx, arry in zip(X, Y):
        tmpx, tmpy = [], []
        pre, idx = 0, 0
        for i in range(len(arrx)):
            x, y = arrx[i], arry[i]
            if x >= idx * time_skip:
                tmpx.append(x)
                tmpy.append(np.average(arry[pre:i + 1]))
                pre = i + 1
                idx += 1
        if pre < len(arrx):
            tmpx.append(arrx[-1])
            tmpy.append(np.average(arry[pre:]))

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
                nums = re.findall(r"\d+\.?\d*", line[line.find(mode):])
                ret.append(float(nums[0]))
    return ret


# 每隔time_skip对acc取一个平均值
def split_list_best(X, Y, best_acc):
    retX, retY = [], []
    for arrx, arry in zip(X, Y):
        tmpx, tmpy = [], []
        for i in range(len(arrx)):
            x, y = arrx[i], arry[i]
            tmpx.append(x)
            tmpy.append(y)
            if y >= best_acc:
                # tmpy[-1] = best_acc
                break
        retX.append(tmpx)
        retY.append(tmpy)
    return retX, retY


if __name__ == '__main__':
    #  = {
    #   'axes.labelsize': '10',
    #   'xtick.labelsize': '10',
    #   'ytick.labelsize': '10',
    #   # 'font.family': 'Times New Roman',
    #   'figure.figsize': '4, 3',  #图片尺寸
    #   'lines.linewidth': 1,
    #   'legend.fontsize': '8',
    #   'legend.loc': 'best', #[]"upper right", "upper left"]
    #   'legend.numpoints': 1,
    #   # 'lines.ncol': 2,
    # }

    myparams = {
        'axes.labelsize': '10',
        'xtick.labelsize': '10',
        'ytick.labelsize': '10',
        'lines.linewidth': 1,
        # 'axes.linewidth': 10,
        # 'bars.linewidth': 100,
        'legend.fontsize': '10',
        'figure.figsize': '2.8, 1.8',
        'legend.loc': 'upper center',  #[]"upper right", "upper left"]
        # 'legend.loc': 'best', #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        # 'font.family': 'Times New Roman',
        'font.family': 'Arial',
        'font.serif': 'Arial',
    }
    # def print_val_acc(mode, datasets, batch_sizes, suffix=None):
    batch_sizes = {
        # 'reddit': (512, 1024,  8192, 16384, 32768,  153431),
        'reddit':
        (128, 512, 1024, 2048, 4096, 8192, 16384, 65536, 153431, 'mix'),
        'reddit': (512, 4096, 8192, 16384, 65536, 'mix'),
        # 'ogbn-arxiv': (128,  1024, 2048, 4096,  16384,  90941,'mix6'),dddd
        # 'ogbn-arxiv': (128, 512, 3072, 6144, 12288, 24576, 49152, 90941),
        'ogbn-arxiv':
        (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 90941, 'mix'),
        # 'ogbn-products': (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,'mix'),
        # 'ogbn-products': (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 'mix'),
        'ogbn-products': (512, 2048, 4096, 8192, 16384, 'mix'),
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
    # datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit']
    datasets = ['reddit', 'ogbn-products']

    for ds in datasets:
        val_acc = []
        run_time = []
        for bs in batch_sizes[ds]:
            log_file = f'./log/{ds}-0.001/{ds}-{bs}.log'
            print(ds, log_file)

            val_acc.append(parse_num(log_file, 'val_acc'))
            run_time.append(parse_num(log_file, 'gcn_run_time'))
        if ds == 'reddit':
            yticks = np.linspace(92, 94, 3)
            y_lim = (92, 94)
            print(yticks, y_lim)
        elif ds == 'ogbn-arxiv':
            yticks = np.linspace(*acc_ylim[ds], 5)
            y_lim = acc_ylim[ds]
        else:
            yticks = np.linspace(84, 92, 3)
            y_lim = (84, 92)
        xticks = np.linspace(0, 300, 5)
        labels = list(batch_sizes[ds])
        labels[-1] = 'Mix'
        xlabel = 'Run Time (s)'
        ylabel = 'Accuracy (%)'

        max_rate = max([max(x) for x in val_acc[:-1]])
        run_time_skip, val_acc_skip = split_list_best(run_time, val_acc,
                                                      max_rate)
        print('max_rate', max_rate)
        print('bs', batch_sizes[ds])
        print('max_acc', [max(x) for x in val_acc])
        print('time', [x[-1] for x in run_time_skip])

        run_time_skip, val_acc_skip = split_list(run_time, val_acc, 25)
        # run_time_skip,val_acc_skip = run_time,val_acc
        max_rate = max([max(x) for x in val_acc_skip[:-1]])
        print(f'{ds} max_rate {max_rate}')
        # run_time_skip,val_acc_skip = split_list_best(run_time_skip,val_acc_skip, max_rate)
        #   run_time_skip,val_acc_skip = run_time,val_acc
        create_dir('./pdf')
        plot_line(myparams, run_time_skip, val_acc_skip, labels, xlabel,
                  ylabel, xticks, yticks, (0, 300), y_lim,
                  f'./pdf/adaptive-batchsize-{ds}.pdf')
