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


def create_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def get_explicit_time(datasets, dirname='explicit'):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for optim in [
            'gcn_sample_time', 'gcn_gather_time', 'gcn_trans_time',
            'gcn_train_time'
    ]:
        time_list = []
        for ds in datasets:
            log_file = f'./log/{dirname}/{ds}.log'
            time_list += parse_num(log_file, optim)
        ret[optim] = time_list
    return ret


def plot_stack_bar(plot_params,
                   Y,
                   labels,
                   xlabel,
                   ylabel,
                   xticks,
                   anchor=None,
                   figpath=None):
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

    # color_list = ['#fdd37a','#e6f397','#73c49a','#415aa4',]
    # color_list = ['#223d4a','#236863','#e4bd5f','#e36347']
    # color_list = ['#223d4a','#236863','#259386','#80a870','#e4bd5f','#e36347']
    # color_list = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    # color_list = ['#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44', ]

    # color_list = ['#f0633a','#fdd37a','#e6f397','#73c49a']
    # color_list.reverse()
    # color_list = ['#99093d','#f0633a','#fdd37a','#e6f397','#73c49a','#415aa4',]
    color_list = [
        '#99093d',
        '#f0633a',
        '#73c49a',
        '#415aa4',
    ]

    color_list = [
        '#f0633a',
        '#99093d',
        '#73c49a',
        '#415aa4',
    ]

    # color_list = ['#44a185','#83ae93','#ead8ba','#ebb561','#d4452c','#783923','',]
    # color_list = ['#44a185','#ead8ba','#ebb561','#d4452c','',]

    n = Y.shape[1]
    ind = np.arange(n)  # the x locations for the groups
    pre_bottom = np.zeros(len(Y[0]))
    for i, y in enumerate(Y):
        # plt.bar(ind+width*i,y,width,color=color_list[i], label =labels[i])
        # print(ind, y)
        plt.bar(ind,
                y,
                width,
                color=color_list[i],
                label=labels[i],
                bottom=pre_bottom)
        pre_bottom += y

    plt.xticks(np.arange(n), xticks)
    plt.ylim(0, 100)
    # plt.yticks(np.linspace(0, 100, 6), ('0%','20%','40%','60%','80%','100%'))
    # plt.yticks(np.arange(5), ('0%','20%','40%','60%','80%','100%'))

    if anchor:
        plt.legend(ncol=2, bbox_to_anchor=anchor)
    else:
        plt.legend(ncol=len(labels))

    # num1, num2 = 1, 1.2
    # plt.legend(ncol=4, bbox_to_anchor=(num1, num2))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the formatter
    axes = plt.gca()  # get current axes
    fmt = '%.0f'  # Format you want the ticks, e.g. '40%'
    ticks_fmt = mtick.FormatStrFormatter(fmt)
    axes.yaxis.set_major_formatter(ticks_fmt)  # set % format to ystick.
    axes.grid(axis='y')
    # axes.grid(axis='x')

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
    print(figpath, 'is plot.')
    plt.close()


if __name__ == '__main__':
    datasets = ['ogbn-arxiv']
    datasets = [
        'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links',
        'ogbn-products'
    ]
    datasets = [
        'ogbn-products', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links'
    ]
    ret = get_explicit_time(datasets)
    for k, v in ret.items():
        print(k, v)
    diff_stage_time = np.array(list(ret.values()))
    print('\nconvert to numpy array:\n', diff_stage_time)
    epoch_time = diff_stage_time.sum(axis=0)
    print('\nepoch time of different dataset:\n', epoch_time)
    diff_stage_time /= epoch_time
    diff_stage_time *= 100  # fmt
    print('\nnormalized time:\n', diff_stage_time)
    # print(diff_stage_time)
    avg_percent = [f'{x:.2%}' for x in np.average(diff_stage_time, axis=1)]
    print('\nsample, gather, transfer, train, avg%:\n', avg_percent)
    # x_name = ['arxiv', 'products', 'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki']

    # params={
    #   'axes.labelsize': '14',
    #   'xtick.labelsize':'14',
    #   'ytick.labelsize':'14',
    #   'lines.linewidth': 1,
    #   # 'legend.fontsize': '14.31',
    #   'legend.fontsize': '14',
    #   'figure.figsize' : '8, 4',
    #   'legend.loc': 'upper center', #[]"upper right", "upper left"]
    #   'legend.frameon': False,

    # }

    params = {
        'axes.labelsize': '15',
        'xtick.labelsize': '15',
        'ytick.labelsize': '15',
        'lines.linewidth': 2,
        # 'legend.fontsize': '14.7',
        'legend.fontsize': '15',
        'figure.figsize': '8, 4',
        'legend.loc': 'upper center',  #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        'font.family': 'Times New Roman',
        'font.serif': 'Times New Roman',
    }

    # print(pylab.rcParams.keys())

    labels = ['sample', 'gather', 'transfer', 'train']
    # xticks = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki', 'ogbn-products']
    xticks = datasets
    xlabel = ''
    ylabel = 'Norm. Execute Time (%)'
    # labels = list(ret.keys())
    plot_stack_bar(params,
                   diff_stage_time,
                   labels,
                   xlabel,
                   ylabel,
                   xticks,
                   anchor=(0.5, 1.25),
                   figpath='explicit-breakdown.pdf')

    tmp = np.ones(5)
    create_dir('./explicit_txt')
    for i, x in enumerate(diff_stage_time):
        tmp += x
        with open(f'./explicit_txt/{labels[i]}.txt', 'w') as f:
            for j, y in enumerate(tmp):
                f.write(str(j) + ' ' + str(y) + '\n')

    exit(0)
