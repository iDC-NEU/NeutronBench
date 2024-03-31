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
            log_file = f'./log/aliyun/log/{dirname}/{ds}.log'
            print(log_file)
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
    # plt.style.use('classic')
    plt.style.use("seaborn-paper")
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    # plt.style.use("seaborn-deep")
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
    color_list = [
        '#d485df',
        '#89ca4d',
        '#a4edee',
        '#f09183',
    ]
    color_list = [
        '#ffeba9',
        '#eff2bd',
        '#c2e3c4',
        '#dcb4e0',
    ]
    color_list = [
        '#d485df',
        '#8cb5db',
        '#a4edee',
        '#f09183',
    ]

    # color_list = ['#44a185','#83ae93','#ead8ba','#ebb561','#d4452c','#783923','',]
    # color_list = ['#44a185','#ead8ba','#ebb561','#d4452c','',]

    n = Y.shape[1]
    ind = np.arange(n)  # the x locations for the groups
    pre_bottom = np.zeros(len(Y[0]))
    for i, y in enumerate(Y):
        plt.bar(ind,
                y,
                width,
                color=color_list[i],
                label=labels[i],
                bottom=pre_bottom,
                linewidth=params['lines.linewidth'],
                edgecolor='black')
        pre_bottom += y

    plt.xticks(np.arange(n), xticks, rotation=25)
    plt.ylim(0, 100)

    if anchor:
        # plt.legend(ncol=2, bbox_to_anchor=anchor)
        plt.legend(ncol=4,
                   bbox_to_anchor=anchor,
                   columnspacing=.5,
                   handletextpad=.15,
                   handleheight=1,
                   handlelength=1.2)  #
        # abelspacing=.2,
    else:
        plt.legend(ncol=len(labels))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=2)

    axes = plt.gca()  # get current axes
    fmt = '%.0f'  # Format you want the ticks, e.g. '40%'
    ticks_fmt = mtick.FormatStrFormatter(fmt)
    axes.yaxis.set_major_formatter(ticks_fmt)  # set % format to ystick.

    axes.spines['bottom'].set_linewidth(params['lines.linewidth'])
    ###设置底部坐标轴的粗细
    axes.spines['left'].set_linewidth(params['lines.linewidth'])
    ####设置左边坐标轴的粗细
    axes.spines['right'].set_linewidth(params['lines.linewidth'])
    ###设置右边坐标轴的粗细
    axes.spines['top'].set_linewidth(params['lines.linewidth'])
    ####设置上部坐标轴的粗细

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
    print(figpath, 'is plot.')
    plt.close()


def plot_multi_stack_bar(plot_params,
                         Y,
                         labels,
                         xlabel,
                         ylabel,
                         xticks,
                         anchor=None,
                         figpath=None):
    # plt.rcParams.update(plt.rcParamsDefault)
    # print(plt.rcParams.keys())
    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # print(plt.style.available)
    # plt.style.use('classic')
    # plt.style.use("seaborn-paper")
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    # plt.style.use("seaborn-deep")
    # plt.style.use("seaborn-notebook")
    # plt.style.use("seaborn-poster")
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Systems = ['P3', 'Metis*', 'DistDGL', 'SALIENT++', 'ByteGNN']

    width = 0.15
    color_list = ['b', 'g', 'c', 'r', 'm']

    color_list = [
        'C0',
        'C1',
        'C2',
        'C3',
    ]
    hatch_list = ['xx', '..', '**', '++']

    fig, ax = plt.subplots()

    n = len(Y[0][0])
    gap = 0
    # ind = np.arange(n) - width/2 - gap/2              # the x locations for the groups
    h_legends = []
    e_legends = []
    offset = (n + 1) * width / -2
    for j, Y_ in enumerate(Y):
        pre_bottom = np.zeros(n)
        ind = np.arange(
            n) + offset + j * width - gap / 2  # the x locations for the groups
        for i, y in enumerate(Y_):
            print(Systems[j], y)
            # leg1 = plt.bar(ind,y,width,color=color_list[i], label =labels[i],  hatch=hatch_list[i], bottom=pre_bottom, linewidth=params['lines.linewidth'], edgecolor='white')
            leg1 = plt.bar(ind,
                           y,
                           width,
                           color=color_list[i],
                           hatch=hatch_list[i],
                           bottom=pre_bottom,
                           linewidth=params['lines.linewidth'],
                           edgecolor='white')
            leg2 = plt.bar(ind,
                           y,
                           width,
                           color='none',
                           bottom=pre_bottom,
                           lw=.5,
                           edgecolor='black')
            h_legends.append(leg1)
            e_legends.append(leg2)
            pre_bottom += y

        # Systems = ['Hash (P3)', 'Metis (Metis*)', 'Metis (DistDGL)', 'Metis (SALIENT++)', 'Streaming (ByteGNN)']
        fontsize = 6
        text_offset = 0
        for x in ind:
            # plt.text(x-text_offset, 101, Systems[j], horizontalalignment='center', verticalalignment='bottom', fontsize=8, rotation=90, fontweight='bold')
            plt.text(x - text_offset,
                     101,
                     Systems[j],
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     fontsize=8,
                     rotation=90)

    # ax.arrow(-width*2,0,-.1,10, color='C0', hatch='xx')
    # ax.arrow(-width*2,0,-.1,10,color='C0', hatch='xx', edgecolor='white',width=0.01,head_width=0.05,head_length=1.5,overhang=1.)
    ax.annotate("0.08%",
                xy=(-.3, 0),
                xytext=(-.6, -19),
                arrowprops=dict(arrowstyle="->", color='C0'),
                color='C0',
                fontsize=9)
    ax.annotate("0.18%",
                xy=(1 - .3, 0),
                xytext=(1 - .6, -19),
                arrowprops=dict(arrowstyle="->", color='C0'),
                color='C0',
                fontsize=9)
    ax.annotate("0.08%",
                xy=(2 - .3, 0),
                xytext=(2 - .6, -19),
                arrowprops=dict(arrowstyle="->", color='C0'),
                color='C0',
                fontsize=9)

    plt.xticks(np.arange(n), xticks, rotation=0)
    plt.ylim(0, 100)
    plt.yticks([0, 20, 40, 60, 80, 100])

    if anchor:
        # plt.legend(ncol=2, bbox_to_anchor=anchor)
        # leg = plt.legend(ncol=2, bbox_to_anchor=anchor, columnspacing=2, labelspacing=.3, handletextpad=.15 , handleheight=1, handlelength=1.2)#
        legs = [(x, y) for x, y in zip(h_legends, e_legends)]
        plt.legend(legs,
                   labels,
                   ncol=2,
                   bbox_to_anchor=anchor,
                   columnspacing=2,
                   labelspacing=.3,
                   handletextpad=.35,
                   handleheight=1,
                   handlelength=1.2)  #

        # for legobj in leg.legendHandles:
        #   legobj.set_linewidth(.5)
        # legobj.set_edgecolor('black')
        # legobj.set_hatchcolor('white')
        # abelspacing=.2,
    else:
        plt.legend(ncol=len(labels))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=2)

    axes = plt.gca()  # get current axes
    fmt = '%.0f'  # Format you want the ticks, e.g. '40%'
    ticks_fmt = mtick.FormatStrFormatter(fmt)
    axes.yaxis.set_major_formatter(ticks_fmt)  # set % format to ystick.

    axes.spines['bottom'].set_linewidth(params['lines.linewidth'])
    ###设置底部坐标轴的粗细
    axes.spines['left'].set_linewidth(params['lines.linewidth'])
    ####设置左边坐标轴的粗细
    axes.spines['right'].set_linewidth(params['lines.linewidth'])
    ###设置右边坐标轴的粗细
    axes.spines['top'].set_linewidth(params['lines.linewidth'])
    ####设置上部坐标轴的粗细
    axes.spines[['right', 'top']].set_visible(False)
    axes.tick_params(bottom=False)

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边

    print(figpath, 'is plot.')
    plt.close()


if __name__ == '__main__':
    datasets = ['Amazon', 'Products', 'Reddit']
    # Systems = ['P3', 'Metis*', 'DistDGL', 'SALIENT++', 'ByteGNN']
    Systems = [
        'Hash', 'Metis-V', 'Metis-VE', 'Metis-VET', 'Stream-V', 'Stream-B'
    ]

    partition_time = {
        'AmazonStream-VPT': 211988,
        'RedditStream-VPT': 23712,
        'ProductsStream-VPT': 93756,
        'AmazonHashPT': 1.32,
        'AmazonMetis-VPT': 87.67,
        'AmazonMetis-VEPT': 102.91,
        'AmazonMetis-VETPT': 104.32,
        'AmazonStream-BPT': 4634.80,
        'ProductsHashPT': 1.76,
        'ProductsMetis-VPT': 27.70,
        'ProductsMetis-VEPT': 31.88,
        'ProductsMetis-VETPT': 40.42,
        'ProductsStream-BPT': 4604.92,
        'RedditHashPT': 0.17,
        'RedditMetis-VPT': 8.37,
        'RedditMetis-VEPT': 10.93,
        'RedditMetis-VETPT': 10.89,
        'RedditStream-BPT': 4573.52,
    }

    train_time = {
        'AmazonStream-VTT': 19.73,
        'RedditStream-VTT': 5,
        'ProductsStream-VTT': 3.825,
        'AmazonHashTT': 20.05,
        'AmazonMetis-VTT': 11.23,
        'AmazonMetis-VETT': 9.83,
        'AmazonMetis-VETTT': 9.73,
        'AmazonStream-BTT': 20.13,
        'ProduTTsHashTT': 5.1903,
        'ProduTTsMetis-VTT': 2.7366,
        'ProduTTsMetis-VETT': 2.8637,
        'ProduTTsMetis-VETTT': 2.8210,
        'ProduTTsStream-BTT': 3.9445,
        'RedditHashTT': 5.6135,
        'RedditMetis-VTT': 2.5836,
        'RedditMetis-VETT': 2.7740,
        'RedditMetis-VETTT': 2.7105,
        'RedditStream-BTT': 4.6549,
    }

    convergence_time = {
        'AmazonStream-VCT': 1351.3527,
        'RedditStream-VCT': 152.8649,
        'ProductsStream-VCT': 601.2642,
        'AmazonHashCT': 1684.1918,
        'AmazonMetis-VCT': 1232.136,
        'AmazonMetis-VECT': 1148.3979,
        'AmazonMetis-VETCT': 971.8273,
        'AmazonStream-BCT': 1253.206,
        'ProductsHashCT': 1001.3743,
        'ProductsMetis-VCT': 774.3417,
        'ProductsMetis-VECT': 744.6696,
        'ProductsMetis-VETCT': 525.9088,
        'ProductsStream-BCT': 925.6982,
        'RedditHashCT': 218.6324,
        'RedditMetis-VCT': 281.9299,
        'RedditMetis-VECT': 171.0852,
        'RedditMetis-VETCT': 142.2276,
        'RedditStream-BCT': 364.3709,
    }

    # partition_time = {
    #   'AmazonP3PT': 1.32, 'AmazonMetis*PT': 87.67 ,'AmazonDistDGLPT': 102.91,'AmazonSALIENT++PT': 104.32,'AmazonByteGNNPT': 4634.80,
    #   'ProductsP3PT': 1.76, 'ProductsMetis*PT': 27.70 ,'ProductsDistDGLPT': 31.88,'ProductsSALIENT++PT': 40.42,'ProductsByteGNNPT': 4604.92,
    #   'RedditP3PT': 0.17, 'RedditMetis*PT': 8.37,'RedditDistDGLPT': 10.93,'RedditSALIENT++PT':10.89 ,'RedditByteGNNPT':4573.52 ,
    # }

    # train_time = {
    #   'AmazonP3TT':20.05, 'AmazonMetis*TT':11.23,'AmazonDistDGLTT':9.83,'AmazonSALIENT++TT':9.73,'AmazonByteGNNTT':20.13,
    #   'ProduTTsP3TT':5.1903, 'ProduTTsMetis*TT':2.7366 ,'ProduTTsDistDGLTT':2.8637, 'ProduTTsSALIENT++TT':2.8210,'ProduTTsByteGNNTT':3.9445,
    #   'RedditP3TT':5.6135, 'RedditMetis*TT':2.5836,'RedditDistDGLTT':2.7740,'RedditSALIENT++TT':2.7105 ,'RedditByteGNNTT':4.6549,
    # }

    # convergence_time = {
    #   'AmazonP3CT':1684.1918, 'AmazonMetis*CT': 1232.136,'AmazonDistDGLCT':1148.3979,'AmazonSALIENT++CT':971.8273,'AmazonByteGNNCT':1253.206,
    #   'ProductsP3CT':1001.3743, 'ProductsMetis*CT':774.3417 ,'ProductsDistDGLCT':744.6696, 'ProductsSALIENT++CT':525.9088,'ProductsByteGNNCT':925.6982,
    #   'RedditP3CT':218.6324, 'RedditMetis*CT':281.9299,'RedditDistDGLCT':171.0852,'RedditSALIENT++CT':142.2276 ,'RedditByteGNNCT': 364.3709,
    # }

    Y_pt, Y_ct = [], []
    for sys in Systems:
        tmp_pt, tmp_ct = [], []
        for ds in datasets:
            tmp_pt.append(partition_time[ds + sys + 'PT'])
            tmp_ct.append(convergence_time[ds + sys + 'CT'])
        Y_ct.append(tmp_ct)
        Y_pt.append(tmp_pt)

    Y_pt = np.array(Y_pt)
    Y_ct = np.array(Y_ct)

    # print(Y_pt)
    # print(Y_ct)

    all_time = Y_pt + Y_ct
    Y_pt = Y_pt / all_time * 100
    Y_ct = Y_ct / all_time * 100

    # print(Y_pt)
    # print(Y_ct)
    Y = []
    for i in range(len(Y_pt)):
        Y.append([Y_pt[i], Y_ct[i]])

    params = {
        'axes.labelsize': '11',
        'xtick.labelsize': '11',
        'ytick.labelsize': '11',
        'lines.linewidth': 1,
        # 'legend.fontsize': '14.7',
        'legend.fontsize': '11',
        'figure.figsize': '6, 1.5',
        # 'figure.figsize' : '4, 2',
        'legend.loc': 'upper center',  #[]"upper right", "upper left"]
        'legend.frameon': False,
        # Times New Roman
        'font.family': 'Arial',
        'font.serif': 'Arial',
    }

    # print(pylab.rcParams.keys())

    # labels = ['sample', 'gather', 'transfer', 'train']
    # labels = ['Sample', 'Gather', 'Transfer', 'NN']
    # labels = ['Graph Partitioning', 'Batch Preparation', 'Data Transferring', 'NN Computation']
    # labels = ['Partitioning time', 'Convergence time']
    labels = ['Partitioning time', 'Training time']
    # xticks = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki']
    # xticks = ['ogb-arxiv', 'ogb-products', 'lj-links', 'lj-large', 'enwiki-links']
    # datasets = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links']

    # avg_percent = [f'{x:.2f}%' for x in np.average(diff_stage_time, axis=1)]
    # print('\n', labels, 'avg time:\n', avg_percent)

    xticks = datasets

    xlabel = ''
    ylabel = 'Norm. Execute Time (%)'

    create_dir('./pdf')
    plot_multi_stack_bar(params,
                         Y,
                         labels,
                         xlabel,
                         ylabel,
                         xticks,
                         anchor=(0.5, 1.75),
                         figpath='./pdf/PT-CT.pdf')
