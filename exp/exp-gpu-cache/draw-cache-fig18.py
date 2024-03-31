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
    'reddit':
    "VERTICES:232965\nEDGE_FILE:../data/reddit/reddit.edge\nFEATURE_FILE:../data/reddit/reddit.feat\nLABEL_FILE:../data/reddit/reddit.label\nMASK_FILE:../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
    'ogbn-arxiv':
    "VERTICES:169343\nEDGE_FILE:../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
    'ogbn-products':
    "VERTICES:2449029\nEDGE_FILE:../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../data/ogbn-products/ogbn-products.label\nMASK_FILE:../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
    'enwiki-links':
    "VERTICES:13593032\nEDGE_FILE:../data/enwiki-links/enwiki-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'livejournal':
    "VERTICES:4846609\nEDGE_FILE:../data/livejournal/livejournal.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-large':
    "VERTICES:7489073\nEDGE_FILE:../data/lj-large/lj-large.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-links':
    "VERTICES:5204175\nEDGE_FILE:../data/lj-links/lj-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'europe_osm':
    "VERTICES:50912018\nEDGE_FILE:../data/europe_osm/europe_osm.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dblp-2011':
    "VERTICES:933258\nEDGE_FILE:../data/dblp-2011/dblp-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'frwiki-2013':
    "VERTICES:1350986\nEDGE_FILE:../data/frwiki-2013/frwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dewiki-2013':
    "VERTICES:1510148\nEDGE_FILE:../data/dewiki-2013/dewiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'itwiki-2013':
    "VERTICES:1016179\nEDGE_FILE:../data/itwiki-2013/itwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'hollywood-2011':
    "VERTICES:1985306\nEDGE_FILE:../data/hollywood-2011/hollywood-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'enwiki-2016':
    "VERTICES:5088560\nEDGE_FILE:../data/enwiki-2016/enwiki-2016.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
}


def create_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params,
              X,
              Y,
              ds,
              labels,
              xlabel,
              ylabel,
              xticks,
              yticks,
              xlim,
              ylim,
              figpath=None):

    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # plt.style.use("grayscale")
    # plt.style.use("classic")
    # plt.style.use("seaborn-paper")
    # plt.style.use("bmh")
    # plt.style.use("ggplot")
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
    # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
    makrer_list = [
        'o', 's', 'v', 'p', '*', 'd', 'X', 'D', 'o', 's', 'v', 'p', '*', 'd',
        'X', 'D'
    ]
    #   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
    marker_every = [5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10]
    # fig1 = plt.figure(1)
    # color_list = ['b', 'g', 'k', 'c', 'm', 'y', 'r']
    # color_list = ['#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44', ]
    color_list = ['C0', 'C1', 'C2']

    axes1 = plt.subplot(111)  #figure1的子图1为axes1
    for i, (x, y) in enumerate(zip(X, Y)):
        # plt.plot(x, y, label = labels[i], color=color_list[i], marker=makrer_list[i], markersize=5,markevery=marker_every[i])
        plt.plot(x,
                 y,
                 label=labels[i],
                 color=color_list[i],
                 linewidth=myparams['lines.linewidth'] + 1)
        # plt.plot(x, y, label = labels[i], markersize=5)
    axes1.set_yticks(yticks)
    axes1.set_xticks(xticks)
    # axes1.set_xticks([0,125,250,375,500])
    ############################
    # axes1.set_ylim(0.92, 0.94)
    # axes1.set_xlim(0, 500)
    axes1.set_ylim(ylim)
    axes1.set_xlim(xlim)
    axes1.set_ylabel(ylabel, labelpad=1)
    # plt.legend(ncol=1, frameon=False)
    # plt.legend(ncol=3, columnspacing=1.2, handletextpad=.3, labelspacing=.1, handlelength=.8, bbox_to_anchor=(0.5, 1.3))

    if ds == 'amazon':
        legend_pos = (.55, .5)
        title_pos = (.4, -.55)
        title_name = '(a) Amazon (power-law graph)'
    elif ds == 'ogbn-papers100M':
        legend_pos = (.35, .6)
        title_pos = (.38, -.55)
        title_name = '(b) OGB-Paper (non-power-law graph)'
    plt.title(title_name, x=title_pos[0], y=title_pos[1], fontsize=9.5)
    # ax1.set_title(title, x=.5, y=-.3, color=color, fontsize=14)
    plt.legend(ncol=1,
               columnspacing=1.2,
               bbox_to_anchor=legend_pos,
               handletextpad=.3,
               labelspacing=.1,
               handlelength=.8)

    ############################

    # axes1 = plt.gca()
    # axes1.grid(True)  # add grid

    # plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    axes1 = plt.subplot(111)  #figure1的子图1为ax1

    # Set the formatter
    # axes = plt.gca()   # get current axes
    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # ticks_fmt = mtick.FormatStrFormatter(fmt)
    # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
    # axes.xaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
    # axes1.grid(axis='y', linestyle='-', )
    # axes1.grid(axis='y', linestyle='', )

    axes1.spines['bottom'].set_linewidth(myparams['lines.linewidth'])
    ###设置底部坐标轴的粗细
    axes1.spines['left'].set_linewidth(myparams['lines.linewidth'])
    ####设置左边坐标轴的粗细
    axes1.spines['right'].set_linewidth(myparams['lines.linewidth'])
    ###设置右边坐标轴的粗细
    axes1.spines['top'].set_linewidth(myparams['lines.linewidth'])
    ####设置上部坐标轴的粗细

    figpath = './line.pdf' if not figpath else figpath
    plt.savefig(figpath,
                dpi=1000,
                bbox_inches='tight',
                pad_inches=0,
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
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
                nums = re.findall(r"\d+\.?\d*", line[line.find(mode):])
                ret.append(float(nums[0]))
    return ret


def print_diff_cache_ratio(datasets, log_path):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for ds in datasets:
        for cache_policy in ['degree', 'sample', 'random']:
            log_file = f'{log_path}/vary-rate-{cache_policy}/{ds}.log'
            print(log_file)
            cache_hit_rate = parse_num(log_file, 'gcn_cache_hit_rate')
            cache_rate = parse_num(log_file, 'gcn_cache_rate')
            ret[ds + cache_policy + 'rate'] = cache_rate
            ret[ds + cache_policy + 'hit'] = cache_hit_rate
    return ret


if __name__ == '__main__':

    myparams = {
        'axes.labelsize': '9',
        'xtick.labelsize': '9',
        'ytick.labelsize': '9',
        'lines.linewidth': 1,
        # 'legend.fontsize': '14.7',
        'legend.fontsize': '9',
        # 'figure.figsize' : '3, 2.5',
        'figure.figsize': '2.1, 1.35',
        # 'legend.loc': 'upper center', #[]"upper right", "upper left"]
        'legend.loc': 'best',  #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        # 'font.family': 'Times New Roman',
        'font.family': 'Arial',
        'font.serif': 'Arial',
    }

    datasets = [
        'hollywood-2011', 'lj-links', 'reddit', 'lj-links', 'enwiki-links',
        'ogbn-arxiv', 'livejournal', 'ogbn-products'
    ]
    datasets = [
        'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links',
        'ogbn-arxiv', 'ogbn-products'
    ]

    # datasets = ['reddit', 'hollywood-2011', 'lj-links', 'enwiki-links']
    datasets = ['road-usa']
    datasets = ['ogbn-products', 'reddit', 'ogbn-arxiv']
    datasets = ['hollywood-2011']
    datasets = ['rmat']
    datasets = ['ogbn-arxiv', 'ogbn-papers100M']
    datasets = [
        'reddit', 'lj-large', 'livejournal', 'lj-links', 'enwiki-links',
        'ogbn-arxiv', 'ogbn-products', 'hollywood-2011', 'ogbn-papers100M',
        'amazon'
    ]
    datasets = ['amazon', 'ogbn-papers100M']

    ret = print_diff_cache_ratio(datasets, './log')
    # ret = print_diff_cache_ratio(datasets, '../log/gpu-cache-dgl2')

    labels = ['random', 'degree', 'sample']
    for ds in datasets:
        X, Y = [], []
        for cache_policy in labels:
            cache_rate = ret[ds + cache_policy + 'rate']
            cache_hit_rate = ret[ds + cache_policy + 'hit']
            # print(ds, cache_policy)
            # print(cache_rate)
            # print(cache_hit_rate)
            # print()
            X.append(cache_rate)
            Y.append(cache_hit_rate)

        # print(Y)
        print(len(X[0]), len(Y[0]), len(Y[1]), len(Y[2]),
              min(len(Y[0]), len(Y[1]), len(Y[2])))
        # X = X[:,:min(len(Y[0]),len(Y[1]),len(Y[2]))]

        Y = np.array(Y) * 100
        X = np.array(X) * 100

        if '100M' in ds:
            x_ticks = np.linspace(0, 50, 6)
        else:
            x_ticks = np.linspace(0, 100, 6)
        y_ticks = np.linspace(0, 100, 6)
        y_lim = (0, 100)
        # y_name = [f'{x*100:.0f}' for x in y_ticks]
        create_dir('./pdf')
        pdf_file = f'./pdf/cache-{ds}.pdf'

        xlabel = 'Cache Ratio (%)'
        ylabel = 'Cache Hit Ratio (%)'
        # print(X)
        # print(Y)
        plot_line(myparams, X, Y, ds, labels, xlabel, ylabel, x_ticks, y_ticks,
                  (x_ticks[0], x_ticks[-1]), y_lim, pdf_file)

        create_dir('./cache_txt')
        for i, (x, y) in enumerate(zip(X, Y)):
            with open(f'./cache_txt/{ds}-{labels[i]}.txt', 'w') as f:
                for a, b in zip(x, y):
                    f.write(str(a) + ' ' + str(b) + '\n')

        # plot_line(tmp, Y, ['random', 'degree', 'sample'], savefile=, x_ticks=x_ticks, x_name=x_name, y_ticks=y_ticks, y_name=y_name, x_label=, y_label='')
