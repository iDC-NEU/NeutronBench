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
              ds,
              markevery,
              figpath=None):

    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # plt.style.use("seaborn-deep")
    # plt.style.use("grayscale")
    plt.style.use("classic")
    # plt.style.use("bmh")
    # plt.style.use("ggplot")
    pylab.rcParams.update(plot_params)  #更新自己的设置

    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
    # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
    makrer_list = [
        'D', 's', 'v', 'p', '*', 'd', 'X', 'D', 'o', 's', 'v', 'p', '*', 'd',
        'X', 'D'
    ]
    #   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
    marker_every = [5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10]
    # fig1 = plt.figure(1)
    color_list = ['b', 'g', 'k', 'c', 'm', 'y', 'r']
    color_list = [
        '#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44', '#2a6ca6',
        '#419136', '#f47a2d', '#c4342b', '#7c4e44'
    ]

    axes1 = plt.subplot(111)  #figure1的子图1为axes1
    for i, (x, y) in enumerate(zip(X, Y)):

        # plt.plot(x, y, label = labels[i], color=color_list[i], marker=makrer_list[i], markersize=8,markevery=100)
        plt.plot(x, y, label=labels[i], color=color_list[i])
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
    axes1.grid(
        axis='y',
        linestyle='',
    )

    figpath = './line.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
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
                nums = re.findall(r"\d+\.?\d*",
                                  line[line.find(mode) + len(mode):])
                ret.append(float(nums[0]))
                # if mode == 'gpu_compute_usage0':
                #    print(nums, nums[0])
    return ret


def print_diff_cache_ratio(datasets, log_path):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for ds in datasets:
        for mode in ['zerocopy', 'pipeline']:
            log_file = f'{log_path}/{ds}-{mode}-usage.log'
            print(log_file)

            cpu_usage = parse_num(log_file, 'cpu_usage')
            cpu_meme_usage = parse_num(log_file, 'cpu_meme_usage')
            gpu_mem_usage0 = parse_num(log_file, 'gpu_mem_usage0')
            gpu_compute_usage0 = parse_num(log_file, 'gpu_compute_usage0')
            # print(ds, mode, len(gpu_compute_usage0), gpu_compute_usage0[-20:])
            gpu_time = parse_num(log_file, 'GPU_info: curr_time')
            cpu_time = parse_num(log_file, 'CPU_info: curr_time')
            print(f'{log_path}/{ds}-{mode}.log')
            start_time = parse_num(f'{log_path}/{ds}-{mode}.log',
                                   'gcn_start_run_at')

            ret[ds + mode + 'cpu'] = cpu_usage
            ret[ds + mode + 'cpumem'] = cpu_meme_usage
            ret[ds + mode + 'gpu'] = gpu_compute_usage0
            ret[ds + mode + 'gpumem'] = gpu_mem_usage0
            # print(cpu_time)
            # print(start_time)
            ret[ds + mode +
                'cputime'] = np.array(cpu_time) - np.array(start_time)
            ret[ds + mode +
                'gputime'] = np.array(gpu_time) - np.array(start_time)
            ret[ds + mode + 'start'] = start_time
    return ret


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


if __name__ == '__main__':

    myparams = {
        'axes.labelsize': '18',
        'xtick.labelsize': '18',
        'ytick.labelsize': '18',
        # 'font.family': 'Times New Roman',
        'figure.figsize': '5, 4',  #图片尺寸
        'lines.linewidth': 1,
        'legend.fontsize': '13',
        'legend.loc': 'best',  #[]"upper right", "upper left"]
        'legend.numpoints': 1,
        # 'lines.ncol': 2,
    }

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

    run_times = {
        'reddit': 50,
        'ogbn-arxiv': 5,
        'ogbn-products': 300,
        'AmazonCoBuy_computers': 50,
        'AmazonCoBuy_photo': 25,
    }

    datasets = [
        'hollywood-2011', 'lj-links', 'reddit', 'lj-links', 'enwiki-links',
        'ogbn-arxiv', 'livejournal', 'ogbn-products'
    ]
    datasets = [
        'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links',
        'ogbn-arxiv', 'ogbn-products'
    ]
    datasets = [
        'reddit', 'lj-links', 'enwiki-links', 'ogbn-arxiv', 'ogbn-products',
        'hollywood-2011'
    ]
    datasets = ['road-usa']
    datasets = ['amazon']
    datasets = ['ogbn-products', 'reddit', 'ogbn-arxiv']
    datasets = ['lj-links', 'ogbn-arxiv']
    datasets = ['lj-large']
    datasets = [
        'ogbn-arxiv', 'ogbn-products', 'reddit', 'livejournal', 'lj-large',
        'hollywood-2011', 'lj-links', 'enwiki-links'
    ]
    datasets = [
        'ogbn-arxiv', 'ogbn-products', 'reddit', 'livejournal', 'lj-large',
        'hollywood-2011'
    ]
    datasets = ['reddit']
    datasets = ['ogbn-arxiv']

    ret = print_diff_cache_ratio(datasets, './log')
    for k, v in ret.items():
        print(k, len(v))

    for ds in datasets:
        X, Y = [], []
        labels = ['zerocopy', 'pipeline']
        for mode in ['zerocopy', 'pipeline']:
            X.append(ret[ds + mode + 'cputime'])
            Y.append(ret[ds + mode + 'cpu'])
            print(len(X[-1]), len(Y[-1]))
        X, Y = split_list(X, Y, 0.2)

        x_ticks = np.linspace(0, run_times[ds], 6)
        y_ticks = np.linspace(0, 100, 6)
        y_lim = (0, 70)

        pdf_file = f'./usage/{ds}-cpu.pdf'
        xlabel = 'Time (s)'
        ylabel = 'CPU Usage (%)'
        plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks, y_ticks,
                  (x_ticks[0], x_ticks[-1]), y_lim, ds, mark_list, pdf_file)

        X, Y = [], []
        labels = ['zerocopy', 'pipeline']
        for mode in ['zerocopy', 'pipeline']:
            X.append(ret[ds + mode + 'gputime'])
            Y.append(ret[ds + mode + 'gpu'])
        X, Y = split_list(X, Y, 0.2)

        x_ticks = np.linspace(0, run_times[ds], 6)
        y_ticks = np.linspace(0, 100, 6)
        y_lim = (0, 100)

        pdf_file = f'./usage/{ds}-gpu.pdf'
        xlabel = 'Time (s)'
        ylabel = 'GPU Usage (%)'
        plot_line(myparams, X, Y, labels, xlabel, ylabel, x_ticks, y_ticks,
                  (x_ticks[0], x_ticks[-1]), y_lim, ds, mark_list, pdf_file)
