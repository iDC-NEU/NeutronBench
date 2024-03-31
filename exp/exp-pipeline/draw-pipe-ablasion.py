import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import numpy as np
import matplotlib.ticker as mtick


def plot_bar(plot_params,
             Y,
             labels,
             xlabel,
             ylabel,
             xticks,
             yticks,
             color_list,
             anchor=None,
             figpath=None):

    plt.rcParams.update(plt.rcParamsDefault)
    # print(plt.rcParams.keys())

    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # print(plt.style.available)
    # plt.style.use('classic')
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    plt.style.use("seaborn-deep")
    # plt.style.use("seaborn-paper")
    # plt.style.use("seaborn-notebook")
    # plt.style.use("seaborn-poster")
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = 0.13

    n = len(Y[0])
    ind = np.arange(n)  # the x locations for the groups
    # ind = [0,1.2,2.4,3.6]                # the x locations for the groups
    # ind = [0,1.3,2.6,3.9]                # the x locations for the groups
    m = len(labels)
    offset = np.arange(m) - m / 2 + 0.5

    for i, y in enumerate(Y):
        # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])
        plt.bar(ind + (offset[i] * width),
                y,
                width,
                color=color_list[i],
                label=labels[i],
                linewidth=plot_params['lines.linewidth'],
                edgecolor='black')

    # plt.xticks(np.arange(n) + (len(labels)/2-0.5)*width, xticks)
    plt.xticks(ind, xticks)
    plt.yticks(yticks, yticks)

    # plt.legend(ncol=len(labels)//2, bbox_to_anchor=anchor)
    # plt.legend(nrow=len(labels)//2, bbox_to_anchor=anchor)

    plt.legend(ncol=3,
               bbox_to_anchor=anchor,
               columnspacing=1.2,
               handletextpad=.35,
               labelspacing=.2,
               handlelength=1.5)  # ,markerscale=10

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=2)

    # Set the formatter
    axes = plt.gca()  # get current axes

    axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['left'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['right'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['top'].set_linewidth(plot_params['lines.linewidth'])

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')
    print(figpath, 'is plot.')
    plt.close()


def plot_bar_balance1(plot_params,
                      Y,
                      labels,
                      xlabel,
                      ylabel,
                      xticks,
                      yticks,
                      color_list,
                      anchor=None,
                      figpath=None):

    plt.rcParams.update(plt.rcParamsDefault)
    # print(plt.rcParams.keys())

    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # print(plt.style.available)
    # plt.style.use('classic')
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    plt.style.use("seaborn-deep")
    # plt.style.use("seaborn-paper")
    # plt.style.use("seaborn-notebook")
    # plt.style.use("seaborn-poster")
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = 0.2
    # color_list = ['#2a6ca6', '#419136', '#7c4e44', '#c4342b', '#f47a2d', '#EA6632', '#f47a2d', ]
    # color_list = ['#b35806','#f1a340','#fee0b6','#d8daeb','#998ec3','#542788']

    n = len(Y[0])
    ind = np.arange(n)  # the x locations for the groups
    # ind = [0,1.2,2.4,3.6]                # the x locations for the groups
    # ind = [0,1.3,2.6,3.9]                # the x locations for the groups
    m = len(labels)
    offset = np.arange(m) - m / 2 + 0.5

    # for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])
    # plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i], linewidth=params['lines.linewidth'], edgecolor='black')

    labels_p = [
        'part1',
        'part2',
        'part3',
    ]
    for i, y in enumerate(Y):
        off = np.linspace(0, len(ind) * width, len(ind) + 1)[:-1] - 1.5 * width
        off = off + np.ones(len(ind)) * i
        # print(i,ind,  off)
        plt.bar(off,
                y,
                width,
                color=color_list[i],
                label=labels[i],
                linewidth=params['lines.linewidth'],
                edgecolor='black')
        # plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i], linewidth=params['lines.linewidth'], edgecolor='black')

    # plt.xticks(np.arange(n) + (len(labels)/2-0.5)*width, xticks)
    plt.xticks(list(range(0, len(labels))), labels, rotation=25)
    plt.yticks(yticks, yticks)

    # plt.legend(ncol=len(labels)//2, bbox_to_anchor=anchor)
    # plt.legend(nrow=len(labels)//2, bbox_to_anchor=anchor)

    plt.legend(ncol=3,
               bbox_to_anchor=anchor,
               columnspacing=1.2,
               handletextpad=.35,
               labelspacing=.2,
               handlelength=1.5)  # ,markerscale=10

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=2)

    axes = plt.gca()
    axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['left'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['right'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['top'].set_linewidth(plot_params['lines.linewidth'])

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
    print(figpath, 'is plot.')
    plt.close()


def plot_bar_balance3(plot_params, labels, anchor=None, figpath=None):
    plt.rcParams.update(plt.rcParamsDefault)
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = 0.25
    color_list = ['C3', 'C1', 'C2', 'C0', 'C4']
    hatch_list = ['xx', '..', '**', '++', '--']

    def plot_one(pltnum, Y, yticks, ylim, title):
        ax1 = plt.subplot(pltnum)
        ax1.set_ylim(*ylim)
        ax1.set_title(title, x=.5, y=-.3)

        ax1.set_xticks([])
        ax1.set_yticks(yticks)
        ax1.set_ylabel('Epoch time (s)', labelpad=1)
        print(Y)
        h_legs, e_legs = [], []
        for i, y in enumerate(Y):
            leg1 = ax1.bar(i,
                           y,
                           width,
                           color=color_list[i],
                           hatch=hatch_list[i],
                           label=labels[i],
                           edgecolor='white')
            leg2 = ax1.bar(i,
                           y,
                           width,
                           color='none',
                           lw=plot_params['lines.linewidth'],
                           edgecolor='black')
            ax1.text(i,
                     y,
                     round(y, 1),
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     fontsize=8)
            h_legs.append(leg1)
            e_legs.append(leg2)

        return h_legs, e_legs

    plot_one(131, [3.902, 2.378, 2.259], [0, 2, 4], (0, 5), '(a) Products')
    plot_one(132, [95.019, 84.459, 70.798], [0, 50, 100], (0, 120),
             '(b) Livejournal')
    h_legs, e_legs = plot_one(133, [107.779, 96.550, 80.333], [0, 50, 100],
                              (0, 120), '(c) Lj-links')

    legs = [(x, y) for x, y in zip(h_legs, e_legs)]
    plt.legend(legs,
               labels,
               ncol=3,
               bbox_to_anchor=(-.8, 1.2),
               columnspacing=3,
               handletextpad=.2,
               labelspacing=.1,
               handlelength=1)
    plt.subplots_adjust(wspace=.35, hspace=0)  #调整子图间距

    axes = plt.gca()  # get current axes
    axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['left'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['right'].set_linewidth(plot_params['lines.linewidth'])
    axes.spines['top'].set_linewidth(plot_params['lines.linewidth'])

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')
    print(figpath, 'is plot.\n')
    plt.close()


def plot_bar_balance5(plot_params, labels, anchor=None, figpath=None):
    plt.rcParams.update(plt.rcParamsDefault)
    # print(plt.rcParams.keys())
    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # print(plt.style.available)
    # plt.style.use('classic')
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    # plt.style.use("seaborn-deep")
    # plt.style.use("seaborn-paper")
    # plt.style.use("seaborn-notebook")
    # plt.style.use("seaborn-poster")
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = 0.25
    color_list = ['C3', 'C1', 'C2', 'C0', 'C4']
    hatch_list = ['xx', '..', '**', '++', '--']

    def plot_one(pltnum,
                 Y,
                 yticks,
                 ylim,
                 title,
                 color='black',
                 pos=(.5, -.49)):
        ax1 = plt.subplot(pltnum)
        ax1.set_ylim(*ylim)
        if pos is None:
            ax1.set_title(title, x=.5, y=-.3, color=color, fontsize=14)
        else:
            ax1.set_title(title, x=pos[0], y=pos[1], color=color, fontsize=14)

        ax1.set_xticks([])
        ax1.set_yticks(yticks)
        ax1.set_ylabel('Epoch time (s)', labelpad=1)
        print(Y)
        h_legs, e_legs = [], []
        for i, y in enumerate(Y):
            leg1 = ax1.bar(i,
                           y,
                           width,
                           color=color_list[i],
                           hatch=hatch_list[i],
                           label=labels[i],
                           edgecolor='white')
            leg2 = ax1.bar(i,
                           y,
                           width,
                           color='none',
                           lw=plot_params['lines.linewidth'],
                           edgecolor='black')
            ax1.text(i,
                     y,
                     round(y, 1),
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     fontsize=8)
            h_legs.append(leg1)
            e_legs.append(leg2)

        return h_legs, e_legs

    # plot_one(231, [4.146, 3.535, 3.190], [0, 2, 4], (0, 5), '(a) Reddit\n(high feature dimension)')
    # plot_one(232, [3.902, 2.378, 2.259], [0, 2, 4], (0, 5), '(b) Products\n(low feature dimension)', 'blue', (.5,-.49))
    # plot_one(233, [95.019, 84.459, 70.798], [0, 50, 100], (0, 120), '(c) Livejournal\n(high feature dimension)', 'blue', (.5,-.49))
    # h_legs, e_legs = plot_one(234, [107.779, 96.550, 80.333], [0, 50, 100], (0, 120), '(d) Lj-links\n(high feature dimension)', 'blue', (.5,-.49))
    # plot_one(235, [91.653, 79.619, 74.445], [0, 50, 100], (0, 110), '(e) Lj-large\n(high feature dimension)')
    # # plot_one(236, [321.128, 278.740, 279.014], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')
    # plot_one(236, [321.128, 278.740, 245.580], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')

    plot_one(231, [3.902, 2.378, 2.259], [0, 2, 4], (0, 5),
             '(a) Products\n(low feature dimension)', 'blue', (.5, -.49))
    plot_one(232, [95.019, 84.459, 70.798], [0, 50, 100], (0, 120),
             '(b) Livejournal\n(high feature dimension)', 'blue', (.5, -.49))
    h_legs, e_legs = plot_one(233, [107.779, 96.550, 80.333], [0, 50, 100],
                              (0, 120),
                              '(c) Lj-links\n(high feature dimension)', 'blue',
                              (.5, -.49))
    plot_one(234, [4.146, 3.535, 3.190], [0, 2, 4], (0, 5),
             '(d) Reddit\n(high feature dimension)')
    plot_one(235, [91.653, 79.619, 74.445], [0, 50, 100], (0, 110),
             '(e) Lj-large\n(high feature dimension)')
    # plot_one(236, [321.128, 278.740, 279.014], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')
    plot_one(236, [321.128, 278.740, 245.580], [0, 150, 300], (0, 370),
             '(f) Enwiki-links\n(high feature dimension)')

    legs = [(x, y) for x, y in zip(h_legs, e_legs)]
    plt.legend(legs,
               labels,
               ncol=3,
               bbox_to_anchor=(-.9, 2.7),
               columnspacing=3,
               handletextpad=.2,
               labelspacing=.1,
               handlelength=1)
    plt.subplots_adjust(wspace=.35, hspace=.5)  #调整子图间距
    # order = [0,2,1]
    #add legend to plot
    # plt.legend([lines[idx] for idx in order], [labels[idx] for idx in order] , ncol=3, bbox_to_anchor=anchor, columnspacing=.5, handletextpad=.2, labelspacing=.1, handlelength=1)

    # plt.xlabel(xlabel)
    axes = plt.gca()  # get current axes
    axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth'])
    ###设置底部坐标轴的粗细
    axes.spines['left'].set_linewidth(plot_params['lines.linewidth'])
    ####设置左边坐标轴的粗细
    axes.spines['right'].set_linewidth(plot_params['lines.linewidth'])
    ###设置右边坐标轴的粗细
    axes.spines['top'].set_linewidth(plot_params['lines.linewidth'])
    ####设置上部坐标轴的粗细

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
    print(figpath, 'is plot.\n')
    plt.close()


def plot_bar_balance3blue(plot_params, labels, anchor=None, figpath=None):
    plt.rcParams.update(plt.rcParamsDefault)
    # print(plt.rcParams.keys())
    # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # print(plt.style.available)
    # plt.style.use('classic')
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    # plt.style.use("seaborn-deep")
    # plt.style.use("seaborn-paper")
    # plt.style.use("seaborn-notebook")
    # plt.style.use("seaborn-poster")
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = 0.25
    color_list = ['C3', 'C1', 'C2', 'C0', 'C4']
    hatch_list = ['xx', '..', '**', '++', '--']

    def plot_one(pltnum,
                 Y,
                 yticks,
                 ylim,
                 title,
                 color='black',
                 pos=(.5, -.49)):
        ax1 = plt.subplot(pltnum)
        ax1.set_ylim(*ylim)
        if pos is None:
            ax1.set_title(title, x=.5, y=-.3, color=color, fontsize=11)
        else:
            ax1.set_title(title, x=pos[0], y=pos[1], color=color, fontsize=11)

        ax1.set_xticks([])
        ax1.set_yticks(yticks)
        ax1.set_ylabel('Epoch time (s)', labelpad=1)
        print(Y)
        h_legs, e_legs = [], []
        for i, y in enumerate(Y):
            leg1 = ax1.bar(i,
                           y,
                           width,
                           color=color_list[i],
                           hatch=hatch_list[i],
                           label=labels[i],
                           edgecolor='white')
            leg2 = ax1.bar(i,
                           y,
                           width,
                           color='none',
                           lw=plot_params['lines.linewidth'],
                           edgecolor='black')
            ax1.text(i,
                     y,
                     round(y, 1),
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     fontsize=8)
            h_legs.append(leg1)
            e_legs.append(leg2)

        return h_legs, e_legs

    # plot_one(231, [4.146, 3.535, 3.190], [0, 2, 4], (0, 5), '(a) Reddit\n(high feature dimension)')
    # plot_one(232, [3.902, 2.378, 2.259], [0, 2, 4], (0, 5), '(b) Products\n(low feature dimension)', 'blue', (.5,-.49))
    # plot_one(233, [95.019, 84.459, 70.798], [0, 50, 100], (0, 120), '(c) Livejournal\n(high feature dimension)', 'blue', (.5,-.49))
    # h_legs, e_legs = plot_one(234, [107.779, 96.550, 80.333], [0, 50, 100], (0, 120), '(d) Lj-links\n(high feature dimension)', 'blue', (.5,-.49))
    # plot_one(235, [91.653, 79.619, 74.445], [0, 50, 100], (0, 110), '(e) Lj-large\n(high feature dimension)')
    # # plot_one(236, [321.128, 278.740, 279.014], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')
    # plot_one(236, [321.128, 278.740, 245.580], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')

    plot_one(131, [3.902, 2.378, 2.259], [0, 2, 4], (0, 5),
             '(a) Products\n(low feature dimension)', 'black', (.5, -.42))
    plot_one(132, [95.019, 84.459, 70.798], [0, 50, 100], (0, 120),
             '(b) Livejournal\n(high feature dimension)', 'black', (.5, -.42))
    h_legs, e_legs = plot_one(133, [107.779, 96.550, 80.333], [0, 50, 100],
                              (0, 120),
                              '(c) Lj-links\n(high feature dimension)',
                              'black', (.5, -.42))
    # plot_one(234, [4.146, 3.535, 3.190], [0, 2, 4], (0, 5), '(d) Reddit\n(high feature dimension)')
    # plot_one(235, [91.653, 79.619, 74.445], [0, 50, 100], (0, 110), '(e) Lj-large\n(high feature dimension)')
    # # plot_one(236, [321.128, 278.740, 279.014], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')
    # plot_one(236, [321.128, 278.740, 245.580], [0, 150, 300], (0, 370), '(f) Enwiki-links\n(high feature dimension)')

    legs = [(x, y) for x, y in zip(h_legs, e_legs)]
    plt.legend(legs,
               labels,
               ncol=3,
               bbox_to_anchor=anchor,
               columnspacing=3,
               handletextpad=.2,
               labelspacing=.1,
               handlelength=1)
    plt.subplots_adjust(wspace=.35, hspace=.5)  #调整子图间距
    # order = [0,2,1]
    #add legend to plot
    # plt.legend([lines[idx] for idx in order], [labels[idx] for idx in order] , ncol=3, bbox_to_anchor=anchor, columnspacing=.5, handletextpad=.2, labelspacing=.1, handlelength=1)

    # plt.xlabel(xlabel)
    axes = plt.gca()  # get current axes
    axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth'])
    ###设置底部坐标轴的粗细
    axes.spines['left'].set_linewidth(plot_params['lines.linewidth'])
    ####设置左边坐标轴的粗细
    axes.spines['right'].set_linewidth(plot_params['lines.linewidth'])
    ###设置右边坐标轴的粗细
    axes.spines['top'].set_linewidth(plot_params['lines.linewidth'])
    ####设置上部坐标轴的粗细

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight',
                format='pdf')  #bbox_inches='tight'会裁掉多余的白边
    print(figpath, 'is plot.\n')
    plt.close()


def create_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def plot_comp_comm(dataset, num_parts):
    params = {
        'axes.labelsize': '10',
        'xtick.labelsize': '9',
        'ytick.labelsize': '10',
        'lines.linewidth': 1,
        # 'axes.linewidth': 10,
        # 'bars.linewidth': 100,
        'legend.fontsize': '13',
        'figure.figsize': '8, 1.5',
        'legend.loc': 'center',  #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        'font.family': 'Arial',
        'font.serif': 'Arial',
    }

    # modes = ['metis1', 'metis2', 'metis4', 'dgl', 'pagraph', 'bytegnn', 'hash']
    # labels = ['metis1', 'metis2', 'metis4', 'dgl', 'pagraph', 'bytegnn', 'hash']
    modes = ['hash', 'metis1', 'dgl', 'metis4', 'bytegnn']
    labels = ['Hash', 'Metis*', 'DistDGL', 'SALIENT++', 'ByteGNN']
    labels = ['Hash', 'Metis-V', 'Metis-VE', 'Metis-VET', 'Stream-B']
    labels = ['No pipe', 'Pipeline BP', 'Pipeline-BP-DT']
    # for ds in datasets:
    #   color_list = ['#bdddf2','#8e8e8e','#f3ec8a','#bfd2bb','#d394de','#b0dbce',]
    #   xlabel = '# partition ID'
    #   # ylabel = 'Communication load (GB)'
    #   # ylabel = 'Graph structure and features\n of communication (GB)'
    #   ylabel = 'Epoch time (s)', 'Total Communication (GB)'

    #   xticks = [f'{x+1}' for x in range(num_parts)]
    #   if ds == 'reddit':
    #     yticks = [[0, 3, 6, 9],[0, 1, 2, 3]]
    #     ylim = [(0, 8),(0, 3)]
    #     all_comm_load = [[4.146, 3.535, 3.190]]
    #   # elif ds == 'ogbn-products':
    #   #   yticks = [[0, 3, 6],[0, 12, 24, 36]]
    #   #   ylim = [(0, 8),(0, 43)]
    #   #   all_comm_load = [[5.1903,2.7366,2.8637,2.8210,3.9445]]
    #   elif ds == 'ogbn-arxiv':
    #     yticks = [[0, .1, .2, .3],[0, .1, .2, .3]]
    #     ylim = [(0, .3),(0, .3)]
    #     all_comm_load = [[0.374, 0.252, 0.222]]
    #   elif ds == 'livejournal':
    #     yticks = [[0, .1, .2, .3],[0, .1, .2, .3]]
    #     ylim = [(0, .3),(0, .3)]
    #     all_comm_load = [[95.019, 84.459, 70.798]]
    #   elif ds == 'lj-links':
    #     yticks = [[0, .1, .2, .3],[0, .1, .2, .3]]
    #     ylim = [(0, .3),(0, .3)]
    #     all_comm_load = [[107.779, 96.550, 80.333]]
    #   elif ds == 'lj-large':
    #     yticks = [[0, .1, .2, .3],[0, .1, .2, .3]]
    #     ylim = [(0, .3),(0, .3)]
    #     all_comm_load = [[91.653, 79.619, 74.445]]
    #   elif ds == 'enwiki-links':
    #     yticks = [[0, .1, .2, .3],[0, .1, .2, .3]]
    #     ylim = [(0, .3),(0, .3)]
    #     all_comm_load = [[321.128, 278.740, 279.014]]

    plot_bar_balance3(params,
                      labels,
                      anchor=(-0.7, 1.18),
                      figpath=f'./pipeline/ablation3.pdf')

    params = {
        'axes.labelsize': '10',
        'xtick.labelsize': '9',
        'ytick.labelsize': '10',
        'lines.linewidth': 1,
        # 'axes.linewidth': 10,
        # 'bars.linewidth': 100,
        'legend.fontsize': '13',
        'figure.figsize': '12, 3',
        'legend.loc': 'center',  #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        'font.family': 'Arial',
        'font.serif': 'Arial',
    }
    plot_bar_balance5(params,
                      labels,
                      anchor=(-0.7, 4.18),
                      figpath=f'./pipeline/ablation6.pdf')

    params = {
        'axes.labelsize': '11',
        'xtick.labelsize': '11',
        'ytick.labelsize': '11',
        'lines.linewidth': 1,
        # 'axes.linewidth': 11,
        # 'bars.linewidth': 110,
        'legend.fontsize': '11',
        'figure.figsize': '8, 1.5',
        'legend.loc': 'center',  #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        'font.family': 'Arial',
        'font.serif': 'Arial',
    }
    plot_bar_balance3blue(params,
                          labels,
                          anchor=(-0.9, 1.1),
                          figpath=f'./pipeline/pipeline-ablation.pdf')


# dataset
# No pipeline
# Pipelien S
# Pipelien ST

# arxiv [0.374, 0.252, 0.222]
# reddit [4.146, 3.535, 3.190]
# livejournal [95.019, 84.459, 70.798]
# lj-links, [107.779, 96.550, 80.333]
# lj-large, [91.653, 79.619, 74.445]
# enwiki-links [321.128, 278.740, 279.014]

if __name__ == '__main__':
    # print(plt.rcParams.keys())
    feat_dims = {
        'amazon': 200,
        'reddit': 200,
        'ogbn-arxiv': 128,
        'ogbn-products': 602,
    }

    datasets = ['amazon']
    datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit', 'computer']
    datasets = ['amazon', 'reddit']
    datasets = ['amazon', 'ogbn-products']
    datasets = ['reddit', 'amazon', 'ogbn-products']
    datasets = [
        'reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links'
    ]

    num_parts = 4
    # dataset
    # No pipeline
    # Pipelien S
    # Pipelien ST

    # plot_data_access(datasets, num_parts)
    plot_comp_comm(datasets, 3)
