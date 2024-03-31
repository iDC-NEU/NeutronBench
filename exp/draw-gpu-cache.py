# -*- coding: utf-8 -*-

from ast import parse
from cProfile import label
from calendar import c
from time import time
from tkinter.messagebox import NO
import numpy as np
import matplotlib
import os, re
import itertools
import math
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


def plot_line(X,
              Y,
              labels,
              savefile=None,
              color=None,
              x_label=None,
              y_label=None,
              show=False,
              x_ticks=None,
              x_name=None,
              loc=None,
              y_ticks=None,
              y_names=None,
              high_mark='.',
              ylim=None,
              draw_small=False,
              xscale=None):
    print(f'plot_line of {savefile}\n')
    # print(len(X), len(Y), len(labels))
    assert (len(X) == len(Y) == len(labels))
    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    # linestyle = "-"
    ax.grid(linestyle="-.")  # 设置背景网格线为虚线
    # ax = ax.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框

    linewidth = 1.0
    markersize = 7

    if color is None:
        # color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue', 'red', 'cyan', 'magenta', 'yellow', 'black']
        color = [
            'blue', 'green', 'orange', 'purple', 'red', 'black', 'yellow',
            'cyan', 'magenta', 'pink', 'deepskyblue', 'tomato'
        ]

    for i in range(len(X)):
        if len(X[i]) == 0:
            continue
        ax.plot(X[i],
                Y[i],
                marker='',
                markersize=markersize,
                color=color[i],
                alpha=1,
                label=labels[i],
                linewidth=linewidth)
        # ax.plot(X[i], Y[i], marker='', markersize=markersize, alpha=1, label=labels[i], linewidth=linewidth)
        pos = np.where(np.amax(Y[i]) == Y[i])[0].tolist()
        pos = pos[0]
        # print(pos)
        # print(Y[i][pos[0]], Y[i][pos[1]])

        ax.plot(X[i][pos],
                Y[i][pos],
                marker='x',
                markersize=markersize,
                color='red',
                alpha=1,
                linewidth=linewidth)
        # ax.plot(X[i][pos], Y[i][pos], marker=high_mark, markersize=markersize-2, alpha=1, linewidth=linewidth)

    if x_ticks or x_name:
        pass
    else:
        max_xticks = max(max(x) if len(x) > 0 else 0 for x in X)
        x_ticks = np.linspace(0, max_xticks, 5).tolist()
        x_name = [f'{x:.2f}' for x in x_ticks]
        ax.xticks(x_ticks, x_name, fontsize=15)  # 默认字体大小为10

    # ax.xlim(0, max_ticks + 0.1)  # 设置x轴的范围
    max_yticks = max(max(x) if len(x) > 0 else 0 for x in Y)
    min_yticks = 0
    if ylim is not None:
        min_yticks = ylim

    y_ticks = np.linspace(min_yticks, max_yticks, 5).tolist()
    y_name = [f'{x:.2f}' for x in y_ticks]

    ax.set_ylim(min_yticks, max_yticks)
    ax.set_yticks(y_ticks, y_name, fontsize=15)  # 默认字体大小为10

    # if not y_ticks:
    #   y_ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
    #   # ax.ylim(0, 1)
    # if not y_names:
    #   y_names = ['10%', '30%', '50%', '70%', '90%']

    ax.set_ylabel(y_label, fontsize=15)
    ax.set_xlabel(x_label, fontsize=15)
    # ax.xlim(0, np.max(X) + 1)  # 设置x轴的范围

    if xscale is not None:
        ax.set_xscale('log', base=xscale)

    # ax.legend()
    # 显示各曲线的图例 loc=3 lower left
    if not loc:
        loc = 1
    ax.legend(loc=loc, numpoints=1, ncol=1)
    # leg = ax.gca().get_legend()
    # ltext = leg.get_texts()
    # ax.setp(ltext, fontsize=15)
    # ax.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
    # ax.tight_layout()

    ######################## small #########################
    if draw_small is True:
        axins = inset_axes(ax,
                           width="40%",
                           height="30%",
                           loc='lower left',
                           bbox_to_anchor=(0.2, 0.05, 1, 1),
                           bbox_transform=ax.transAxes)
        for i, (x, y) in enumerate(zip(X, Y)):
            axins.plot(x,
                       y,
                       color=color[i],
                       linewidth=linewidth,
                       marker='',
                       markersize=5)
            # markeredgecolor=color[i], markerfacecolor=color[i])
        # # pubmed
        # xlim0, xlim1 = 0, 0.5
        # ylim0, ylim1 = 0.83, 0.87

        # phpto
        # xlim0, xlim1 = 0, 0.5
        # ylim0, ylim1 = 0.3, 0.8

        # computer
        xlim0, xlim1 = 0, 0.5
        ylim0, ylim1 = 0.3, 0.8
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        # axins.set_xticks([])
        # axins.set_yticks([])

        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    ######################## small #########################

    if not savefile:
        savefile = 'plot_line.png'
    fig.savefig(f'{savefile}',
                format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    if show:
        plt.show()
    plt.close()


def get_acc_time_list(accs, times, best=None):
    early_stop = False if not best else True

    if not isinstance(times, list):
        times = [times for _ in range(len(accs))]

    idx = len(accs)
    if early_stop:
        idx = 0
        while accs[idx] < best:
            idx += 1
    # idx = bisect.bisect(accs, best)
    idx = min(idx + 10, len(accs))
    accs_ret = accs[:idx + 1]
    times_ret = list(itertools.accumulate(times[:idx + 1]))
    assert len(accs_ret) == len(times_ret)
    return [accs_ret, times_ret]


def parse_log(filename=None):
    assert filename
    if not os.path.exists(filename):
        print(f'{filename} not exist')
    train_acc = []
    val_acc = []
    test_acc = []
    avg_time_list = []
    time_cost = dict()
    # avg_train_time = None
    # avg_val_time = None
    # avg_test_time = None
    dataset = None
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            # print(line)
            if line.find('Epoch ') >= 0:
                nums = re.findall(r"\d+\.?\d*", line)
                # print(nums)
                train_acc.append(float(nums[1]))
                val_acc.append(float(nums[2]))
                test_acc.append(float(nums[3]))
            elif line.find('edge_file') >= 0:
                l, r = line.rfind('/'), line.rfind('.')
                dataset = line[l + 1:r]
            elif line.find('Avg') >= 0:
                nums = re.findall(r"\d+\.?\d*", line)
                avg_time_list.append(float(nums[0]))
                avg_time_list.append(float(nums[1]))
                avg_time_list.append(float(nums[2]))
            elif line.find('TIME') >= 0:
                nums = re.findall(r"\d+\.?\d*", line)
                time_cost[int(nums[0])] = [float(x) for x in nums[1:]]
                # TIME(0) sample 0.000 compute_time 2.977 comm_time 0.003 mpi_comm 0.302 rpc_comm 0.000 rpc_wait_time 2.675
    return dataset, [train_acc, val_acc, test_acc], avg_time_list, time_cost


def parse_acc_with_epoch(filename, epoch):
    print(f'parse_acc_with_epoch of {filename}')
    if not os.path.exists(filename):
        print(f'{filename} not exist')
    loss, train_acc, val_acc, test_acc = [], [], [], []
    with open(filename) as f:
        for line in f.readlines():
            if line.find('Epoch ') >= 0:
                nums = re.findall(r"\d+\.?\d*", line)
                loss.append(float(nums[1]))
                train_acc.append(float(nums[2]))
                val_acc.append(float(nums[3]))
                test_acc.append(float(nums[4]))
    runs = len(train_acc) // epoch
    # print(runs * epoch, len(train_acc), epoch)
    # assert(runs * epoch == len(train_acc))
    # train_acc = sum(np.array_split(train_acc, runs)) / runs
    # val_acc = sum(np.array_split(val_acc, runs)) / runs
    # test_acc = sum(np.array_split(test_acc, runs)) / runs
    # return [train_acc, val_acc, test_acc]
    return [train_acc[:epoch], val_acc[:epoch], test_acc[:epoch]]


def parse_mode(filename, mode='eval_acc'):
    print(f'parse_acc of {filename}, mode {mode}')
    if not os.path.exists(filename):
        print(f'{filename} not exist')
    ret = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find(mode) >= 0:
                nums = re.findall(r"\d+\.?\d*", line[line.find(mode):])
                ret.append(float(nums[0]))
    return ret


def parse_time(filename):
    print(f'parse_time of {filename}')
    if not os.path.exists(filename):
        print(f'{filename} not exist')
    time_list = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find('Avg') >= 0:
                nums = re.findall(r"\d+\.?\d*", line)
                time_list.append(float(nums[0]))
                time_list.append(float(nums[1]))
                time_list.append(float(nums[2]))
    runs = len(time_list) // 3
    assert (runs * 3 == len(time_list))
    time_list = sum(np.array_split(time_list, runs)) / runs
    # print(time_list)
    return time_list


def parse_best_acc(filename):
    print(f'parse_best_acc of {filename}')
    if not os.path.exists(filename):
        print(f'{filename} not exist')

    with open(filename) as f:
        best_acc_mean, best_acc_var = 0, 0
        for line in f.readlines():
            if line.find('Val-mean-var ') >= 0:
                nums = re.findall(r"\d+\.?\d*", line)
                best_acc_mean = nums[1]
                best_acc_var = nums[2]
    return (best_acc_mean, best_acc_var)


def parse_all_time(filename):
    print(f'parse_time of {filename}')
    if not os.path.exists(filename):
        print(f'{filename} not exist')
    train_time_list = []
    eval_time_list = []
    test_time_list = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find('train_epoch') >= 0:
                line = line[line.find('train_epoch'):]
                nums = re.findall(r"\d+\.?\d*", line)
                train_time_list.append(float(nums[1]))

            if line.find('eval_epoch') >= 0:
                line = line[line.find('eval_epoch'):]
                nums = re.findall(r"\d+\.?\d*", line)
                eval_time_list.append(float(nums[1]))

            if line.find('test_epoch') >= 0:
                line = line[line.find('test_epoch'):]
                nums = re.findall(r"\d+\.?\d*", line)
                test_time_list.append(float(nums[1]))

    return [train_time_list, eval_time_list, test_time_list]


def parse_dataset(filename):
    print(f'parse_dataset of {filename}')
    if not os.path.exists(filename):
        print(f'{filename} not exist')
    dataset = None
    with open(filename) as f:
        for line in f.readlines():
            if line.find('edge_file') >= 0:
                l, r = line.rfind('/'), line.rfind('.')
                dataset = line[l + 1:r]
                break
    assert (dataset)
    return dataset


def draw_batch_size(batch_size,
                    epochs,
                    pre_path,
                    plot_path,
                    datasets,
                    pre_epoch=None,
                    pre_time=None,
                    suffix_name='.pdf'):
    # pre_path = './log/batch-size/'
    # plot_path = './pdf/batch-size/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    labels = []
    # for k, v in batch_size.items():
    for k in datasets:
        v = batch_size[k]
        plot_time, plot_acc, labels = [], [], []
        for b in v:
            file_name = k + '_' + b
            file_path = pre_path + file_name + '.log'
            if not os.path.exists(file_path):
                print(file_path, 'not exist.')
                continue
            if not epochs:
                train_accs, val_accs, test_accs = parse_acc(file_path)
            else:
                train_accs, val_accs, test_accs = parse_acc_with_epoch(
                    file_path, epochs[k])

            train_time, val_time, test_time = parse_time(file_path)
            val_acc, train_time = get_acc_time_list(val_accs, train_time)

            labels.append(file_name)

            if pre_time is not None:
                idx = 0
                while idx < len(train_time) and train_time[idx] <= pre_time[k]:
                    # while idx < len(train_time) and train_time[idx] < pre_time:
                    idx += 1
                # print(k, b, pre_time[k], idx, train_time[:idx])
                train_time = np.array(train_time)[:idx].tolist()
                val_acc = np.array(val_acc)[:idx].tolist()
                # print(train_time)
                assert (len(train_time) == len(val_acc))
                # plot_file_name = plot_path + k + f'-preT.pdf'

            plot_acc.append(val_acc)
            plot_time.append(train_time)

        if len(plot_acc) == 0:
            continue
        # plot_file_name = plot_path + k + '.pdf'
        plot_file_name = plot_path + k + suffix_name

        if pre_epoch is not None:
            assert isinstance(pre_epoch, int)
            plot_time = np.array(plot_time)[:, :pre_epoch].tolist()
            plot_acc = np.array(plot_acc)[:, :pre_epoch].tolist()
            plot_file_name = plot_path + k + f'-{pre_epoch}.pdf'

        # if pre_time is not None:
        #     plot_file_name = plot_path + k + f'-preT.pdf'
        # print(plot_time)

        # print(plot_file_name)
        # print(type(plot_time))
        # print(len(plot_acc[0]), len(plot_time[0]))
        # print(plot_time)
        plot_line(X=plot_time,
                  Y=plot_acc,
                  labels=labels,
                  savefile=plot_file_name,
                  y_label='Eval Acc')


def draw_batch_size1(batch_size,
                     epochs,
                     pre_path,
                     plot_path,
                     datasets,
                     pre_epoch=None,
                     pre_time=None):
    # pre_path = './log/batch-size/'
    # plot_path = './pdf/batch-size/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    labels = []
    # for k, v in batch_size.items():
    for k in datasets:
        v = batch_size[k]
        plot_time, plot_acc, labels = [], [], []
        for b in v:
            file_name = k + '_' + b
            file_path = pre_path + file_name + '.log'
            if not os.path.exists(file_path):
                print(file_path, 'not exist.')
                continue
            train_accs, val_accs, test_accs = parse_acc(file_path)
            train_time, val_time, test_time = parse_time(file_path)
            val_acc, train_time = get_acc_time_list(val_accs, train_time)

            labels.append(file_name)

            if pre_time is not None:
                idx = 0
                while idx < len(train_time) and train_time[idx] <= pre_time[k]:
                    # while idx < len(train_time) and train_time[idx] < pre_time:
                    idx += 1
                # print(k, b, pre_time[k], idx, train_time[:idx])
                train_time = np.array(train_time)[:idx].tolist()
                val_acc = np.array(val_acc)[:idx].tolist()
                # print(train_time)
                assert (len(train_time) == len(val_acc))
                # plot_file_name = plot_path + k + f'-preT.pdf'

            plot_acc.append(val_acc)
            plot_time.append(train_time)

        if len(plot_acc) == 0:
            continue
        plot_file_name = plot_path + k + '.pdf'

        if pre_epoch is not None:
            assert isinstance(pre_epoch, int)
            plot_time = np.array(plot_time)[:, :pre_epoch].tolist()
            plot_acc = np.array(plot_acc)[:, :pre_epoch].tolist()
            plot_file_name = plot_path + k + f'-{pre_epoch}.pdf'

        if pre_time is not None:
            plot_file_name = plot_path + k + f'-preT.pdf'
        # print(plot_time)

        # print(plot_file_name)
        # print(type(plot_time))
        # print(len(plot_acc[0]), len(plot_time[0]))
        # print(plot_time)
        plot_line(X=plot_time,
                  Y=plot_acc,
                  labels=labels,
                  savefile=plot_file_name,
                  y_label='Eval Acc')


def get_best_accs(pre_path, datasets, batch_size):
    # def draw_batch_size(batch_size, epochs, , pre_epoch=None):
    # pre_path = './log/batch-size/'
    # plot_path = './pdf/batch-size/'
    acc_dict = {}
    for ds in datasets:
        bs = batch_size[ds]
        b_max = max(bs)
        for b in bs:
            file_name = ds + '_' + b
            file_path = pre_path + file_name + '.log'
            if not os.path.exists(file_path):
                print(file_path, 'not exist.')
                continue
            acc_mean, acc_var = parse_best_acc(file_path)
            # print(acc_mean, acc_var)
            # b = 'full' if b == b_max else str(b)
            b = str(b)
            acc_dict[str(ds) + b] = f'{acc_mean}({acc_var})'
    # print(acc_dict)
    return acc_dict


def get_best_accs_from_list(acc_list):
    # def draw_batch_size(batch_size, epochs, , pre_epoch=None):
    # pre_path = './log/batch-size/'
    # plot_path = './pdf/batch-size/'
    acc_dict = {}

    for k, v in acc_list.items():
        acc_dict[k] = max(v)
        print(k, acc_dict[k])
    return acc_dict


def get_accs(pre_path, datasets, batch_size):
    acc_dict = {}
    for ds in datasets:
        bs = batch_size[ds]
        b_max = max(bs)
        for b in bs:
            file_name = ds + '_' + b
            file_path = pre_path + file_name + '.log'
            if not os.path.exists(file_path):
                print(file_path, 'not exist.')
                continue
            train_acc, eval_acc, test_acc = parse_acc(file_path)
            b = str(b)
            acc_dict[str(ds) + b] = [train_acc, eval_acc, test_acc]
    return acc_dict


def get_list_mode(pre_path, datasets, batch_size, mode):
    acc_dict = {}
    for ds in datasets:
        print(ds, batch_size.keys())
        bs = batch_size[ds]
        b_max = max(bs)
        for b in bs:
            file_name = ds + '_' + b
            file_path = pre_path + file_name + '.log'
            if not os.path.exists(file_path):
                print(file_path, 'not exist.')
                continue
            parse_list = parse_mode(file_path, mode)
            acc_dict[str(ds) + str(b)] = parse_list
    return acc_dict


def get_list_mode(pre_path, dataset_name, mode):
    file_name = dataset_name
    file_path = pre_path + file_name + '.log'
    if not os.path.exists(file_path):
        print(file_path, 'not exist.')
    parse_list = parse_mode(file_path, mode)
    return parse_list


def draw_batch_size_epoch_time(acc_dict, train_time_dict, datasets, batch_size,
                               plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        print('\ndataset: ' + ds)
        top_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(acc_dict)
        for b in batch_size[ds]:
            # top_acc.append(float(acc_dict[str(ds) + str(b)]))
            if '(' in train_time_dict[str(ds) + str(b)]:
                train_time.append(
                    float(train_time_dict[str(ds) + str(b)]
                          [:train_time_dict[str(ds) + str(b)].find('(')]))
            else:
                train_time.append(float(train_time_dict[str(ds) + str(b)]))
            # b = 'full' if b == batch_size[ds][-1] else b
            # bs.append(float(b))
            bs.append(math.log2(float(b)))
        # print('|batch size|' + '|'.join(bs) + '|')
        # print('|top val acc|' + '|'.join(train_time) + '|')
        # print(bs, train_time)
        assert (len(bs) == len(train_time))
        labels.append(ds)
        X.append(bs)
        Y.append(train_time)
        # print(bs, train_time)

        # draw_batch_size(batch_size, epochs, './log/batch-size/gpu/', './pdf/batch-size/gpu/', datasets)
    max_ticks = max(max(x) for x in X)
    x_names = [x for x in range(math.ceil(max_ticks))]
    max_y = max(max(x) if len(x) > 0 else 0 for x in Y)
    y_ticks = np.linspace(0, max_y, 5).tolist()
    y_name = [f'{x:.2f}' for x in y_ticks]

    print(y_ticks, y_name)
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=plot_path + 'epoch_time.pdf',
              y_label='Epoch Time (s)',
              x_ticks=x_names,
              x_name=x_names,
              x_label='batch size (log)',
              loc='upper right',
              y_ticks=y_ticks,
              y_names=y_name)
    # assert False


def draw_batch_size_top_acc(acc_dict, datasets, batch_size, plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        print('\ndataset: ' + ds)
        top_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(acc_dict)
        for b in batch_size[ds]:
            # top_acc.append(float(acc_dict[str(ds) + str(b)]))
            # if '(' in acc_dict[str(ds) + str(b)]:
            #   top_acc.append(float(acc_dict[str(ds) + str(b)][:acc_dict[str(ds) + str(b)].find('(')]))
            # else:
            #   top_acc.append(float(acc_dict[str(ds) + str(b)]))
            top_acc.append(float(acc_dict[str(ds) + str(b)]))
            # b = 'full' if b == batch_size[ds][-1] else b
            # bs.append(float(b))
            bs.append(math.log2(float(b)))
        # print('|batch size|' + '|'.join(bs) + '|')
        # print('|top val acc|' + '|'.join(top_acc) + '|')
        # print(bs, top_acc)

        assert (len(bs) == len(top_acc))
        labels.append(ds)
        X.append(bs)
        Y.append(top_acc)
        # print(bs, top_acc)

        # draw_batch_size(batch_size, epochs, './log/batch-size/gpu/', './pdf/batch-size/gpu/', datasets)
    max_ticks = max(max(x) for x in X)
    # print(max_ticks)
    x_names = [x for x in range(math.ceil(max_ticks))]
    # # x_names = []
    # tmp = 0
    # while pow(2, tmp) < max_ticks:
    #   x_ticks.append(pow(2, tmp))
    #   x_names.append(tmp)
    #   tmp += 1
    # print(x_ticks)
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=plot_path,
              y_label='Eval Acc',
              x_ticks=x_names,
              x_name=x_names,
              x_label='batch size (log)',
              loc='lower right',
              high_mark='x')


def get_speed_acc_epoch(val_list, epoch_num, epoch_time):
    # print('input', epoch_time, times)
    ret = 0
    max_acc = 0

    epoch_num = min(epoch_num, len(val_list))
    for i in range(epoch_num):
        x = float(val_list[i])
        if x > max_acc:
            max_acc = x
            ret = x / (float(epoch_time) * (i + 1))
    return ret


def get_speed_acc(val_list, epoch_time, times):
    ret = 0
    max_acc = 0
    now_time = 0
    print(len(val_list), len(epoch_time))
    print(val_list[-10:], epoch_time[-10:])
    # assert(len(val_list) == len(epoch_time))
    for x, y in zip(val_list, epoch_time):
        x = float(x)
        y = float(y)

        if now_time > times:
            break

        now_time += y

        ####################################
        if x > max_acc:
            max_acc = x
            ret = x / now_time

        ####################################
        # ret = x / now_time
        ####################################

    return ret


def draw_batch_size_converge_speed(acc_dict, train_time_dict, datasets,
                                   batch_size, pre_epoch, pre_time, plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        # print('\ndataset: ' + ds)
        speed_acc = []
        train_time = []
        bs = []
        # print(acc_dict)
        idx = pre_epoch[ds]
        for b in batch_size[ds]:
            print(f'batch_size_speed {ds}_{b}')
            val_accs = acc_dict[str(ds) + str(b)]
            # print(ds, b, idx)
            train_time = train_time_dict[str(ds) + str(b)]
            speed_acc.append(val_accs[idx] / train_time[idx])
            # speed_acc.append(get_speed_acc(val_accs, train_time, float(time_length)))
            # bs.append(math.log2(float(b)))
            bs.append(float(b))
        assert (len(bs) == len(speed_acc))
        labels.append(ds)
        X.append(bs)
        Y.append(speed_acc)
    max_ticks = max(max(x) for x in X)
    x_names = [x for x in range(math.ceil(max_ticks))]
    # print(x_names)
    max_yticks = max(max(x) for x in Y)
    y_ticks = np.linspace(0, max_yticks, 5).tolist()
    y_names = [f'{x:.2f}' for x in y_ticks]
    save_file_path = plot_path + 'acc_speed.pdf'
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=save_file_path,
              y_label='Acc Speed (acc/s)',
              x_ticks=x_names,
              x_name=x_names,
              x_label='batch size (log scale)',
              y_ticks=y_ticks,
              y_names=y_names,
              loc='upper right',
              high_mark='x',
              xscale=2)

    # print(bs, top_acc)


def accumulate_time_list(time_list):
    return list(itertools.accumulate(time_list))


def accumulate_time_dict(time_dict):
    ret = {}
    for k, v in time_dict.items():
        ret[k] = accumulate_time_list(v)
    return ret


def draw_batch_size_converge(acc_dict, train_time_dict, datasets, batch_size,
                             pre_time, plot_path, y_lim):
    create_dir(plot_path)
    for ds in datasets:
        X = []
        Y = []
        labels = []

        speed_acc = []
        bs = []

        for b in batch_size[ds]:
            val_accs = acc_dict[str(ds) + str(b)]
            train_time = train_time_dict[str(ds) + str(b)]
            if pre_time is not None:
                idx = 0
                for x in train_time:
                    idx += 1
                    if pre_time[ds] <= x:
                        break
                # print(idx)
                val_accs = val_accs[:idx]
                train_time = train_time[:idx]

            assert (len(val_accs) == len(train_time))
            print(ds, b, len(train_time))
            X.append(train_time)
            Y.append(val_accs)
            labels.append(str(b))
            # labels.append(str(ds) + str(b))
        max_x = max(max(x) for x in X)
        max_y = max(max(x) for x in Y)
        x_names = np.linspace(0, max_x, 5).tolist()
        x_names = [round(x, 1) for x in x_names]
        y_names = np.linspace(0, max_y, 5).tolist()
        y_names = [round(x, 1) for x in y_names]

        savefile = plot_path + ds + '-converge.pdf'
        plot_line(X=X,
                  Y=Y,
                  labels=labels,
                  savefile=savefile,
                  y_label='Val Acc',
                  x_ticks=x_names,
                  x_name=x_names,
                  x_label='time (s)',
                  y_ticks=y_names,
                  y_names=y_names,
                  loc='lower right',
                  high_mark='x',
                  ylim=y_lim[ds],
                  draw_small=True)


def draw_batch_size_acc(acc_dict, train_time_dict, datasets, batch_size,
                        pre_time, plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        # print('\ndataset: ' + ds)
        top_acc = []
        speed_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(acc_dict)
        time_length = pre_time[ds]
        for b in batch_size[ds]:
            print(f'batch_size_speed {ds}_{b}')
            val_accs = acc_dict[str(ds) + str(b)]
            train_time = train_time_dict[str(ds) + str(b)]
            if len(train_time) > len(val_accs):
                train_time = train_time[:-1]
            train_time_sum = list(itertools.accumulate(train_time))
            X.append(train_time_sum)
            Y.append(val_accs)
            # print(bs[-1], speed_acc[-1])
            labels.append(ds + '_' + b)

    max_ticks = max(max(x) for x in X)
    # x_names = [x for x in range(math.ceil(max_ticks))]
    x_names = np.linspace(0, max_ticks, 5).tolist()
    max_yticks = max(max(x) for x in Y)
    y_ticks = np.linspace(0, max_yticks, 5).tolist()
    y_names = [f'{x:.2f}' for x in y_ticks]
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=plot_path,
              y_label='Acc',
              x_ticks=x_names,
              x_name=x_names,
              x_label='batch size (log)',
              y_ticks=y_ticks,
              y_names=y_names,
              loc='lower right',
              high_mark='x')


def draw_batch_size_speed_epoch(acc_dict, train_time_dict, datasets,
                                batch_size, epochs, plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        # print('\ndataset: ' + ds)
        top_acc = []
        speed_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(acc_dict)
        epoch_nums = epochs[ds]
        for b in batch_size[ds]:
            val_accs = acc_dict[str(ds) + str(b)][1]
            train_time = train_time_dict[str(ds) + str(b)]
            speed_acc.append(
                get_speed_acc_epoch(val_accs, epoch_nums, train_time))
            bs.append(math.log2(float(b)))
            print(bs[-1], train_time, speed_acc[-1])
        assert (len(bs) == len(speed_acc))
        labels.append(ds)
        X.append(bs)
        Y.append(speed_acc)

        # draw_batch_size(batch_size, epochs, './log/batch-size/gpu/', './pdf/batch-size/gpu/', datasets)
    max_ticks = max(max(x) for x in X)
    x_names = [x for x in range(math.ceil(max_ticks))]

    max_yticks = max(max(x) for x in Y)
    y_ticks = np.linspace(0, max_yticks, 5).tolist()
    y_names = [f'{x:.2f}' for x in y_ticks]
    # # x_names = []
    # tmp = 0
    # while pow(2, tmp) < max_ticks:
    #   x_ticks.append(pow(2, tmp))
    #   x_names.append(tmp)
    #   tmp += 1
    # print(x_ticks)
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=plot_path,
              y_label='Acc Speed (acc/s)',
              x_ticks=x_names,
              x_name=x_names,
              x_label='batch size (log)',
              y_ticks=y_ticks,
              y_names=y_names,
              loc='lower right',
              high_mark='x')


def draw_sample_rate_top_acc(acc_dict, train_time_dict, datasets, batch_size,
                             plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        print('\ndataset: ' + ds)
        top_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(acc_dict)
        for b in batch_size[ds]:
            # top_acc.append(float(acc_dict[str(ds) + str(b)]))
            if '(' in acc_dict[str(ds) + str(b)]:
                top_acc.append(
                    float(acc_dict[str(ds) +
                                   str(b)][:acc_dict[str(ds) +
                                                     str(b)].find('(')]))
            else:
                top_acc.append(float(acc_dict[str(ds) + str(b)]))
            # b = 'full' if b == batch_size[ds][-1] else b
            if ',' in b:
                b = b[:b.find(',')]
            bs.append(float(b))
            # bs.append(math.log2(float(b)))
        # print('|batch size|' + '|'.join(bs) + '|')
        # print('|top val acc|' + '|'.join(top_acc) + '|')
        # print(bs, top_acc)

        assert (len(bs) == len(top_acc))
        labels.append(ds)
        X.append(bs)
        Y.append(top_acc)
        # print(bs, top_acc)

        # draw_batch_size(batch_size, epochs, './log/batch-size/gpu/', './pdf/batch-size/gpu/', datasets)
    max_ticks = max(max(x) for x in X)
    # print(max_ticks)
    # x_names = [x for x in range(int(math.ceil(max_ticks)), 0.1)]
    # x_names = [x for x in np.linspace(0.1, max_ticks, 11)]
    x_names = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # print(x_names)
    # print(max_ticks)
    # assert False
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=plot_path + 'topacc.pdf',
              y_label='Eval Acc',
              x_ticks=x_names,
              x_name=x_names,
              x_label='sample rate',
              loc='lower right',
              high_mark='x')


def draw_sample_rate_epoch_time(acc_dict, train_time_dict, datasets,
                                batch_size, plot_path):
    labels = []
    X = []
    Y = []
    for ds in datasets:
        print('\ndataset: ' + ds)
        top_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(acc_dict)
        for b in batch_size[ds]:
            # top_acc.append(float(acc_dict[str(ds) + str(b)]))
            if '(' in train_time_dict[str(ds) + str(b)]:
                train_time.append(
                    float(train_time_dict[str(ds) + str(b)]
                          [:train_time_dict[str(ds) + str(b)].find('(')]))
            else:
                train_time.append(float(train_time_dict[str(ds) + str(b)]))
            # b = 'full' if b == batch_size[ds][-1] else b
            # bs.append(float(b))
            if ',' in b:
                b = b[:b.find(',')]
            bs.append(float(b))
        # print('|batch size|' + '|'.join(bs) + '|')
        # print('|top val acc|' + '|'.join(train_time) + '|')
        # print(bs, train_time)
        assert (len(bs) == len(train_time))
        labels.append(ds)
        X.append(bs)
        Y.append(train_time)
        # print(bs, train_time)
        # draw_batch_size(batch_size, epochs, './log/batch-size/gpu/', './pdf/batch-size/gpu/', datasets)
    max_ticks = max(max(x) for x in X)
    # x_names = [x for x in range(math.ceil(max_ticks))]
    x_names = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    max_y = max(max(x) if len(x) > 0 else 0 for x in Y)
    y_ticks = np.linspace(0, max_y, 5).tolist()
    y_name = [f'{x:.2f}' for x in y_ticks]

    print(y_ticks, y_name)
    plot_line(X=X,
              Y=Y,
              labels=labels,
              savefile=plot_path + 'epoch_time.pdf',
              y_label='Epoch Time (s)',
              x_ticks=x_names,
              x_name=x_names,
              x_label='sample rate',
              loc='upper left',
              y_ticks=y_ticks,
              y_names=y_name)


def print_markdown_table(train_time_dict, acc_dict, datasets, batch_size):
    print("====================================\n")
    for ds in datasets:
        print('\ndataset: ' + ds)
        # print('|dataset|' + '|'.join(batch_size[ds]) + '|')
        # print('|' + ' ---- | ' * (len(batch_size[ds]) + 1))
        top_acc = []
        train_time = []
        bs = []
        bs_max = max(batch_size[ds])
        # print(train_time_dict)
        for b in batch_size[ds]:
            top_acc.append(train_time_dict[str(ds) + str(b)])
            train_time.append(acc_dict[str(ds) + str(b)])
            b = 'full' if b == batch_size[ds][-1] else b
            bs.append(b)
        print('|batch size|' + '|'.join(bs) + '|')
        print('|' + ' ---- | ' * (len(batch_size[ds]) + 1))
        print('|top val acc|' + '|'.join(top_acc) + '|')
        print('|train time|' + '|'.join(train_time) + '|')

    print("\n====================================\n")


def show_mark_table(x_aixs, y_aixs, dict):
    # assert(len(x_aixs) == len(y_aixs))
    print("====================================\n")

    print('|dataset|' + '|'.join(y_aixs) + '|')
    print('|' + ' ---- | ' * (len(y_aixs) + 1))
    for x in x_aixs:
        one_line = f'|{x}|'
        for y in y_aixs:
            if x + y in dict:
                one_line += dict[x + y] + '|'
            else:
                one_line += '|'
        print(one_line)
    print("\n====================================\n")


def draw_batch_type():
    datasets = [
        'ppi', 'ppi-large', 'flickr', 'AmazonCoBuy_computers',
        'AmazonCoBuy_photo'
    ]
    batch_size = {
        'ppi': ('sequence', 'shuffle', 'random', 'metis'),
        'ppi-large': ('sequence', 'shuffle', 'random', 'metis'),
        'flickr': ('sequence', 'shuffle', 'random', 'metis'),
        'AmazonCoBuy_computers': ('sequence', 'shuffle', 'random', 'metis'),
        'AmazonCoBuy_photo': ('sequence', 'shuffle', 'random', 'metis'),
    }
    pre_time = {
        # 'cora': 200, 'citeseer': 200, 'pubmed': 200,
        'ppi': 10,
        'ppi-large': 15,
        'flickr': 50,
        'AmazonCoBuy_computers': 20,
        'AmazonCoBuy_photo': 10,
        # 'reddit': 100, 'yelp': 100, 'ogbn-arxiv': 100,
    }
    draw_batch_size(batch_size, None, './log/batch-type/', './pdf/batch-type/',
                    datasets)
    draw_batch_size(batch_size,
                    None,
                    './log/batch-type/',
                    './pdf/batch-type/',
                    datasets,
                    pre_time=pre_time,
                    suffix_name='-preT.pdf')

    acc_dict = get_best_accs('./log/batch-type/', datasets, batch_size)
    train_time_dict = get_train_time('./log/batch-type/', datasets, batch_size)
    # show_mark_table(['ppi', 'ppi-large', 'flickr', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo', 'reddit'], ['512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072', '262144', 'full'], acc_dict)
    print_latex_table(acc_dict,
                      train_time_dict,
                      datasets,
                      batch_size,
                      table_name='batch type',
                      desc='Training performance on ')
    # print_markdown_table(acc_dict, train_time_dict, datasets, batch_size)
    # draw_batch_size_top_acc(acc_dict, train_time_dict, datasets, batch_size)
    # print(acc_dict)


def create_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def exp01_batch_size(log_path, plot_path):
    create_dir(plot_path)
    batch_size = {
        # 'ppi': ('1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '2048', '4096', '9716'),
        # 'ppi-large': ('1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '44906'),
        # 'flickr': ('1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '44625'),
        # 'AmazonCoBuy_computers': ('1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8250'),
        # 'AmazonCoBuy_photo': ('1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4590'),
        # 'ogbn-arxiv': ('1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', '90941'),
        'pubmed': ('128', '512', '1024', '2048', '4096', '8192', '11830'),
        'AmazonCoBuy_computers': ('512', '1024', '2048', '4096', '8250'),
        'AmazonCoBuy_photo': ('512', '1024', '2048', '4590'),
        # 'reddit': ('512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072', '153431', 'full'),
        # 'ogbn-arxiv': ('24576', '49152', '90941', 'full', '90941-full', '49152-full', '24576-full', 'full-no-dropout'),
        'ogbn-arxiv':
        ('24576-val-sample', '49152-val-sample', '90941-val-sample',
         'fullgraph', '90941-val-full', '49152-val-full', '24576-val-full'),
        'ogbn-arxiv': ('24576-val-sample', '49152-val-sample',
                       '90941-val-sample', 'fullgraph', '90941-val-full',
                       '49152-val-full', '24576-val-full', 'mini-fullgraph'),
        'ogbn-arxiv': ('512', '12288', '49152', '90941'),
        'ogbn-arxiv':
        ('512', '3072', '6144', '12288', '24576', '49152', '90941'),
        'reddit': ('512', '2048', '8192', '32768', '131072', '153431'),
        # 'ogbn-arxiv': ('512', '3072', '24576', '49152', '90941', 'switch'),
        # 'ogbn-products': ('512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072', '196615')
    }

    pre_epoch = {
        # 'cora': 200, 'citeseer': 200, 'pubmed': 200,
        'ppi': 1,
        'ppi-large': 5,
        'flickr': 5,
        'AmazonCoBuy_computers': 0,
        'AmazonCoBuy_photo': 0,
        'reddit': 0,
        'yelp': 0,
        'ogbn-arxiv': 0,
        'pubmed': 0,
        'ogbn-products': 0
    }
    pre_time = {
        # 'cora': 200, 'citeseer': 200, 'pubmed': 200,
        'ppi': 5,
        'ppi-large': 5,
        'flickr': 5,
        'AmazonCoBuy_computers': 7,
        'AmazonCoBuy_photo': 7,
        'ogbn-arxiv': 500,
        'pubmed': 10,
        'reddit': 50,
        'ogbn-arxiv': 20,
        'pubmed': 3,
        'reddit': 500,
        'ogbn-products': 20
        # 'reddit': 100, 'yelp': 100, 'ogbn-arxiv': 100,
    }
    y_lim = {
        # 'cora': 200, 'citeseer': 200, 'pubmed': 200,
        # 'ppi': 0.6, 'ppi-large': 5,'flickr': 5,
        # 'AmazonCoBuy_computers': 0.5, 'AmazonCoBuy_photo': 0.65,
        # 'ogbn-arxiv': 0.60, 'pubmed': 0.84, 'reddit': 0.73,
        # 'ogbn-products': 0.65
        'ppi': 0,
        'ppi-large': 0,
        'flickr': 0,
        'AmazonCoBuy_computers': 0,
        'AmazonCoBuy_photo': 0,
        'ogbn-arxiv': 0.60,
        'pubmed': 0.8,
        'reddit': 0.8,
        'ogbn-products': 0
        # 'reddit': 100, 'yelp': 100, 'ogbn-arxiv': 100,
    }
    # datasets = ['pubmed', 'reddit', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo', 'ogbn-arxiv', 'ogbn-products']
    # datasets = ['ogbn-arxiv']
    datasets = [
        'ogbn-arxiv', 'pubmed', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo'
    ]
    datasets = ['pubmed']
    datasets = ['AmazonCoBuy_photo']
    datasets = ['AmazonCoBuy_computers']
    datasets = ['ogbn-arxiv']
    datasets = ['reddit']

    # batch_size 8192 gcn_run_tim 23.187 gpu_mem 2246.438M
    gcn_batch_size_dict = get_list_mode(log_path, datasets, 'gcn_batch_size')
    gcn_run_time_dict = get_list_mode(log_path, datasets, 'gcn_run_time')
    gcn_gpu_mem_dict = get_list_mode(log_path, datasets, 'gcn_gpu_mem')

    print(gcn_batch_size_dict)
    print(gcn_run_time_dict)
    print(gcn_gpu_mem_dict)

    assert False

    eval_acc_dict = get_list_mode(log_path, datasets, batch_size, 'val_acc')
    # print(eval_acc_dict.keys())
    # assert False
    train_time_epoch_dict = get_list_mode(log_path, datasets, batch_size,
                                          'train_time')
    train_time_dict = accumulate_time_dict(train_time_epoch_dict)
    # best_acc_dict = get_best_accs_from_list(eval_acc_dict)

    draw_batch_size_converge(eval_acc_dict,
                             train_time_dict,
                             datasets,
                             batch_size,
                             None,
                             plot_path=plot_path,
                             y_lim=y_lim)
    # draw_batch_size_converge(eval_acc_dict, train_time_dict, datasets, batch_size, pre_time, plot_path=plot_path, y_lim=y_lim)

    draw_batch_size_converge_speed(eval_acc_dict,
                                   train_time_dict,
                                   datasets,
                                   batch_size,
                                   pre_epoch,
                                   pre_time,
                                   plot_path=plot_path)
    # draw_batch_size_top_acc(best_acc_dict, datasets, batch_size, plot_path=plot_path + 'top_acc.pdf')


if __name__ == '__main__':

    # exp01_batch_size('./log/batch-size/aliyun/epoch-2-2/', './log/batch-size/aliyun/pdf/epoch-2-2/')
    # exp01_batch_size('./log/batch-size/epoch-4-4/', './log/batch-size/pdf/epoch-4-4/')
    # exp01_batch_size('./log/batch-size/epoch-10-25/', './log/batch-size/pdf/epoch-10-25/')

    # exp01_batch_size('./log/batch-size/dabian/epoch-2-2/', './log/batch-size/dabian/pdf/epoch-2-2/')
    # exp01_batch_size('./log/batch-size/dabian/epoch-4-8/', './log/batch-size/dabian/pdf/epoch-4-8/')
    # exp01_batch_size('./log/batch-size/dabian/epoch-8-16/', './log/batch-size/dabian/pdf/epoch-8-16/')

    # exp01_batch_size('./log/batch-size/dabian-val-mini/epoch-2-2/', './log/batch-size/dabian-val-mini/pdf/epoch-2-2/')

    # exp01_batch_size('./log/batch-size/dabian-val-full/epoch-4-8/', './log/batch-size/dabian-val-full/pdf/epoch-4-8/')

    # exp01_batch_size('./log/batch-size/dabian-val-full/epoch-8-16/', './log/batch-size/dabian-val-full/pdf/epoch-8-16/')

    def parse_all(path, ds):
        gcn_batch_size_dict = get_list_mode(path, ds, 'gcn_batch_size')
        gcn_run_time_dict = get_list_mode(path, ds, 'gcn_run_time')
        # gcn_gpu_mem_dict = get_list_mode(path, ds, 'gcn_gpu_mem')
        gcn_cache_rate = get_list_mode(path, ds, 'cache_rate')
        # return [gcn_batch_size_dict, gcn_run_time_dict, gcn_gpu_mem_dict, gcn_cache_rate]
        return [gcn_batch_size_dict, gcn_run_time_dict, gcn_cache_rate]

    # batch_size 8192 gcn_run_tim 23.187 gpu_mem 2246.438M
    path = './log/gpu-cache/cache-mem/4-8/'
    datasets = ['reddit', 'ogbn-arxiv']
    for ds in datasets:
        print(ds)
        bs, run, cache = parse_all(path, ds)
        print('batch_size_dict', bs)
        print('run_time_dict', run)
        # print('gpu_mem_dict', mem)
        print('gpu_cache_dict', cache)

    # gcn_batch_size_dict = get_list_mode('./log/gpu-cache/cache-mem/epoch-4-8/', datasets, 'gcn_batch_size')
    # gcn_run_time_cache = get_list_mode('./log/gpu-cache/cache-mem/epoch-4-8/', datasets, 'gcn_run_time_cache')
    # gcn_gpu_mem_cache = get_list_mode('./log/gpu-cache/cache-mem/epoch-4-8/', datasets, 'gcn_gpu_mem_cache')
    # gcn_cache_mem = get_list_mode('./log/gpu-cache/cache-mem/epoch-4-8/', datasets, 'gcn_cache_mem')

    assert False

    exp01_batch_size('./log/gpu-cache/batch-size-mem/4-8/',
                     './log/gpu-cache/batch-size-mem/pdf/4-8/')
    # exp01_batch_size('./log/batch-size/dabian-val-full/epoch-8-16/', './log/batch-size/dabian-val-full/pdf/epoch-8-16/')

    # exp01_batch_size('./log/batch-size/aliyun-full/epoch-16-32/', './log/batch-size/aliyun-full/pdf/epoch-16-32/')
    # exp01_batch_size('./log/batch-size/aliyun/dabian-val-full/epoch-16-32/', './log/batch-size/aliyun/pdf/dabian-val-full/epoch-16-32/')
