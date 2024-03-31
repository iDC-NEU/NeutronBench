import os
import re
import struct
import time
from functools import wraps
import numpy as np
import pandas as pd
import scipy.sparse as ss
from collections import Counter
import matplotlib.pyplot as plt


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


# def create_fi(path):
#   if path and not os.path.exists(path):
#     os.makedirs(path)


def read_edge_list_fron_binfile(filepath):
    binfile = open(filepath, 'rb')
    edge_num = os.path.getsize(filepath) // 8
    edge_list = []
    for _ in range(edge_num):
        u = struct.unpack('i', binfile.read(4))[0]
        v = struct.unpack('i', binfile.read(4))[0]
        edge_list.append((u, v))
    binfile.close()
    return edge_list


def show_time(func):

    @wraps(
        func
    )  # need add this for multiprocessing, keep the __name__ attribute of func
    def with_time(*args, **kwargs):
        time_cost = time.time()
        func(*args, **kwargs)
        time_cost = time.time() - time_cost
        print('function {} cost {:.2f}s'.format(func.__name__, time_cost))

    return with_time


def read_edgelist(filename, sep='\t'):
    data = pd.read_csv(filename, sep=sep, encoding='utf-8', header=None)
    return np.array(data)


def edgelist_to_coo_matrix(edgelist):
    "Read data file and return sparse matrix in coordinate format."
    # if the nodes are integers, use 'dtype = np.uint32'
    rows = edgelist[:, 0]
    cols = edgelist[:, 1]
    n_nodes = np.max(edgelist) + 1
    ones = np.ones(len(rows), np.uint32)
    matrix = ss.coo_matrix((ones, (rows, cols)), shape=(n_nodes, n_nodes))
    # print(matrix.shape)
    return matrix


def read_edlist_to_coo_matrix(filename='edges.txt'):
    "Read data file and return sparse matrix in coordinate format."
    # if the nodes are integers, use 'dtype = np.uint32'
    data = pd.read_csv(filename, sep='\t', encoding='utf-8', header=None)
    rows = data.iloc[:, 0]  # Not a copy, just a reference.
    cols = data.iloc[:, 1]
    n_nodes = max(data.max()) + 1
    ones = np.ones(len(rows), np.uint32)
    matrix = ss.coo_matrix((ones, (rows, cols)), shape=(n_nodes, n_nodes))
    # print(matrix.shape)
    return matrix


def draw_power_law(dataset, edge_list, lock=None):
    # convert to networkx is too slowly
    # G = nx.from_edgelist(edge_list, create_using=nx.DiGraph())
    # assert len(edge_list) == G.number_of_edges()
    # print('{} has {} edges'.format(filepath, G.number_of_edges()))
    # degree_freq = nx.degree_histogram(G)
    # degrees = range(len(degree_freq))

    # compute
    degree_dict = {}
    for u, v in edge_list:
        if u not in degree_dict:
            degree_dict[u] = 1
        else:
            degree_dict[u] += 1
    degree_list = degree_dict.values()
    max_degree = max(degree_list)
    tmp = Counter(degree_list)
    degrees = np.array(list(tmp.keys()))
    degree_freq = np.array(list(tmp.values()))
    # print(type(degrees), degrees)

    # save
    with open(f'./degree-distribute/npz/{dataset}.npz', 'wb') as f:
        np.save(f, degrees)
        np.save(f, degree_freq)

    # # load
    # with open(f'./degree-distribute/npz/{dataset}.npz', 'rb') as f:
    #   degrees = np.load(f)
    #   degree_freq = np.load(f)

    try:
        if lock is not None:
            lock.acquire()
        plt.figure(figsize=(8, 6))
        plt.loglog(degrees[:], degree_freq[:], '.')
        plt.xlabel('Degree', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        # plt.xlim(1, max(degrees))
        # plt.ylim(1, max(degree_freq))
        print(f'save to {os.getcwd()}/degree-distribute/{dataset}.pdf')
        plt.savefig(f'./degree-distribute/{dataset}.pdf', format='pdf')
    except Exception as err:
        raise err
    finally:
        if lock is not None:
            lock.release()


def plot_bar(x_name, y_name, datas, labels, filename='bar.pdf', color=None):
    assert (len(datas[0]) == len(x_name))
    #  == len(labels)
    # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
    # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
    # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="-.")  # 设置背景网格线为虚线
    # linestyle = "-"
    x = np.arange(len(x_name))
    fontsize = 12
    # n 为有几个柱子
    # total_width, n = 0.8, 2
    total_width, n = 0.8, len(datas)
    width = total_width / n
    offset = (total_width - width) / 2
    x = x - offset
    # x = x - total_width /2

    # low = 0.05
    # up = 0.44
    low = 0
    up = np.max(datas)
    up = np.max(datas) + 1
    plt.ylim(low, up)
    # plt.xlabel("Amount of Data", fontsize=15)
    # plt.ylabel(f"Time (s)", fontsize=20)
    plt.ylabel(y_name, fontsize=fontsize)
    # labels = ['GraphScope', 'NTS']

    # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
    if color is None:
        color = ['blue', 'green', 'orange', 'tomato', 'purple', 'deepskyblue']

    for i, data in enumerate(datas):
        plt.bar(x + width * i,
                data,
                width=width,
                color=color[i],
                edgecolor='w')  # , edgecolor='k',)

    plt.xticks(x + offset, labels=x_name, fontsize=fontsize, rotation=0)
    plt.yticks(fontsize=fontsize)

    # num1, num2 = 1, 1.1
    # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))
    plt.legend(labels=labels, ncol=4, prop={'size': fontsize}, loc='best')
    # num1, num2 = 0.9, 1.2
    # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))

    plt.tight_layout()
    print(f"save to {filename}")
    plt.savefig(filename, format='pdf')
    plt.show()
    # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中


def plot_stack_bar(x_name,
                   y_name,
                   datas,
                   labels,
                   filename='bar.pdf',
                   color=None):
    assert (len(datas[0]) == len(x_name))
    #  == len(labels)
    # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
    # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
    # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="-.")  # 设置背景网格线为虚线
    # linestyle = "-"
    x = np.arange(len(x_name))
    # n 为有几个柱子
    # total_width, n = 0.8, 2
    total_width, n = 0.8, len(datas)
    width = total_width / n
    offset = (total_width - width) / 2
    x = x - offset
    # x = x - total_width /2

    # low = 0.05
    # up = 0.44
    low = 0
    up = 1
    plt.ylim(low, up)
    # plt.xlabel("Amount of Data", fontsize=15)
    # plt.ylabel(f"Time (s)", fontsize=20)
    plt.ylabel(y_name, fontsize=12)
    # labels = ['GraphScope', 'NTS']

    # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
    if color is None:
        color = ['blue', 'green', 'orange', 'tomato', 'purple', 'deepskyblue']

    pre_bottom = np.zeros_like(datas[0])
    for i, data in enumerate(datas):
        plt.bar(x + width,
                data,
                label=labels[i],
                width=width,
                color=color[i],
                edgecolor=None,
                bottom=pre_bottom)  # , edgecolor='k',)
        pre_bottom += data
        # plt.bar(x + width * i, data, width=width, color=color[i], edgecolor='w')  # , edgecolor='k',)

    plt.xticks(x + offset, labels=x_name, fontsize=12, rotation=0)
    y_ticks = [f'{x:.0%}' for x in np.linspace(start=0, stop=1, num=6)]
    plt.yticks(np.linspace(start=0, stop=1, num=6),
               labels=y_ticks,
               fontsize=12)

    num1, num2 = 1, 1.2
    plt.legend(labels=labels,
               ncol=4,
               prop={'size': 11},
               bbox_to_anchor=(num1, num2))
    # plt.legend(labels=labels, ncol=2, prop={'size': 11}, loc='best')

    plt.tight_layout()
    print(f"save to {filename}")
    plt.savefig(filename, format='pdf')
    plt.show()
    # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中


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
              y_name=None,
              high_mark='.',
              ylim=None,
              draw_small=False,
              xscale=None):
    assert (len(X) == len(Y) == len(labels))
    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    # plt.figure(figsize=(8, 6))
    # linestyle = "-"
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=400)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.grid(linestyle="-.")  # 设置背景网格线为虚线
    # ax = ax.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    fontsize = 13
    linewidth = 1.2
    markersize = 7

    if color is None:
        # color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue', 'red', 'cyan', 'magenta', 'yellow', 'black']
        # color = ['blue', 'green', 'orange', 'purple', 'red', 'black', 'yellow', 'cyan', 'magenta', 'pink',  'deepskyblue', 'tomato']
        color = [
            'orange', 'blue', 'green', 'tomato', 'purple', 'deepskyblue', 'red'
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

        # plot max point
        # pos = np.where(np.amax(Y[i]) == Y[i])[0].tolist()
        # pos = pos[0]
        # ax.plot(X[i][pos], Y[i][pos], marker='x', markersize=markersize, color='red', alpha=1, linewidth=linewidth)
        # ax.plot(X[i][pos], Y[i][pos], marker=high_mark, markersize=markersize-2, alpha=1, linewidth=linewidth)

    if x_ticks is not None and x_name is not None:
        # print(x_ticks)
        ax.set_xticks(x_ticks, x_name, fontsize=fontsize - 2)  # 默认字体大小为10
        ax.set_xlim(np.min(x_ticks), np.max(x_ticks))
    else:
        max_xticks = max(max(x) if len(x) > 0 else 0 for x in X)
        x_ticks = np.linspace(0, max_xticks, 6).tolist()
        ax.set_xlim(np.min(x_ticks), np.max(x_ticks))
        x_name = [f'{x:.2f}' for x in x_ticks]
        ax.set_xticks(x_ticks, x_name, fontsize=fontsize - 2)  # 默认字体大小为10

    if y_ticks is not None and y_name is not None:
        # print(y_ticks)
        ax.set_yticks(y_ticks, y_name, fontsize=fontsize - 2)  # 默认字体大小为10
        ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
    else:
        max_xticks = max(max(x) if len(x) > 0 else 0 for x in Y)
        y_ticks = np.linspace(0, max_xticks, 6).tolist()
        ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
        y_name = [f'{x:.2f}' for x in y_ticks]
        ax.set_yticks(y_ticks, y_name, fontsize=fontsize - 2)  # 默认字体大小为10

    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    # ax.xlim(0, np.max(X) + 1)  # 设置x轴的范围

    if xscale is not None:
        ax.set_xscale('log', base=xscale)

    # ax.legend()
    # 显示各曲线的图例 loc=3 lower left
    if not loc:
        loc = 'best'
    ax.legend(loc=loc, numpoints=1, ncol=1, prop={'size': fontsize})
    # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))
    # leg = ax.gca().get_legend()
    # ltext = leg.get_texts()
    # ax.setp(ltext, fontsize=15)
    # ax.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
    # ax.tight_layout()

    if not savefile:
        savefile = 'plot_line.pdf'
    print(f'save to {savefile}')
    fig.savefig(f'{savefile}',
                format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    print("this utils fuction tools.")
