# -*- coding: utf-8 -*-

from ast import parse
from cProfile import label
from calendar import c
from time import time
import numpy as np
import matplotlib
import os, re
import itertools

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


def plot_bar(x_name, y_label,x_label, datas, labels, filename='bar.pdf', color=None):
  # print(x_name, y_name)
  # print(datas)
  # print(labels)
  assert (len(datas[0]) == len(x_name))
  #  == len(labels)
  # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
  # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
  # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]  

  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  plt.figure(figsize=(7, 4))
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
  up = np.max(datas)
  plt.ylim(low, up + 1)
  # plt.xlabel("Amount of Data", fontsize=15)
  # plt.ylabel(f"Time (s)", fontsize=20)
  plt.ylabel(y_label, fontsize=20)
  plt.xlabel(x_label, fontsize=20)
  # labels = ['GraphScope', 'NTS']

  # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
  if color is None:
    color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue']
  

  for i, data in enumerate(datas):
    plt.bar(x + width * i, data, width=width, color=color[i], edgecolor='w')  # , edgecolor='k',)
    

  plt.xticks(x + offset, labels=x_name, fontsize=15)

  plt.legend(labels=labels, ncol=2, prop={'size': 14})

  plt.tight_layout()
  plt.savefig(filename, format='pdf')
  plt.show()
  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中


def plot_line(X, Y, labels, savefile=None, color=None, y_label=None):
  assert(len(X) == len(Y) == len(labels))
  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  plt.figure(figsize=(8, 6))
  # linestyle = "-"
  plt.grid(linestyle="-.")  # 设置背景网格线为虚线
  # ax = plt.gca()
  # ax.spines['top'].set_visible(False)  # 去掉上边框
  # ax.spines['right'].set_visible(False)  # 去掉右边框

  linewidth = 2.0
  markersize = 7

  if color is None:
    color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue']
  
  for i in range(len(X)):
    plt.plot(X[i], Y[i], marker='', markersize=markersize, color=color[i], alpha=1, label=labels[i], linewidth=linewidth)
    pos = np.where(np.amax(Y[i]) == Y[i])[0].tolist()
    pos = pos[0]
    # print(pos)
    # print(Y[i][pos[0]], Y[i][pos[1]])

    plt.plot(X[i][pos], Y[i][pos], marker='x', markersize=markersize, color='red', alpha=1, linewidth=linewidth)
    plt.plot(X[i][pos], Y[i][pos], marker='.', markersize=markersize-2, color=color[i], alpha=1, linewidth=linewidth)


  
  x_ticks = np.linspace(0, np.max(X), 5).tolist()
  y_labels = [f'{x:.2f}' for x in x_ticks]
  plt.xticks(x_ticks, y_labels, fontsize=15)  # 默认字体大小为10

  y_ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
  y_lables = ['10%', '30%', '50%', '70%', '90%']
  plt.yticks(np.array(y_ticks), y_lables, fontsize=15)
  # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
  # plt.text(1, label_position, dataset,fontsize=25, fontweight='bold')
  # plt.xlabel("Edge Miss Rate", fontsize=15)
  if not y_label:
    y_label = "Val"
  plt.ylabel(f"{y_label} Acc", fontsize=15)
  plt.xlabel(f"Time (s)", fontsize=15)
  plt.xlim(0, np.max(X) + 1)  # 设置x轴的范围
  plt.ylim(0, 1)

  # plt.legend()
  # 显示各曲线的图例 loc=3 lower left
  plt.legend(loc=0, numpoints=1, ncol=2)
  leg = plt.gca().get_legend()
  ltext = leg.get_texts()
  plt.setp(ltext, fontsize=15)
  # plt.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
  plt.tight_layout()
  if not savefile:
    savefile = 'plot_line.png'
  plt.savefig(f'./{savefile}', format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
  plt.show()


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
  idx = min(idx+10, len(accs))
  accs_ret = accs[:idx+1]
  times_ret = list(itertools.accumulate(times[:idx+1]))
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
        dataset = line[l+1:r]
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


def parse_acc(filename, epoch):
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
  assert(runs * epoch == len(train_acc))
  train_acc = sum(np.array_split(train_acc, runs)) / runs
  val_acc = sum(np.array_split(val_acc, runs)) / runs
  test_acc = sum(np.array_split(test_acc, runs)) / runs
  return [train_acc, val_acc, test_acc]


def parse_time(filename, epoch):
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
  assert(runs * 3 == len(time_list))
  time_list = sum(np.array_split(time_list, runs)) / runs
  return time_list


def parse_dataset(filename):
  if not os.path.exists(filename):
    print(f'{filename} not exist')
  dataset = None
  with open(filename) as f:
    for line in f.readlines():
      if line.find('edge_file') >= 0:
        l, r = line.rfind('/'), line.rfind('.')
        dataset = line[l+1:r]
        break
  assert(dataset)
  return dataset


def draw_batch_size():
  batch_size = {'cora': ('32', '64', '140'),
                'citeseer': ('32', '64', '120'),
                'pubmed': ('32', '60'),
                'ppi': ('512' '1024' '2048' '4096' '8192' '9716'),
                'ppi-large': ('512' '1024' '2048' '4096' '8192' '16384' '32768' '44906'),
                'flickr': ('512' '1024' '2048' '4096' '8192' '16384' '32768' '44625'),
                'AmazonCoBuy_computers': ('512' '1024' '2048' '4096' '8192' '8250'),
                'AmazonCoBuy_photo': ('512' '1024' '2048' '4096' '4590'),
                'reddit': ('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '232965'),
                'yelp': ('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '131072' '262144' '537635'),
                'ogbn-arxiv': ('512' '1024' '2048' '4096' '8192' '16384' '32768' '65536' '90941'),
               }

  epochs = {'cora': 200, 'citeseer': 200, 'pubmed': 200, 
            'ppi': 200, 'ppi-large': 200,'flickr': 200,
            'AmazonCoBuy_computers': 200, 'AmazonCoBuy_photo': 200,
            'reddit': 100, 'yelp': 100, 'ogbn-arxiv': 100,
           }             

  pre_path = './log/batch-size/'
  labels = []

  for k, v in batch_size.items():
    plot_time, plot_acc, labels = [], [], []
    for b in v:
      file_name = k + '_' + b
      file_path = pre_path + file_name + '.log'
      if not os.path.exists(file_path):
        print(file_path, 'not exist.')
        continue
      train_accs, val_accs, test_accs = parse_acc(file_path, epochs[k])
      train_time, val_time, test_time = parse_time(file_path, epochs[k])
      val_acc, val_time = get_acc_time_list(val_accs, val_time)
      plot_acc.append(val_acc)
      plot_time.append(val_time)
      labels.append(file_name)
    
    if len(plot_acc) == 0:
      continue
    plot_line(plot_time, plot_acc, labels, pre_path + k + '-batch_size.pdf', y_label='Eval Acc')

      

if __name__ == '__main__':
  draw_batch_size()




# def draw():
#   X, Y = [], []
#   labels = []

#   # files = ['cora_seq.log', '_rand.log']
#   # datasets = ['cora', 'pubmed', 'citeseer', 'arxiv', 'reddit']
#   datasets = ['cora', 'pubmed', 'citeseer', 'arxiv', 'reddit']
#   modes = ['seq', 'shuffle', 'rand', 'low', 'high']
#   pre_path = './log/'
#   host_num = 8

#   for ds in datasets:
#     for type in ['Val', 'Test']:
#       X, Y, labels = [], [], []
#       T, T1, T2= [], [], []
#       idx = 1 if type == 'Val' else 2
#       for ms in modes:
#         name = pre_path + ds + '_' + ms + '.log'
#         if not os.path.exists(name):
#           print(name, 'not exist.')
#           continue
#         dataset, acc_list, time_list, time_cost = parse_log(name)
#         # print(time_cost)
#         # TIME(2) sample 0.236 compute_time 0.930 comm_time 1.966 mpi_comm 0.117 rpc_comm 0.812 rpc_wait_time 0.000
#         # print(time_cost)
#         # print(ds, ms)
#         if ds in ['arxiv', 'reddit']:
#           # print(time_cost)
#           compute_time = [time_cost[i][1] for i in range(host_num)]
#           comm_time = [time_cost[i][2] for i in range(host_num)]
#           all_time = [time_cost[i][1] + time_cost[i][2] for i in range(host_num)]
#           T.append(compute_time)
#           T1.append(comm_time)
#           T2.append(all_time)
#         # print(compute_time)
#         ret = get_time_acc(acc_list[idx], time_list[0], max(acc_list[idx]), False)
#         X.append(ret[0])
#         Y.append(ret[1])
#         labels.append(ds + '-' +ms)
#         print(ds+'_'+ms+'_'+type, max(ret[1]))
#       plot_line(X, Y, labels, pre_path+ds+'-'+type+'.pdf', y_label= type,)
#       print('len T', len(T))
#       if ds in ['arxiv', 'reddit']:
#         plot_bar([f'{i}' for i in range(host_num)], 'Compute Time (s)', 'Host', T, labels, pre_path+ds+'-compute'+'.pdf')
#         plot_bar([f'{i}' for i in range(host_num)], 'Comm Time (s)', 'Host', T1, labels, pre_path+ds+'-comm'+'.pdf')
#         plot_bar([f'{i}' for i in range(host_num)], 'Train Time (s)', 'Host', T2, labels, pre_path+ds+'-train'+'.pdf')
#   # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
#   # labels = ['GraphScope', 'NTS']
#   # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
#   # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435] 
#   # plot_bar(x_name, 'Time (s)', [aligraph, aligraph], labels, 'xx.pdf')
#   # assert(False)
