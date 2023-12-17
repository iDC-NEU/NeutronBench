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


# def plot_bar_balance(plot_params, Y, labels, xlabel, ylabel, xticks, yticks, color_list, ylim, anchor=None, figpath=None):
def plot_time(plot_params, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
  
  Y = [
    
    [4.175, 0.359,3.998,96.379, 110.079, 92.822, 324.820], # zerocopy epoch time
    [3.251, 0.315,3.356, 802.387, 902.274, 1080.576, 4609.499], # unified epoch time
  ]


  # to numpy
  for i in range(len(Y)):
    Y[i] = np.array(Y[i])
  

  # normalize Y
  Y[0] = np.log2(Y[0])
  Y[1] = np.log2(Y[1])
  print('normalize Y to log2')
  print(Y)

  
  pylab.rcParams.update(plot_params)  #更新自己的设置
  ax1 = plt.subplot(111)#figure1的子图1为ax1
  width = 0.18
  n = len(Y[0])
  labels = ['zero-copy', 'unified memory']
  ind = np.arange(n)                # the x locations for the groups
  # ind = np.arange(len(labels))                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + .5

  color_list = ['C1','C0','C2','C0',]
  hatch_list = ['xx','..','**','++']
  
  h_legs, e_legs = [], []
  for i, y in enumerate(Y):
    leg1 = ax1.bar(ind+(offset[i]*width),y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i], edgecolor='white')  
    leg2 = ax1.bar(ind+(offset[i]*width),y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
    h_legs.append(leg1)
    e_legs.append(leg2)

  plt.text(.47, .5, f'{np.power(2, Y[0][1]):.2f}',fontsize=8, fontweight='bold', color='C1')
  plt.text(1.03, .5, f'{np.power(2, Y[1][1]):.2f}',fontsize=8, fontweight='bold', color='C0')
  


  yticks = [0, 2, 4, 7, 10, 12]
  yticks = [0, 3, 6, 9, 12]
  yticks_str = np.power(2, yticks)
  yticks_str[0] = 0
  ylim = (0, 13)
  ax1.set_xticks(list(range(0, len(datasets))), datasets, rotation=25)
  ax1.set_yticks(yticks, yticks_str)
  ax1.set_ylim(*ylim)
  
  legs = [(x,y) for x,y in zip(h_legs, e_legs)]
  lines, labels = ax1.get_legend_handles_labels()

  plt.legend(lines, labels, ncol=3, bbox_to_anchor=anchor, columnspacing=1.5, handletextpad=.2, labelspacing=.1, handlelength=1)

  # ylabel = 'Epoch Time (s)'
  ylabel = 'Per-epoch Runtime (s)'
  ax1.set_ylabel(ylabel, labelpad=2.5)

  # Set the formatter
  axes = plt.gca()   # get current axes

  axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth']);###设置底部坐标轴的粗细
  axes.spines['left'].set_linewidth(plot_params['lines.linewidth']);####设置左边坐标轴的粗细
  axes.spines['right'].set_linewidth(plot_params['lines.linewidth']);###设置右边坐标轴的粗细
  axes.spines['top'].set_linewidth(plot_params['lines.linewidth']);####设置上部坐标轴的粗细

  
  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.\n')
  plt.close()





# def plot_bar_balance(plot_params, Y, labels, xlabel, ylabel, xticks, yticks, color_list, ylim, anchor=None, figpath=None):
def plot_memory(plot_params, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
    

  Y1 = [
    [4032 ,1734,2880,8800 ,5790 ,4382 ,7904], # zerocopy gpu memmory
    [4672 ,1862,3902,15106 ,15106 ,15106 ,15106], # unified gpu memory
  ]

  # to numpy
  for i in range(len(Y1)):
    Y1[i] = np.array(Y1[i])
  
  # normalized to GB
  Y1[0] = Y1[0] / 1024
  Y1[1] = Y1[1] / 1024
  print('normalize Y1 to GB', Y1)

  
  pylab.rcParams.update(plot_params)  #更新自己的设置
  ax1 = plt.subplot(111)#figure1的子图1为ax1
  width = 0.18
  n = len(Y1[0])
  labels = ['zero-copy', 'unified memory']
  ind = np.arange(n)                # the x locations for the groups
  # ind = np.arange(len(labels))                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + .5

  color_list = ['C1','C0','C2','C0',]
  hatch_list = ['xx','..','**','++']
  
  h_legs, e_legs = [], []
  for i, y in enumerate(Y1):
    # leg1 = ax1.scatter(ind, y, marker='+', color=color_list[i], s=100)
    # leg2 = ax1.scatter(ind, y, marker='x', color=color_list[i], s=100)
    leg1 = ax1.bar(ind+(offset[i]*width),y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i], edgecolor='white')  
    leg2 = ax1.bar(ind+(offset[i]*width),y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  

    h_legs.append(leg1)
    e_legs.append(leg2)


  yticks = [0, 3, 6, 9, 12, 15]
  ylim = (0, 15.6)
  ax1.set_xticks(list(range(0, len(datasets))), datasets, rotation=25)
  ax1.set_yticks(yticks)
  ax1.set_ylim(*ylim)


  legs = [(x,y) for x,y in zip(h_legs, e_legs)]
  lines, labels = ax1.get_legend_handles_labels()

  plt.legend(lines, labels, ncol=3, bbox_to_anchor=anchor, columnspacing=1.5, handletextpad=.2, labelspacing=.1, handlelength=1)

  ylabel = 'Memory Consumption (GB)'
  ylabel = 'Memory Cons. (GB)'
  ax1.set_ylabel(ylabel, labelpad=1)

  # Set the formatter
  axes = plt.gca()   # get current axes

  axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth']);###设置底部坐标轴的粗细
  axes.spines['left'].set_linewidth(plot_params['lines.linewidth']);####设置左边坐标轴的粗细
  axes.spines['right'].set_linewidth(plot_params['lines.linewidth']);###设置右边坐标轴的粗细
  axes.spines['top'].set_linewidth(plot_params['lines.linewidth']);####设置上部坐标轴的粗细

  
  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.\n')
  plt.close()


# def plot_bar_balance(plot_params, Y, labels, xlabel, ylabel, xticks, yticks, color_list, ylim, anchor=None, figpath=None):
def plot_time_memory(plot_params, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)

  Y = [
    [4.175, 0.359,3.998,96.379, 110.079, 92.822, 324.820], # zerocopy epoch time
    [3.251, 0.315,3.356, 802.387, 902.274, 1080.576, 4609.499], # unified epoch time
  ]
  # to numpy
  for i in range(len(Y)):
    Y[i] = np.array(Y[i])
  
  # normalize Y
  Y[0] = np.log2(Y[0])
  Y[1] = np.log2(Y[1])
  print('normalize Y to log2')
  print(Y)

  
  pylab.rcParams.update(plot_params)  #更新自己的设置
  ax1 = plt.subplot(111)#figure1的子图1为ax1
  width = 0.18
  n = len(Y[0])
  labels = ['zero-copy', 'unified memory']
  ind = np.arange(n)                # the x locations for the groups
  # ind = np.arange(len(labels))                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + .5

  color_list = ['C1','C0','C2','C0',]
  hatch_list = ['xx','..','**','++']
  
  h_legs, e_legs = [], []
  # text_pos = np.zeros_like(ind)
  # tot_vol = np.zeros_like(ind)
  for i, y in enumerate(Y):
    leg1 = ax1.bar(ind+(offset[i]*width),y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i], edgecolor='white')  
    leg2 = ax1.bar(ind+(offset[i]*width),y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
    # text_pos = np.maximum(text_pos, y)
    # tot_vol = tot_vol + y
    # print(text_pos, y)
    h_legs.append(leg1)
    e_legs.append(leg2)


  Y1 = [
    [4032 ,1734,2880,8800 ,5790 ,4382 ,7904], # zerocopy gpu memmory
    [4672 ,1862,3902,15106 ,15106 ,15106 ,15106], # unified gpu memory
  ]

  # to numpy
  for i in range(len(Y1)):
    Y1[i] = np.array(Y1[i])
  
  # normalized to GB
  Y1[0] = Y1[0] / 1024
  Y1[1] = Y1[1] / 1024
  print('normalize Y1 to GB', Y1)




  ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴
  for i, y in enumerate(Y1):
    print(ind, y)
    leg1 = ax2.scatter(ind, y, marker='+', color=color_list[i], s=100)
    leg2 = ax2.scatter(ind, y, marker='x', color=color_list[i], s=100)
    h_legs.append(leg1)
    e_legs.append(leg2)

  yticks = [0, 2, 6, 10, 12]
  yticks_str = np.power(2, yticks)
  yticks_str[0] = 0
  ylim = (0, 13)
  ax1.set_xticks(list(range(0, len(datasets))), datasets, rotation=25)
  ax1.set_yticks(yticks, yticks_str)
  ax1.set_ylim(*ylim)


  yticks1 = [0, 3, 6, 9, 12, 15]
  ylim1 = (0, 16)
  # ax1.set_xticks(list(range(0, len(datasets))), datasets, rotation=25)
  ax2.set_yticks(yticks1)
  ax2.set_ylim(*ylim1)

  legs = [(x,y) for x,y in zip(h_legs, e_legs)]
  lines, labels = ax1.get_legend_handles_labels()

  plt.legend(lines, labels, ncol=3, bbox_to_anchor=anchor, columnspacing=1.5, handletextpad=.2, labelspacing=.1, handlelength=1)

  ylabel = ['Epoch Time (s)', 'Memory Usage (GB)']
  ax1.set_ylabel(ylabel, labelpad=1)
  # ax2.set_ylabel(ylabel[1], labelpad=3) 


  # Set the formatter
  axes = plt.gca()   # get current axes

  axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth']);###设置底部坐标轴的粗细
  axes.spines['left'].set_linewidth(plot_params['lines.linewidth']);####设置左边坐标轴的粗细
  axes.spines['right'].set_linewidth(plot_params['lines.linewidth']);###设置右边坐标轴的粗细
  axes.spines['top'].set_linewidth(plot_params['lines.linewidth']);####设置上部坐标轴的粗细

  
  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.\n')
  plt.close()


def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)



if __name__ == '__main__':

    datasets = ['reddit', 'livejournal', 'lj-links','lj-large','enwiki']
    datasets = ['reddit', 'arxiv', 'products', 'livejournal', 'lj-links','lj-large','enwiki']
    params={
      'axes.labelsize': '11',
      'xtick.labelsize':'11',
      'ytick.labelsize':'11',
      'lines.linewidth': 1,
      # 'axes.linewidth': 10,
      # 'bars.linewidth': 100,
      'legend.fontsize': '11.5',
      'figure.figsize' : '4, 1.5',
      'legend.loc': 'upper center', #[]"upper right", "upper left"]
      'legend.frameon': False,
      # 'font.family': 'Arial'
      'font.family': 'Arial',
      'font.serif': 'Arial',
    }

    create_dir('./pdf')

    plot_time(params, anchor=(0.5, 1.28), figpath=f'./pdf/zero-unified-time.pdf')
    plot_memory(params, anchor=(0.5, 1.28), figpath=f'./pdf/zero-unified-memory.pdf')
    # plot_time_memory(params, anchor=(0.5, 1.28), figpath=f'./pdf/zero-unified-time-memory.pdf')
