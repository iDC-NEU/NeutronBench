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




def plot_bar(plot_params, Y, labels, xlabel, ylabel, xticks, yticks, color_list, anchor=None, figpath=None):
  
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
  
  
  width = 0.13
  # color_list = ['#2a6ca6', '#419136', '#7c4e44', '#c4342b', '#f47a2d', '#EA6632', '#f47a2d', ]
  # color_list = ['#b35806','#f1a340','#fee0b6','#d8daeb','#998ec3','#542788']

  n = len(Y[0])
  ind = np.arange(n)                # the x locations for the groups
  # ind = [0,1.2,2.4,3.6]                # the x locations for the groups
  # ind = [0,1.3,2.6,3.9]                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + 0.5

  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i], linewidth=params['lines.linewidth'], edgecolor='black')  
  
  # plt.xticks(np.arange(n) + (len(labels)/2-0.5)*width, xticks)
  plt.xticks(ind, xticks)
  plt.yticks(yticks, yticks)
  
  # plt.legend(ncol=len(labels)//2, bbox_to_anchor=anchor)
  # plt.legend(nrow=len(labels)//2, bbox_to_anchor=anchor)

  plt.legend(ncol=3, bbox_to_anchor=anchor, columnspacing=1.2, handletextpad=.35, labelspacing=.2, handlelength=1.5) # ,markerscale=10


  plt.xlabel(xlabel)
  plt.ylabel(ylabel, labelpad=2)



  # Set the formatter
  axes = plt.gca()   # get current axes
  # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  # ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes.grid(axis='y', linestyle='-.')
  # axes.grid(axis='x')

  axes.spines['bottom'].set_linewidth(params['lines.linewidth']);###设置底部坐标轴的粗细
  axes.spines['left'].set_linewidth(params['lines.linewidth']);####设置左边坐标轴的粗细
  axes.spines['right'].set_linewidth(params['lines.linewidth']);###设置右边坐标轴的粗细
  axes.spines['top'].set_linewidth(params['lines.linewidth']);####设置上部坐标轴的粗细

  


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()



def plot_bar_balance1(plot_params, Y, labels, xlabel, ylabel, xticks, yticks, color_list, anchor=None, figpath=None):
  
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
  
  
  width = 0.2
  # color_list = ['#2a6ca6', '#419136', '#7c4e44', '#c4342b', '#f47a2d', '#EA6632', '#f47a2d', ]
  # color_list = ['#b35806','#f1a340','#fee0b6','#d8daeb','#998ec3','#542788']

  n = len(Y[0])
  ind = np.arange(n)                # the x locations for the groups
  # ind = [0,1.2,2.4,3.6]                # the x locations for the groups
  # ind = [0,1.3,2.6,3.9]                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + 0.5

  # for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    # plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i], linewidth=params['lines.linewidth'], edgecolor='black')  
  
  labels_p = ['part1', 'part2', 'part3', ]
  for i,y in enumerate(Y):
    off = np.linspace(0, len(ind) * width, len(ind) + 1)[:-1] - 1.5 * width
    off = off + np.ones(len(ind)) * i
    # print(i,ind,  off)
    plt.bar(off,y,width,color=color_list[i], label=labels[i], linewidth=params['lines.linewidth'], edgecolor='black')  
    # plt.bar(ind+(offset[i]*width),y,width,color=color_list[i], label=labels[i], linewidth=params['lines.linewidth'], edgecolor='black')  


  # plt.xticks(np.arange(n) + (len(labels)/2-0.5)*width, xticks)
  plt.xticks(list(range(0, len(labels))), labels, rotation=25)
  plt.yticks(yticks, yticks)
  
  # plt.legend(ncol=len(labels)//2, bbox_to_anchor=anchor)
  # plt.legend(nrow=len(labels)//2, bbox_to_anchor=anchor)

  plt.legend(ncol=3, bbox_to_anchor=anchor, columnspacing=1.2, handletextpad=.35, labelspacing=.2, handlelength=1.5) # ,markerscale=10


  plt.xlabel(xlabel)
  plt.ylabel(ylabel, labelpad=2)



  # Set the formatter
  axes = plt.gca()   # get current axes
  # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  # ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes.grid(axis='y', linestyle='-.')
  # axes.grid(axis='x')

  axes.spines['bottom'].set_linewidth(params['lines.linewidth']);###设置底部坐标轴的粗细
  axes.spines['left'].set_linewidth(params['lines.linewidth']);####设置左边坐标轴的粗细
  axes.spines['right'].set_linewidth(params['lines.linewidth']);###设置右边坐标轴的粗细
  axes.spines['top'].set_linewidth(params['lines.linewidth']);####设置上部坐标轴的粗细

  


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()



def plot_bar_balance(plot_params, labels, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  width = 0.25
  color_list = ['C3','C1','C2','C0','C4','C5']
  hatch_list = ['xx','..','**','++','--','oo']

  ax1 = plt.subplot(131)#figure1的子图1为ax1
  yticks = [[0, 7, 14, 21], [0, 6, 12, 18, 24]]
  ylim = [(0, 25), (0, 23)]
  Y = [20.05,11.23,9.83,9.73,20.13]
  Y = [20.05,11.23,9.83,9.73,19.73,20.13]
  h_legs, e_legs = [], []
  ax1.set_xticks([])
  ax1.set_yticks(yticks[0], yticks[0])
  ax1.set_ylim(*ylim[0])
  ylabel = 'Epoch time (s)'
  ax1.set_ylabel(ylabel, labelpad=3)
  ax1.set_title('(a) Amazon',x=0.5,y=-.46, fontsize=14)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    leg1 = ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i] ,edgecolor='white')  
    leg2 = ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
    # text_pos = np.maximum(text_pos, y)
    # tot_vol = tot_vol + y
    # print(text_pos, y)
    h_legs.append(leg1)
    e_legs.append(leg2)
  
  legs = [(x,y) for x,y in zip(h_legs, e_legs)]
  lines, labels = ax1.get_legend_handles_labels()
  plt.legend(lines, labels , ncol=6, bbox_to_anchor=(1.78, 1.38), columnspacing=1.4, handletextpad=.2, labelspacing=.1, handlelength=1)

  
  ax1 = plt.subplot(133)#figure1的子图1为ax1
  yticks = [[0, 3, 6, 9],[0, 1, 2, 3]]
  ylim = [(0, 8),(0, 3)]
  Y = [5.6135,2.5836,2.7740,2.7105,4.6549]
  Y = [5.6135,2.5836,2.7740,2.7105,5.0, 4.6549]
  ax1.set_xticks([])
  ax1.set_yticks(yticks[0], yticks[0])
  ax1.set_ylim(*ylim[0])
  ax1.set_ylabel(ylabel, labelpad=3)
  ax1.set_title('(c) Reddit',x=0.5,y=-.46, fontsize=14)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i] ,edgecolor='white')  
    ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  


  ax1 = plt.subplot(132)#figure1的子图1为ax1
  yticks = [[0, 3, 6],[0, 12, 24, 36]]
  ylim = [(0, 8),(0, 43)]
  Y = [5.1903,2.7366,2.8637,2.8210,3.9445]
  Y = [5.1903,2.7366,2.8637,2.8210,3.825,3.9445]
  ax1.set_xticks([])
  ax1.set_yticks(yticks[0], yticks[0])
  ax1.set_ylim(*ylim[0])
  ax1.set_ylabel(ylabel, labelpad=3)
  ax1.set_title('(b) Products',x=0.5,y=-.46, fontsize=14)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i] ,edgecolor='white')  
    ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  


  axes = plt.gca()   # get current axes
  # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
  # ticks_fmt = mtick.FormatStrFormatter(fmt)   
  # axes.yaxis.set_major_formatter(ticks_fmt) # set % format to ystick.
  # axes.grid(axis='y', linestyle='-.')
  # axes.grid(axis='x')

  axes.spines['bottom'].set_linewidth(plot_params['lines.linewidth']);###设置底部坐标轴的粗细
  axes.spines['left'].set_linewidth(plot_params['lines.linewidth']);####设置左边坐标轴的粗细
  axes.spines['right'].set_linewidth(plot_params['lines.linewidth']);###设置右边坐标轴的粗细
  axes.spines['top'].set_linewidth(plot_params['lines.linewidth']);####设置上部坐标轴的粗细

  plt.subplots_adjust(wspace=.3, hspace =0)#调整子图间距
  


  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.\n')
  plt.close()


def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)


def plot_comp_comm(dataset, num_parts):
  params={
    'axes.labelsize': '12',
    'xtick.labelsize':'12',
    'ytick.labelsize':'12',
    'lines.linewidth': 1,
    # 'axes.linewidth': 12,
    # 'bars.linewidth': 120,
    'legend.fontsize': '12',
    'figure.figsize' : '8, 1.5',
    'legend.loc': 'upper center', #[]"upper right", "upper left"]
    'legend.frameon': False,
    # 'font.family': 'Arial'
    'font.family': 'Arial',
    'font.serif': 'Arial',
  }

  modes = [ 'hash', 'metis1', 'dgl', 'metis4',  'bytegnn']
  # modes = [ 'hash', 'metis1', 'dgl', 'metis4', 'pagraph', 'bytegnn']
  labels = ['Hash', 'Metis-V', 'Metis-VE', 'Metis-VET', 'Stream-V', 'Stream-B']
  # for ds in datasets:
  #   color_list = ['#bdddf2','#8e8e8e','#f3ec8a','#bfd2bb','#d394de','#b0dbce',]
  xlabel = '# partition ID'
  #   # ylabel = 'Communication load (GB)'
  #   # ylabel = 'Graph structure and features\n of communication (GB)'
  # ylabel = 'Epoch time (s)'
    
  #   xticks = [f'{x+1}' for x in range(num_parts)]
  #   if ds == 'amazon':
  #     yticks = [[0, 7, 14, 21], [0, 6, 12, 18, 24]]
  #     ylim = [(0, 25), (0, 23)]
  #     all_comm_load = [[20.05,11.23,9.83,9.73,19.73,20.13]]
  #   elif ds == 'reddit':
  #     yticks = [[0, 3, 6, 9],[0, 1, 2, 3]]
  #     ylim = [(0, 8),(0, 3)]
  #     all_comm_load = [[5.6135,2.5836,2.7740,2.7105,5.0, 4.6549]]
  #   elif ds == 'ogbn-products':
  #     yticks = [[0, 3, 6],[0, 12, 24, 36]]
  #     ylim = [(0, 8),(0, 43)]
  #     all_comm_load = [[5.1903,2.7366,2.8637,2.8210,3.825,3.9445]]
  #   elif ds == 'ogbn-arxiv':
  #     yticks = [[0, .1, .2, .3],[0, .1, .2, .3]]
  #     ylim = [(0, .3),(0, .3)]
  #     all_comm_load = [[5.6135,2.5836,2.7740,2.7105,4.6549]]

  create_dir('./pdf')
  plot_bar_balance(params, labels, figpath=f'./pdf/epoch-time.pdf')


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
    datasets = ['reddit','amazon', 'ogbn-products']
    num_parts = 4

    # plot_data_access(datasets, num_parts)
    plot_comp_comm(datasets, num_parts)