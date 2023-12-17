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

def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)


def plot_bar_balance(plot_params, labels, xlabel, ylabel, xticks, color_list, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
  # print(plt.rcParams.keys())
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # print(plt.style.available)
  # plt.style.use("seaborn-poster")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  width = 0.25
  color_list = ['C3','C1','C2','C0','C4','C5']
  hatch_list = ['xx','..','**','++','--','oo']
      
  ax1 = plt.subplot(131)#figure1的子图1为ax1
  yticks = [0, 6, 12, 18]
  ylim = (0, 23)
  Y = [0.0031,15.855,18.7706,17.725,9.3801,0.0164]
  h_legs, e_legs = [], []
  ax1.set_xticks([])
  ax1.set_yticks(yticks, yticks)
  ax1.set_ylim(*ylim)
  ax1.set_ylabel(ylabel, labelpad=3,fontsize=11.1)
  ax1.set_title('(a) Amazon',x=0.5,y=-.58, fontsize=14)
  ax1.set_xticks(list(range(6)), labels, rotation=35, fontsize=7.5)
  

  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    leg1 = ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i] ,edgecolor='white')  
    leg2 = ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
    h_legs.append(leg1)
    e_legs.append(leg2)
  ax1.text(-.3, 1,f'{Y[0]:.3f}',fontsize=8)
  ax1.text(5-.7, 1,f'{Y[4]:.3f}',fontsize=8)

  
  legs = [(x,y) for x,y in zip(h_legs, e_legs)]
  lines, labels = ax1.get_legend_handles_labels()
  # plt.legend(lines, labels , ncol=6, bbox_to_anchor=(1.78, 1.38), columnspacing=1.4, handletextpad=.2, labelspacing=.1, handlelength=1)

  
  ax1 = plt.subplot(133)#figure1的子图1为ax1
  yticks = [0, 2, 4]
  ylim = (0, 5.5)
  Y = [0.0015,3.8229,3.9453,3.9744,2.7279,0.0168]
  ax1.set_xticks([])
  ax1.set_yticks(yticks, yticks)
  ax1.set_ylim(*ylim)
  ax1.set_ylabel(ylabel, labelpad=3,fontsize=11.1)
  ax1.set_title('(c) Reddit',x=0.5,y=-.58, fontsize=14)
  ax1.set_xticks(list(range(6)), labels, rotation=35, fontsize=7.5)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i] ,edgecolor='white')  
    ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
  ax1.text(-.3, .3,f'{Y[0]:.3f}',fontsize=8)
  ax1.text(5-.7, .3,f'{Y[4]:.3f}',fontsize=8)


  ax1 = plt.subplot(132)#figure1的子图1为ax1
  yticks = [0, 2, 4]
  ylim = (0, 5.5)
  Y = [0.0023,4.4177,4.4726,4.8894,0.6426,0.0156]
  ax1.set_xticks([])
  ax1.set_yticks(yticks, yticks)
  ax1.set_ylim(*ylim)
  ax1.set_ylabel(ylabel, labelpad=3,fontsize=11.1)
  ax1.set_title('(b) Products',x=0.5,y=-.58, fontsize=14)
  ax1.set_xticks(list(range(6)), labels, rotation=35, fontsize=7.5)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i] ,edgecolor='white')  
    ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
  ax1.text(-.3, .3,f'{Y[0]:.3f}',fontsize=8)
  ax1.text(5-.7, .3,f'{Y[4]:.3f}',fontsize=8)

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


def plot_bar_balance1(plot_params, labels, xlabel, ylabel, xticks, color_list, anchor=None, figpath=None):
  plt.rcParams.update(plt.rcParamsDefault)
  # print(plt.rcParams.keys())
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # print(plt.style.available)
  # plt.style.use("seaborn-poster")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  width = 0.25
  color_list = ['C3','C1','C2','C0','C4','C5']
  hatch_list = ['xx','..','**','++','--','oo']
      
  ax1 = plt.subplot(131)#figure1的子图1为ax1
  yticks = [0, 6, 12, 18]
  ylim = (0, 23)
  Y = [0.0031,15.855,18.7706,17.725,9.3801,0.0164]
  h_legs, e_legs = [], []
  ax1.set_xticks([])
  ax1.set_yticks(yticks, yticks)
  ax1.set_ylim(*ylim)
  ax1.set_ylabel(ylabel, labelpad=2.3,fontsize=plot_params['axes.labelsize'])
  # ax1.set_ylabel(ylabel, labelpad=2.3)
  ax1.set_title('(a) Amazon',x=0.5,y=-.25, fontsize=14)
  # ax1.set_xticks(list(range(6)), labels, rotation=35, fontsize=7.5)
  

  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    leg1 = ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i] ,edgecolor='white')  
    leg2 = ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
    h_legs.append(leg1)
    e_legs.append(leg2)
  ax1.text(-.3, 1,f'{Y[0]:.3f}',fontsize=8,color='C3')
  ax1.text(5-.7, 1,f'{Y[4]:.3f}',fontsize=8,color='C5')

  
  legs = [(x,y) for x,y in zip(h_legs, e_legs)]
  lines, labels = ax1.get_legend_handles_labels()
  plt.legend(lines, labels , ncol=6, bbox_to_anchor=anchor, columnspacing=1.4, handletextpad=.2, labelspacing=.1, handlelength=1)

  # (1.78, 1.38)
  ax1 = plt.subplot(133)#figure1的子图1为ax1
  yticks = [0, 2, 4]
  ylim = (0, 5.5)
  Y = [0.0015,3.8229,3.9453,3.9744,2.7279,0.0168]
  ax1.set_xticks([])
  ax1.set_yticks(yticks, yticks)
  ax1.set_ylim(*ylim)
  ax1.set_ylabel(ylabel, labelpad=2.3,fontsize=plot_params['axes.labelsize'])
  ax1.set_title('(c) Reddit',x=0.5,y=-.25, fontsize=14)
  # ax1.set_xticks(list(range(6)), labels, rotation=35, fontsize=7.5)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i] ,edgecolor='white')  
    ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
  ax1.text(-.3, .3,f'{Y[0]:.3f}',fontsize=8,color='C3')
  ax1.text(5-.7, .3,f'{Y[4]:.3f}',fontsize=8,color='C5')


  ax1 = plt.subplot(132)#figure1的子图1为ax1
  yticks = [0, 2, 4]
  ylim = (0, 5.5)
  Y = [0.0023,4.4177,4.4726,4.8894,0.6426,0.0156]
  ax1.set_xticks([])
  ax1.set_yticks(yticks, yticks)
  ax1.set_ylim(*ylim)
  ax1.set_ylabel(ylabel, labelpad=2.3,fontsize=plot_params['axes.labelsize'])
  ax1.set_title('(b) Products',x=0.5,y=-.25, fontsize=14)
  # ax1.set_xticks(list(range(6)), labels, rotation=35, fontsize=7.5)
  for i, y in enumerate(Y):
    # plt.bar(ind+(offset[i]*width),y,width, label=labels[i])  
    ax1.bar(i,y,width,color=color_list[i], hatch=hatch_list[i] ,edgecolor='white')  
    ax1.bar(i,y,width,color='none', lw=plot_params['lines.linewidth'], edgecolor='black')  
  ax1.text(-.3, .3,f'{Y[0]:.3f}',fontsize=8,color='C3')
  ax1.text(5-.7, .3,f'{Y[4]:.3f}',fontsize=8,color='C5')

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



def plot_comp_comm(dataset, num_parts):
  params={
    'axes.labelsize': '11',
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

  params1={
    'axes.labelsize': '10',
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

  # labels = ['metis1', 'metis2', 'metis4', 'dgl', 'pagraph', 'bytegnn', 'hash']
  labels = ['Hash', 'Metis*', 'DistDGL', 'SALIENT++',  'ByteGNN']
  labels = ['Hash', 'Metis-V', 'Metis-VE', 'Metis-VET',  'Stream-B']
  labels = ['Hash', 'Metis-V', 'Metis-VE', 'Metis-VET', 'Stream-V', 'Stream-B']
  for ds in datasets:
    color_list = ['#bdddf2','#8e8e8e','#f3ec8a','#bfd2bb','#d394de','#b0dbce',]
    xlabel = '# partition ID'
    # ylabel = 'Communication load (GB)'
    # ylabel = 'Graph structure and features\n of communication (GB)'
    # ylabel = 'Memory consumption (GB)'
    ylabel = 'Memory Cons. (GB)'
    
    xticks = [f'{x+1}' for x in range(num_parts)]
    

    create_dir('./pdf')    
    plot_bar_balance(params, labels, xlabel, ylabel, xticks, color_list, anchor=(1.78, 1.38), figpath=f'./pdf/partition-memory.pdf')
    plot_bar_balance1(params1, labels, xlabel, ylabel, xticks, color_list, anchor=(1.72, 1.3), figpath=f'./pdf/partition-memory1.pdf')
    


if __name__ == '__main__':
    # print(plt.rcParams.keys())
    feat_dims = {
      'amazon': 200,
      'reddit': 200,
      'ogbn-arxiv': 128,
      'ogbn-products': 602,
    }

    datasets = ['reddit','amazon', 'ogbn-products']
    datasets = ['amazon']
    num_parts = 4

    # plot_data_access(datasets, num_parts)
    plot_comp_comm(datasets, num_parts)