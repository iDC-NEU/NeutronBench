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
from colored import fg

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


def print_different_optim(mode, datasets, log_path='../log/gpu-cache'):
  # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
  ret = {}
  # for optim in ['explicit', 'zerocopy', 'pipeline1', 'pipeline3', 'pipeline3-degree', 'pipeline3-sample']:
  for optim in ['explicit', 'zerocopy', 'unified', 'pipeline3', 'pipeline3-sample']:
    time_list = []
    for ds in datasets:
      log_file = f'{log_path}/{optim}/{ds}.log'
      time_list += parse_num(log_file, mode)
    ret[optim] = time_list
  return ret


def plot_bar(plot_params, Y, labels, xlabel, ylabel, xticks, anchor=None, figpath=None):
  
  # plt.rcParams.update(plt.rcParamsDefault)
  # print(plt.rcParams.keys())
  
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # print(plt.style.available)
  # plt.style.use('classic')
  # plt.style.use('"bmh"')
  # plt.style.use('"ggplot"')
  # plt.style.use('grayscale')
  # plt.style.use("seaborn-deep")
  # plt.style.use("seaborn-paper")
  # plt.style.use("seaborn-notebook")
  # plt.style.use("seaborn-poster")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  
  
  width = 0.16
  color_list = ['b', 'g', 'c', 'r', 'm']
  # color_list = ['#2a6ca6', '#419136', '#7c4e44', '#c4342b', '#f47a2d', ] 
  # color_list = ['#2a6ca6', '#419136', '#7c4e44', '#f47a2d', '#c4342b'] 
  # color_list = ['#2a6ca6', '#419136', '#f47a2d', '#c4342b', '#7c4e44', ] 
  # color_list = ['#223d4a','#236863','#259386','#80a870','#e4bd5f','#e36347']
  # color_list = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
  # color_list = ['#44a185','#83ae93','#ead8ba','#ebb561','#d4452c','#783923','',]
  # color_list = ['#44a185','#ead8ba','#ebb561','#d4452c','',]
  # color_list = ['#44a185','#83ae93','#ead8ba','#ebb561','#d4452c','#783923','',]
  # color_list = ['#44a185','#ead8ba','#ebb561','#d4452c','',]
  # color_list = ['#99093d','#f0633a','#fdd37a','#e6f397','#73c49a','#415aa4',]
  # color_list = ['#99093d','#f0633a','#73c49a','#415aa4',]
  # color_list = ['#bf90d2','#adce80','#83ceda','#eee28d',]
  # color_list = ['#d1a7ea','#92ccba','#b5d9f0','#f2deac',]
  # color_list = ['C4','C5','C6','C7',]
  # color_list = ['C6','C7','C8','C9',]
  # color_list = ['C0','C1','C2','C3',]
  # color_list = ['C3','C1','C2','C0','C4','C5']
  # hatch_list = ['xx','..','**','++','--','oo']
  color_list = ['C3','C1','C4','C2','C0','C5']
  hatch_list = ['xx','..','--','**','++','oo']
  # color_list = ['#be8fd0','#be8fd0','#d1a7ea','#f2deac',]

      
  n = len(Y[0])
  ind = np.arange(n)                # the x locations for the groups
  m = len(labels)
  offset = np.arange(m) - m / 2 + 0.5
  # gap = np.array([-2,-1,0,1,2]) * 0.02
  gap = np.array([-2,-1,0,1,2]) * 0

  h_legs, e_legs = [], []
  for i, y in enumerate(Y):
    leg1 = plt.bar(ind+(offset[i]*width + gap[i]),y,width,color=color_list[i], hatch=hatch_list[i], label=labels[i], linewidth=0, edgecolor='white')  
    leg2 = plt.bar(ind+(offset[i]*width + gap[i]),y,width,color='none', linewidth=1, edgecolor='black')  
    h_legs.append(leg1)
    e_legs.append(leg2)



  
  plt.xticks(np.arange(n), xticks, rotation=25)
  plt.yticks([0,1,2,3,4])
  legs = [(x, y) for x,y in zip(h_legs, e_legs)]
  leg = plt.legend(legs, labels, ncol=3, bbox_to_anchor=anchor, columnspacing=1, handletextpad=.25, labelspacing=.2,handlelength=1)

  leg_tex = leg.get_texts()
  leg_tex[2].set_color('blue')


    # text.set_color("red")

  # plt.legend(ncol=2, bbox_to_anchor=anchor, columnspacing=1.5, handletextpad=.45, labelspacing=.2)

  # plt.text(4.9, 1, 'Z for the zero-copy\noptimization, P for the\npipeline optimization, \nand C for the GPU\ncache optimization.',horizontalalignment='left', verticalalignment='bottom', fontsize=15)
  # plt.text(4.8, 1, 'Z for the zero-copy optimization.\nU for the unified memory optimization.\nP for the pipeline optimization.\nC for the GPU cache optimization.',horizontalalignment='left', verticalalignment='bottom', fontsize=11)#
  # import matplotlib
  # matplotlib.use('ps')
  # from matplotlib import rc
  # matplotlib.rc('text',usetex=True)
  # matplotlib.rc('text.latex', preamble='\\usepackage{color}')

  # https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/rainbow_text.html
  from matplotlib import transforms
  t = plt.gca().transData
  canvas = plt.gca().figure.canvas
  # text_list = ['Z for the zero-copy optimization.\n', 'U for the unified memory optimization.\n', '\nP for the pipeline optimization.\nC for the GPU cache optimization.']
  # text_list = ['Z for the zero-copy optimization.\n', 'U for the unified memory optimization.\n', '\nP for the pipeline optimization.\nC for the GPU cache optimization.']
  text_list = ['Z for the zero-copy optimization.', 'U for the unified memory optimization.','P for the pipeline optimization.', 'C for the GPU cache optimization.']
  text_color = ['black', 'blue', 'black', 'black']
  pre_h, pre_w = 0, 0
  for txt, c in zip(text_list[::-1], text_color[::-1]):
    text = plt.text(4.8, 1, txt,horizontalalignment='left', verticalalignment='bottom', fontsize=11, transform=t, color=c)#
    text.draw(canvas.get_renderer())
    ex = text.get_window_extent()
    t = transforms.offset_copy(text.get_transform(), y=ex.height, units='dots')

  plt.xlabel(xlabel)
  # plt.ylabel(ylabel)
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
  axes.tick_params(bottom=False)

  axes.set_axisbelow(True)

  figpath = 'plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', pad_inches=0, format='pdf')#bbox_inches='tight'会裁掉多余的白边
  plt.savefig(figpath, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()




def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)



if __name__ == '__main__':
  params={
    'axes.labelsize': '11',
    'xtick.labelsize':'11',
    'ytick.labelsize':'11',
    'lines.linewidth': .5,
    'hatch.linewidth' : 0.4,
    # 'legend.fontsize': '14.7',
    'legend.fontsize': '11',
    'figure.figsize' : '5, 1.5',
    'legend.loc': 'upper center', #[]"upper right", "upper left"]
    'legend.frameon': False,
    # 'font.family': 'Arial'
    'font.family': 'Arial',
    'font.serif': 'Arial',
  }

  datasets = ['ogbn-arxiv', 'ogbn-products', 'lj-links', 'lj-large', 'enwiki-links']
  datasets = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki-links']
  # ret = print_different_optim('one_epoch_time', datasets, '../log/gpu-cache-t4')
  ret = print_different_optim('one_epoch_time', datasets, './log')
  # to numpy array
  for k,v in ret.items():
    print(k, v)
    ret[k] = np.array(v)

  # rename key
  # for x, y in zip(['base', 'zerocopy', 'zerocopy+P', 'zerocopy+PC'], ['explicit', 'pipeline1', 'pipeline3', 'pipeline3-degree']):
  tmp_ret = {}
  # for x, y in zip(['base', 'zerocopy', 'zerocopy+P', 'zerocopy+PC'], ['explicit', 'zerocopy', 'pipeline3', 'pipeline3-degree']):
  for x, y in zip(['base', 'zero', 'unified', 'zero+P', 'zero+PC'], ['explicit', 'zerocopy', 'unified', 'pipeline3', 'pipeline3-sample']):
    tmp_ret[x] = ret[y]
  ret = tmp_ret

  print('\nafter rename:')
  for k,v in ret.items():
    print(k, v)

  # normalized
  for k in ['zero', 'unified', 'zero+P', 'zero+PC']:
    print(k, len(ret['base']), len(ret[k]))
    ret[k] = ret['base'] / ret[k]
  ret['base'] = np.ones_like(ret['base'])

  print('\nafter normalized:')
  for k,v in ret.items():
    print(k, v, f'avg: {np.average(v):.3f}', )


  xticks = ['reddit', 'livejournal', 'lj-links', 'lj-large', 'enwiki']
  ylabel = 'Normalized Speedup'
  xlabel = ''
  # labels = ['base', 'zerocopy', 'pipeline', 'pipeline+cache']
  labels = list(ret.keys())
  Y = list(ret.values())
  labels = ['base', 'zero', 'zero+P', 'zero+PC']
  labels = ['Baseline', 'Baseline+Z', 'Baseline+Z+P', 'Baseline+Z+P+C']
  labels = ['Baseline', 'Baseline+Z', 'Baseline+U', 'Baseline+Z+P', 'Baseline+Z+P+C']
  create_dir('./pdf')
  plot_bar(params, Y, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.43), figpath='./pdf/diff-optim-unified.pdf')

  create_dir('./diff-optim_txt')
  for i, x in enumerate(Y):
    with open (f'./diff-optim_txt/{labels[i]}_unified.txt', 'w') as f:
      for j,y in enumerate(x):
        f.write(str(j) + ' ' + str(y) + '\n')

  # plot_bar(params, Y, labels, xlabel, ylabel, xticks, anchor=(0.5, 1.145), figpath='diff-optim.pdf')
