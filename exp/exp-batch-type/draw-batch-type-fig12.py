import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import numpy as np
import matplotlib.ticker as mtick
import os
import re

def create_dir(path=None):
    if path and not os.path.exists(path):
        os.makedirs(path)

# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, X, Y, labels, xlabel, ylabel, xticks, yticks, xlim, ylim, figpath=None):
  # plt.style.use("seaborn-deep")
  pylab.rcParams.update(plot_params)  #更新自己的设置
  plt.rcParams['pdf.fonttype'] = 42

  makrer_list = [ 'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
  color_list = ['C0','C1','C0','C2','C4','C3',]
  # fig1 = plt.figure(1)
  axes1 = plt.subplot(111) #figure1的子图1为axes1
  for i, (x, y) in enumerate(zip(X, Y)):
    y = np.array(y) * 100
    plt.plot(x, y, label=labels[i], color=color_list[i])
    plt.plot(x[-1], y[-1], color=color_list[i],marker='x', markersize=6)
  axes1.set_yticks(yticks)
  axes1.set_xticks(xticks)
  ############################
  print(y_lim, yticks)
  axes1.set_ylim(ylim[0], ylim[1])
  axes1.set_xlim(xlim[0], xlim[1])

  # axes1.spines['bottom'].set_linewidth(myparams['lines.linewidth']);###设置底部坐标轴的粗细
  # axes1.spines['left'].set_linewidth(myparams['lines.linewidth']);####设置左边坐标轴的粗细
  # axes1.spines['right'].set_linewidth(myparams['lines.linewidth']);###设置右边坐标轴的粗细
  # axes1.spines['top'].set_linewidth(myparams['lines.linewidth']);####设置上部坐标轴的粗细


  # plt.legend(ncol=2)
  # plt.legend(ncol=3, bbox_to_anchor=(1.08, 1.28), columnspacing=0.5, handletextpad=.1, labelspacing=.1, handlelength=1.5)
  plt.legend(ncol=4, columnspacing=1, handletextpad=.3, labelspacing=.1, handlelength=.7, bbox_to_anchor=(0.5, 1.26))


  plt.ylabel(ylabel,labelpad=2)
  plt.xlabel(xlabel)

  figpath = './log/batch-size/reddit-exp5/plot.pdf' if not figpath else figpath
  # plt.savefig(figpath, dpi=1000,  bbox_inches='tight', pad_inches=0, format='pdf')#bbox_inches='tight'会裁掉多余的白边
  plt.savefig(figpath, dpi=1000,  bbox_inches='tight', pad_inches=0, format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()


# 每隔time_skip对acc取一个平均值
def split_list(X, Y, time_skip):
    retX, retY = [], []
    for arrx,arry in zip(X, Y):
        tmpx, tmpy = [], []
        pre, idx = 0, 0
        for i in range(len(arrx)):
            x, y = arrx[i], arry[i]
            if x >= idx * time_skip:
                tmpx.append(x)
                tmpy.append(np.average(arry[pre : i + 1]))
                pre = i + 1
                idx += 1
        if pre < len(arrx):
            tmpx.append(arrx[-1])
            tmpy.append(np.average(arry[pre : ]))

        retX.append(tmpx)
        retY.append(tmpy)
    return retX, retY


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
                nums = re.findall(r"\d+\.?\d*", line[line.find(mode) :])
                ret.append(float(nums[0]))
    return ret



# 每隔time_skip对acc取一个平均值
def split_list_best(X, Y, best_acc):
    retX, retY = [], []
    for arrx,arry in zip(X, Y):
        tmpx, tmpy = [], []
        flag = False
        for i in range(len(arrx)):
            x, y = arrx[i], arry[i]
            tmpx.append(x)
            tmpy.append(y)
            if y >= best_acc:
              # tmpy[-1] = best_acc
              flag = True
              break
        if flag:
          retX.append(tmpx)
          retY.append(tmpy)
        else:
          max_acc = -1
          max_idx = -1

          for i in range(len(arrx)):
            x, y = arrx[i], arry[i]
            if y > max_acc:
              max_acc = y
              max_idx = i

          retX.append(arrx[:max_idx])
          retY.append(arry[:max_idx])


    return retX, retY




def get_target_acc_time(target_acc, accs, times):
  target_accs, target_time = [], []
  for one_acc_list, one_time_list in zip(accs, times):
    flag = False
    max_acc = -1
    max_time = -1
    for x,y in zip(one_acc_list, one_time_list):
      if x > max_acc:
        max_acc = x
        max_time = y
      if x >= target_acc:
        flag = True
        target_accs.append(x)
        target_time.append(y)
        break
    if not flag:
      assert max_acc == max(one_acc_list)
      # for x,y in zip(one_acc_list, one_time_list):
      #   if x == max_acc:
      target_accs.append(max_acc)
      target_time.append(max_time)
  assert len(target_accs) == len(target_time) == len(accs)
  return target_accs, target_time






if __name__ == '__main__':

  myparams={
        'axes.labelsize': '10',
        'xtick.labelsize':'10',
        'ytick.labelsize':'10',
        'lines.linewidth': 1,
        # 'axes.linewidth': 10,
        # 'bars.linewidth': 100,
        'legend.fontsize': '10',
        'figure.figsize' : '2.3, 1.3',
        'legend.loc': 'upper center', #[]"upper right", "upper left"]
        # 'legend.loc': 'best', #[]"upper right", "upper left"]
        'legend.frameon': False,
        # 'font.family': 'Arial'
        # 'font.family': 'Times New Roman',
        
        'font.family': 'Arial',
        'font.serif': 'Arial',
      }
      
  system = ('hash', 'metis1', 'metis2', 'metis4', 'bytegnn')
  # labels = ('P3', 'Metis*', 'DistDGL', 'SALIENT++', 'ByteGNN')
  # datasets = ['amazon', 'ogbn-products', 'reddit']
  datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit']
  datasets = ['ogbn-products', 'reddit']
  # batch_type = ['shuffle', 'low', 'high', 'metis']
  # labels = ['random', 'low', 'high', 'metis']
  batch_type = ['shuffle', 'metis']
  labels = ['Random', 'Cluster-based']

  log_path = './log'
  for ds in datasets:
    val_acc = []
    run_time = []
    for bt in batch_type:
      # log_file = f'./old-log/{ds}/{ds}-{bt}.log'
      # log_file = f'{log_path}/{ds}-time/{ds}-{bt}.log'
      log_file = f'{log_path}/{ds}-0.001/{ds}-{bt}-0.001.log'
      print(log_file)
      # val_acc.append(parse_num(log_file, 'All Val ACC '))
      val_acc.append(parse_num(log_file, 'val_acc '))
      run_time.append(parse_num(log_file, 'gcn_run_time '))

    
    # partiion_accs = [max(x) for x in val_acc]
    # print(system)
    # print('top val acc', partiion_accs)
    # min_accs = min(partiion_accs)
    # print(ds, 'mini acc', min_accs)

    # if ds == 'reddit':
    #   min_accs = 0.9614

    # run_time,val_acc = split_list_best(run_time, val_acc, min_accs)
    # run_time,val_acc = split_list(run_time, val_acc, 6)

    # print('acc:', [x[-1] for x in val_acc])
    # print('time:', [x[-1] for x in run_time])
    # print()
    
    

    xlabel = 'Run Time (s)'
    ylabel = 'Accuracy (%)'

    if ds == 'reddit':
      # y_lim = (89, 97)
      # yticks = np.linspace(90, 96, 3)
      # x_lim = (0, 400)
      # xticks = np.linspace(*x_lim, 5)
      # y_lim = (70, 96)
      y_lim = (89, 95)
      yticks = np.linspace(*y_lim, 3)
      x_lim = (0, 60)
      xticks = np.linspace(*x_lim, 5)
    elif ds == 'ogbn-products':
      # y_lim = (70, 92)
      y_lim = (76, 92)
      yticks = np.linspace(*y_lim, 3)
      x_lim = (0, 60)
      xticks = np.linspace(*x_lim, 5)
    elif ds == 'ogbn-arxiv':
      y_lim = (60, 70)
      yticks = np.linspace(*y_lim, 3)
      x_lim = (0, 50)
      xticks = np.linspace(*x_lim, 5)
    elif ds == 'amazon':
      y_lim = (55, 65)
      yticks = np.linspace(*y_lim, 3)
      x_lim = (0, 2000)
      xticks = np.linspace(*x_lim, 5)
    else:
      assert False
    

    plot_line(myparams, run_time, val_acc, labels, xlabel, ylabel, xticks, yticks, x_lim, y_lim, f'./pdf/batchtype-{ds}-0.001.pdf')
    # plot_line(myparams, run_time_skip, val_acc_skip, labels, xlabel, ylabel, xticks, yticks, (0,300), y_lim, f'./nts-old-sample/mix-{ds}.pdf')