# -*- coding: utf-8 -*-
# @Time    : 2020/9/5 23:04
# @Author  : Hui Wang

from ast import parse
from calendar import c
import numpy as np
import matplotlib
import os, re
import itertools

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


def plot_bar(x_name, y_name, datas, labels, filename='bar.png', color=None):
  assert (len(datas[0]) == len(x_name))
  assert (len(labels) == len(datas))
  # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
  # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
  # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]  

  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  plt.figure(figsize=(7, 4))
  # linestyle = "-"
  x = np.arange(7)
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
  up = 21.5
  plt.ylim(low, up)
  # plt.xlabel("Amount of Data", fontsize=15)
  # plt.ylabel(f"Time (s)", fontsize=20)
  plt.ylabel(y_name, fontsize=20)
  # labels = ['GraphScope', 'NTS']

  # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
  if color is None:
    color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue']
  

  for i, data in enumerate(datas):
    plt.bar(x + width * i, data, width=width, color=color[i], edgecolor='w')  # , edgecolor='k',)
    

  # plt.bar(x, aligraph, width=width, color='blue', edgecolor='w')  # , edgecolor='k',)
  # plt.bar(x + width, nts, width=width, color='green', edgecolor='w')  # , edgecolor='k',)

  # for i in range(len(x)):
  #   text = 'OOM' if aligraph[i] == 0 else aligraph[i]
  #   plt.text(i-width+0.01, aligraph[i]+0.1, text, fontsize=10)

  #   text = 'OOM' if nts[i] == 0 else nts[i]
  #   plt.text(i+0.01, nts[i]+0.1, text, fontsize=10)      

  # plt.bar(x + 2*width, model3, width=width, color='orange', edgecolor='w')  # , edgecolor='k',)
  # plt.bar(x + 3*width, Ours, width=width, color='tomato', edgecolor='w')  # , edgecolor='k',)

  # plt.xticks(x+0.5*width, labels=x_name, fontsize=15)
  plt.xticks(x + offset, labels=x_name, fontsize=15)
  # plt.xticks(x+total_width/2, labels=x_name, fontsize=15)

  y_lables = ['0', '4', '8', '16', '20']
  y_ticks = [float(i) for i in y_lables]
  x_lables = ['0', '1', '2', '3', '4']
  x_ticks = [float(i) for i in x_lables]
  # plt.yscale('linear')
  # y_ticks = [0.25, 0.30, 0.35, 0.40, 0.45]
  # y_lables = ['0.25', '0.30', '0.35', '0.40', '0.45']
  plt.yticks(np.array(y_ticks), y_lables, fontsize=20)#bbox_to_anchor=(0.30, 1)
  # plt.xticks(np.array(x_ticks), x_lables, fontsize=20)#bbox_to_anchor=(0.30, 1)
  plt.legend(labels=labels, ncol=2,
              prop={'size': 14})

  plt.tight_layout()
  plt.savefig(f'./{filename}', format='png')
  plt.show()
  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

# x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
# labels = ['GraphScope', 'NTS']
# aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
# nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435] 
# plot_bar(x_name, 'Time (s)', [aligraph, aligraph], labels, 'xx.png')

def get_time_acc(accs, times, best, early_stop=True):
  # print(times)
  if not isinstance(times, list):
    times = [times for _ in range(len(accs))]
  # print(times)
  # assert(False)
  # print(times)
  # print(accs,best, 'find')
  idx = len(accs)
  
  if early_stop:
    idx = 0
    while accs[idx] < best:
      idx += 1
  # idx = bisect.bisect(accs, best)
  idx = min(idx+10, len(accs))
  accs_ret = accs[:idx+1]
  times_ret = list(itertools.accumulate(times[:idx+1]))
  # print(len(accs_ret))
  # print(accs_ret[-1], best)
  # assert accs_ret[-1] >= best
  assert len(accs_ret) == len(times_ret)
  # print(len(accs_ret))
  return [times_ret, accs_ret]


# ali_gcn_cpu_pubmed_best=0.7896197160110984
# ali_gcn_cpu_pubmed_test=[0.4529133344214134, 0.47127468581687615, 0.39962461237147057, 0.43814264729884117, 0.5816060062020565, 0.6419128447853762, 0.7018932593438877, 0.6943039007670965, 0.7111147380447201, 0.732577117675861, 0.7058919536477885, 0.7280887873347478, 0.7204178227517545, 0.7341276317937, 0.747102986779827, 0.7443283825689571, 0.7563244654806593, 0.7633425820140363, 0.7655459441814918, 0.7692182144605842, 0.7684837604047657, 0.778847723192427, 0.7787661171862249, 0.7748490288885262, 0.7780316631304064, 0.7833360535335401, 0.7805614493226701, 0.7835808715521463, 0.7845601436265709, 0.7827648114901257, 0.784641749632773, 0.7804798433164681, 0.78611065774441, 0.7835808715521463, 0.7800718132854578, 0.7888036559490779, 0.7896197160110984, 0.7843153256079648, 0.7792557532234372, 0.7866002937816223, 0.7888036559490779, 0.784723355638975, 0.7784396931614167, 0.781785539415701, 0.7830912355149339, 0.7843153256079648, 0.7841521135955606, 0.7859474457320059, 0.7738697568141015, 0.7813775093846907, 0.7890484739676841, 0.777460421086992, 0.7830912355149339, 0.7830096295087319, 0.784641749632773, 0.7883140199118656, 0.7848049616451771, 0.7859474457320059, 0.7843969316141668, 0.7819487514281052, 0.7796637832544475, 0.7786845111800228, 0.7848049616451771, 0.7851313856699853, 0.7661987922311082, 0.7790925412110331, 0.7845601436265709, 0.7790109352048311, 0.7729720907458789, 0.7799902072792557, 0.7790925412110331, 0.7755834829443446, 0.77460421086992, 0.7760731189815571, 0.7795821772482455, 0.77737881508079, 0.7796637832544475, 0.77737881508079, 0.7728904847396768, 0.7767259670311735, 0.7709319405908275, 0.7559980414558511, 0.7754202709319405, 0.7765627550187694, 0.7646482781132692, 0.7655459441814918, 0.7739513628203035, 0.7754202709319405, 0.7715031826342419, 0.7668516402807246, 0.7749306348947282, 0.774685816876122, 0.758691039660519, 0.7724008487024645, 0.773216908764485, 0.7692182144605842, 0.7681573363799575, 0.767341276317937, 0.7711767586094337, 0.7700342745226049, 0.7621184919210053, 0.7579565856047005, 0.7690550024481801, 0.761628855883793, 0.7607311898155704, 0.7654643381752897, 0.7685653664109678, 0.7604047657907622, 0.7675044883303411, 0.767341276317937, 0.7608127958217725, 0.7553451934062347, 0.7667700342745226, 0.7661171862249061, 0.7548555573690223, 0.7551819813938306, 0.761547249877591, 0.761547249877591, 0.7559164354496491, 0.7648930961318753, 0.7654643381752897, 0.7608127958217725, 0.7623633099396115, 0.7635057940264404, 0.7613024318589848, 0.7634241880202383, 0.7612208258527827, 0.760159947772156, 0.7618736739023992, 0.761628855883793, 0.7562428594744574, 0.7520809531581525, 0.7619552799086012, 0.7610576138403786, 0.7524889831891627, 0.7617920678961971, 0.7609760078341766, 0.7596703117349437, 0.758691039660519, 0.7582830096295088, 0.7590990696915293, 0.7547739513628203, 0.7521625591643545, 0.7547739513628203, 0.7568957075240738, 0.7573037375550841, 0.7574669495674882, 0.7583646156357108, 0.757222131548882, 0.7559980414558511, 0.7590990696915293, 0.7573853435612861, 0.7556716174310429, 0.7553451934062347, 0.75714052554268, 0.7577933735922964, 0.7515097111147381, 0.7492247429410804, 0.7530602252325771, 0.7559164354496491, 0.7526521952015668, 0.7504488330341114, 0.7496327729720907, 0.7476742288232414, 0.7520809531581525, 0.7567324955116697, 0.7573853435612861, 0.7530602252325771, 0.7482454708666558, 0.7542843153256079, 0.7523257711767586, 0.7492247429410804, 0.7521625591643545, 0.7527338012077689, 0.7515913171209401, 0.7534682552635874, 0.751428105108536, 0.7511016810837278, 0.748571894891464, 0.7527338012077689, 0.7512648930961319, 0.7484086828790599, 0.7489799249224743, 0.7529786192263751, 0.7409825363146728, 0.7474294108046352, 0.7503672270279093, 0.7553451934062347, 0.7435939285131385, 0.7533866492573853, 0.7487351069038681, 0.7489799249224743, 0.7468581687612208, 0.7422882324139056, 0.750040803003101, 0.7502040150155052, 0.7506936510527175, 0.750040803003101, 0.7508568630651216, 0.7541211033132038]
# ali_gcn_cpu_pubmed_time=[0.0345214990997314, 0.03452110290527344, 0.033989667892456055, 0.033892154693603516, 0.03524971008300781, 0.03382468223571777, 0.03410792350769043, 0.03497958183288574, 0.034047842025756836, 0.03474116325378418, 0.03518223762512207, 0.036756277084350586, 0.03428387641906738, 0.0342557430267334, 0.03416895866394043, 0.03426408767700195, 0.03436756134033203, 0.03472256660461426, 0.03421425819396973, 0.03425168991088867, 0.034201622009277344, 0.03450727462768555, 0.03397488594055176, 0.03418254852294922, 0.03417205810546875, 0.0342710018157959, 0.033960819244384766, 0.03409242630004883, 0.03415179252624512, 0.03441810607910156, 0.03502964973449707, 0.033995866775512695, 0.034605979919433594, 0.03541970252990723, 0.03421783447265625, 0.03406691551208496, 0.03410053253173828, 0.034035444259643555, 0.03431367874145508, 0.035227060317993164, 0.03444242477416992, 0.03664112091064453, 0.0343623161315918, 0.03450274467468262, 0.033976078033447266, 0.03423500061035156, 0.0343165397644043, 0.03461027145385742, 0.03435182571411133, 0.03434395790100098, 0.03461623191833496, 0.03404998779296875, 0.03420114517211914, 0.03428292274475098, 0.03420710563659668, 0.0340723991394043, 0.03429555892944336, 0.03411579132080078, 0.03450584411621094, 0.03460884094238281, 0.034172773361206055, 0.034079551696777344, 0.034295082092285156, 0.034470558166503906, 0.03424382209777832, 0.03399085998535156, 0.03416728973388672, 0.0344243049621582, 0.03428506851196289, 0.03404593467712402, 0.03384566307067871, 0.03664803504943848, 0.03419303894042969, 0.034940481185913086, 0.03431820869445801, 0.03477287292480469, 0.03402519226074219, 0.03422093391418457, 0.034094810485839844, 0.03438878059387207, 0.0343785285949707, 0.03433108329772949, 0.0344846248626709, 0.03442692756652832, 0.03387141227722168, 0.03420305252075195, 0.0340571403503418, 0.03429603576660156, 0.03416275978088379, 0.034085988998413086, 0.033649444580078125, 0.03391909599304199, 0.034327030181884766, 0.03416728973388672, 0.03410696983337402, 0.03417348861694336, 0.034249305725097656, 0.034012556076049805, 0.03380870819091797, 0.03385138511657715, 0.03430891036987305, 0.0342106819152832, 0.034214019775390625, 0.03414511680603027, 0.034224748611450195, 0.034029483795166016, 0.03432130813598633, 0.03480935096740723, 0.03447675704956055, 0.034136295318603516, 0.03407168388366699, 0.03390669822692871, 0.034433841705322266, 0.03420710563659668, 0.03388667106628418, 0.0339655876159668, 0.034935712814331055, 0.03392839431762695, 0.03493022918701172, 0.034996986389160156, 0.034188270568847656, 0.03428459167480469, 0.03407096862792969, 0.03397202491760254, 0.04296445846557617, 0.03433871269226074, 0.03429365158081055, 0.034043073654174805, 0.03396129608154297, 0.034216880798339844, 0.03468799591064453, 0.03409075736999512, 0.03440427780151367, 0.03501701354980469, 0.03409123420715332, 0.03409528732299805, 0.03444242477416992, 0.03608250617980957, 0.034061431884765625, 0.03442573547363281, 0.03692317008972168, 0.03426980972290039, 0.035288095474243164, 0.03428792953491211, 0.03391838073730469, 0.033727407455444336, 0.03491616249084473, 0.03386712074279785, 0.03541231155395508, 0.03477835655212402, 0.034090280532836914, 0.033951520919799805, 0.03500175476074219, 0.03383064270019531, 0.034216880798339844, 0.035149574279785156, 0.034157752990722656, 0.03519558906555176, 0.034502506256103516, 0.03419756889343262, 0.03642749786376953, 0.03418922424316406, 0.03406643867492676, 0.03481650352478027, 0.03405570983886719, 0.03404355049133301, 0.035004615783691406, 0.03404498100280762, 0.034409284591674805, 0.03495001792907715, 0.03412675857543945, 0.03641963005065918, 0.03465104103088379, 0.035321712493896484, 0.03482341766357422, 0.03420877456665039, 0.03409600257873535, 0.03482985496520996, 0.03679847717285156, 0.034272193908691406, 0.0351099967956543, 0.03382563591003418, 0.0353090763092041, 0.03449273109436035, 0.037456512451171875, 0.03482961654663086, 0.03398299217224121, 0.034150123596191406, 0.03405928611755371, 0.03423428535461426, 0.0339970588684082, 0.03712105751037598, 0.03411984443664551, 0.03407883644104004, 0.034088850021362305, 0.033823251724243164, 0.0339818000793457, 0.03395533561706543, 0.034017324447631836, 0.03362274169921875]
# ali_gcn_cpu_pubmed_test, ali_gcn_cpu_pubmed_time = init_data(ali_gcn_cpu_pubmed_test, ali_gcn_cpu_pubmed_time, ali_gcn_cpu_pubmed_best)


def plot_line(X, Y, labels, savefile=None, color=None):
  assert(len(X) == len(Y) == len(labels))
  # assert(len(X) == len(labels))
  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  plt.figure(figsize=(5, 4))
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

  # plt.plot(ali_gcn_cpu_cora_time, ali_gcn_cpu_cora_test, marker='', markersize=markersize, color="orange", alpha=0.5, label="AliGraph-cora", linewidth=linewidth)
  # plt.plot(ali_gcn_cpu_pubmed_time, ali_gcn_cpu_pubmed_test, marker='', markersize=markersize, color="tomato", label="AliGraph-pubmed", linewidth=linewidth)
  # plt.plot(x, Ours, marker='X', markersize=markersize, color="tomato", label="Ours", linewidth=linewidth)

  
  x_ticks = np.linspace(0, np.max(X), 5).tolist()
  y_labels = [f'{x:.2f}' for x in x_ticks]
  plt.xticks(x_ticks, y_labels, fontsize=15)  # 默认字体大小为10

  y_ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
  y_lables = ['10%', '30%', '50%', '70%', '90%']
  plt.yticks(np.array(y_ticks), y_lables, fontsize=15)
  # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
  # plt.text(1, label_position, dataset,fontsize=25, fontweight='bold')
  # plt.xlabel("Edge Miss Rate", fontsize=15)
  plt.ylabel(f"Test Acc", fontsize=15)
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
  plt.savefig(f'./{savefile}', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
  plt.show()



# plot_line([ali_gcn_cpu_cora_time, ali_gcn_cpu_pubmed_time], [ali_gcn_cpu_cora_test, ali_gcn_cpu_pubmed_test], ['aligraph', 'aaa'])

def parse_log(filename=None):
  assert filename
  if not os.path.exists(filename):
    print(f'{filename} not exist')
  train_acc = []
  val_acc = []
  test_acc = []
  avg_time_list = []
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

  return dataset, [train_acc, val_acc, test_acc], avg_time_list

X, Y = [], []

files = ['seq.log', 'rand.log']
for file in files:
  dataset, acc_list, time_list = parse_log(file)
  ret = get_time_acc(acc_list[1], time_list[1], max(acc_list[1]), False)
  X.append(ret[0])
  Y.append(ret[1])

plot_line(X, Y, ['sequence', 'random'], 'seq-rand.png')