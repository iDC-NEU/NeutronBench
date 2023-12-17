import sys
import os
import time
import utils
import numpy as np
import matplotlib.pyplot as plt

init_command = [
    "WEIGHT_DECAY:0.0001",
    "DROP_RATE:0.5",
    "DECAY_RATE:0.97",
    "DECAY_EPOCH:100",
    "PROC_OVERLAP:0",
    "PROC_LOCAL:0",
    "PROC_CUDA:0",
    "PROC_REP:0",
    "LOCK_FREE:1",
    "TIME_SKIP:3",
    "MINI_PULL:1",
    "BATCH_NORM:0",
    "PROC_REP:0",
    "LOCK_FREE:1",
    "CACHE_TYPE:none",
    "CACHE_POLICY:none",
    "CACHE_RATE:0",
]

graph_config = {
    'reddit': "VERTICES:232965\nEDGE_FILE:../data/reddit/reddit.edge\nFEATURE_FILE:../data/reddit/reddit.feat\nLABEL_FILE:../data/reddit/reddit.label\nMASK_FILE:../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
    'ogbn-arxiv': "VERTICES:169343\nEDGE_FILE:../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
    'ogbn-products': "VERTICES:2449029\nEDGE_FILE:../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../data/ogbn-products/ogbn-products.label\nMASK_FILE:../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
    'AmazonCoBuy_computers': "VERTICES:13752\nEDGE_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.edge\nFEATURE_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.feat\nLABEL_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.label\nMASK_FILE:../data/AmazonCoBuy_computers/AmazonCoBuy_computers.mask\nLAYERS:767-128-10\n",
    'AmazonCoBuy_photo': "VERTICES:7650\nEDGE_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.edge\nFEATURE_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.feat\nLABEL_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.label\nMASK_FILE:../data/AmazonCoBuy_photo/AmazonCoBuy_photo.mask\nLAYERS:745-128-8\n",
    'enwiki-links': "VERTICES:13593032\nEDGE_FILE:../data/enwiki-links/enwiki-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'livejournal': "VERTICES:4846609\nEDGE_FILE:../data/livejournal/livejournal.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-large': "VERTICES:7489073\nEDGE_FILE:../data/lj-large/lj-large.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'lj-links': "VERTICES:5204175\nEDGE_FILE:../data/lj-links/lj-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'europe_osm': "VERTICES:50912018\nEDGE_FILE:../data/europe_osm/europe_osm.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dblp-2011': "VERTICES:933258\nEDGE_FILE:../data/dblp-2011/dblp-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'frwiki-2013': "VERTICES:1350986\nEDGE_FILE:../data/frwiki-2013/frwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'dewiki-2013': "VERTICES:1510148\nEDGE_FILE:../data/dewiki-2013/dewiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'itwiki-2013': "VERTICES:1016179\nEDGE_FILE:../data/itwiki-2013/itwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'hollywood-2011': "VERTICES:1985306\nEDGE_FILE:../data/hollywood-2011/hollywood-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    'enwiki-2016': "VERTICES:5088560\nEDGE_FILE:../data/enwiki-2016/enwiki-2016.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
}


def new_command(
    dataset,
    fanout='2,2',
    valfanout='-1,-1',
    batch_size='6000',
    algo='GCNNEIGHBORGPU',
    epochs='10',
    batch_type='random',
    lr='0.01',
    run='1',
    classes='1',
    **kw,
):

    other_config = init_command
    other_config.append(f'ALGORITHM:{algo}')
    other_config.append(f'FANOUT:{fanout}')
    other_config.append(f'VALFANOUT:{valfanout}')
    other_config.append(f'BATCH_SIZE:{batch_size}')
    other_config.append(f'EPOCHS:{epochs}')
    other_config.append(f'BATCH_TYPE:{batch_type}')
    other_config.append(f'LEARN_RATE:{lr}')
    other_config.append(f'RUNS:{lr}')
    other_config.append(f'CLASSES:{classes}')
    other_config.append(f'RUNS:{run}')
    for k, v in kw.items():
        other_config.append(f'{k}:{v}')
        print(k, v)
    # assert False
    ret = graph_config[dataset] + '\n'.join(init_command)
    return ret


def run(dataset, cmd, log_path, suffix=''):
    if not os.path.exists(log_path):
        utils.create_dir(log_path)

    run_time = time.time()
    with open('tmp.cfg', 'w') as f:
        f.write(cmd)

    run_command = f'mpiexec -hostfile hostfile -np 1 ./build/nts tmp.cfg > {log_path}/{dataset}{suffix}.log'
    print('running: ', run_command)
    os.system(run_command)

    run_time = time.time() - run_time
    print(f'done! cost {run_time:.2f}s')


def exp2(datasets, batch_sizes, run_times, lr_=0.01):
    for ds in datasets:
        file_path = f'./log/batch-size-nts-dgl-sdl/{ds}-{lr_}'
        utils.create_dir(file_path)
        for bs in batch_sizes[ds]:
            # cmd = new_command(ds, batch_type='shuffle', fanout='10,25', valfanout='10,25', epochs=3, batch_size=bs, RUN_TIME=run_times[ds])
            # cmd = new_command(ds, batch_type='shuffle', fanout='10,25', lr=0.01, epochs=3, batch_size=bs, RUN_TIME=run_times[ds], valfanout='10,25')
            cmd = new_command(
                ds,
                batch_type='shuffle',
                fanout='10,25',
                lr=lr_,
                epochs=3,
                batch_size=bs,
                RUN_TIME=run_times[ds],
                valfanout='10,25',
                MODE='zerocopy',
                CACHE_TYPE='gpu_memory',
                CACHE_POLICY='degree',
            )
            # if ds == 'reddit':
            #   cmd = new_command(ds, batch_type='shuffle', fanout='10,25', lr=0.001, epochs=3, batch_size=bs, RUN_TIME=run_times[ds], valfanout='10,25')
            run(ds, cmd, file_path, suffix=f'-{bs}')


def print_val_acc(mode, datasets, batch_sizes):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for ds in datasets:
        for bs in batch_sizes[ds]:
            val_acc_list = []
            log_file = f'./log/batch-size/{ds}/{ds}-{bs}.log'
            val_acc = utils.parse_num(log_file, mode)
            val_acc_list += val_acc

            ret[ds + str(bs)] = val_acc_list
    # print(ret)
    return ret


def print_train_time(mode, datasets, batch_sizes):
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products', 'enwiki-links', 'livejournal', 'lj-large', 'lj-links']
    ret = {}
    for ds in datasets:
        for bs in batch_sizes[ds]:
            val_acc_list = []
            log_file = f'./log/batch-size/{ds}/{ds}-{bs}.log'
            val_acc = utils.parse_num(log_file, mode)
            val_acc_list += val_acc

            ret[ds + str(bs)] = val_acc_list
    # print(ret)
    return ret


def print_best_val_acc(datasets, batch_sizes, run_times):
    val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
    for k, v in val_acc_dict.items():
        val_acc_dict[k] = np.max(v)
        print(k, '\t', np.max(v))
    # print(val_acc_dict)


def draw_converge_speed(datasets, batch_sizes, run_times, y_lim=None):
    val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
    train_time_dict = print_val_acc('train_time', datasets, batch_sizes)
    # val_accs = {  'ogbn-arxiv': 0.75,
    #               'AmazonCoBuy_computers': 1,
    #               'AmazonCoBuy_photo': 1,
    #               'reddit': 1,
    #             }

    for ds in datasets:
        X, Y = [], []
        for i, bs in enumerate(batch_sizes[ds]):
            Y.append(val_acc_dict[ds + str(bs)])
            # print(ds+str(bs), 'len', len(train_time_dict[ds+str(bs)][3:]))
            # print(train_time_dict[ds+str(bs)])
            avg_train_time = np.average(train_time_dict[ds + str(bs)][3:])
            # print(avg_train_time, avg_train_time * len(Y[-1]))
            X.append(np.cumsum([avg_train_time] * len(Y[-1])))
        rename_dict = {
            'reddit': 'reddit',
            'ogbn-arxiv': 'arxiv',
            'ogbn-products': 'products',
            'computer': 'computer',
            'photo': 'photo',
        }
        # labels = [f'{ds}-{bs}' for bs in batch_sizes[ds]]
        # labels = [f'{rename_dict[ds]}-{bs}' for bs in batch_sizes[ds]]
        labels = [f'{bs}' for bs in batch_sizes[ds]]

        x_ticks = np.linspace(0, run_times[ds], 6)
        print(x_ticks)
        if run_times[ds] >= 6:
            x_name = [f'{x:.0f}' for x in x_ticks]
        else:
            x_name = [f'{x:.1f}' for x in x_ticks]

        Y = [np.array(y) / np.array(x) for x, y in zip(X, Y)]
        if y_lim is not None:
            y_ticks = np.linspace(y_lim[ds][0], y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        else:

            max_y = max([max(y) for y in Y])
            y_ticks = np.linspace(0, max_y, 6)
            # print(y_ticks)
            # print('max', max(max(Y)))
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        # print(len(X), len(Y))
        # utils.plot_line(X, Y, labels, savefile=f'{os.getcwd()}/overleaf-gnn-eval/exp3-gpu-cache/vary_cache_ratio_{ds}.pdf', x_ticks=x_ticks_dict[ds], x_label='Cache Ratio (%)', y_label='Cache Hit Ratio (%)')
        # utils.plot_line(X, Y, labels, savefile=f'{os.getcwd()}/log/batch-size/pdf/{ds}.pdf', x_ticks=x_ticks, x_name=x_name, y_ticks=y_ticks, y_name=y_name, x_label='Time (s)', y_label='Val ACC (%)')
        pdf_dir = f'{os.getcwd()}/log/batch-size/pdf'
        utils.create_dir(pdf_dir)

        plot_line(
            X,
            Y,
            labels,
            savefile=f'{pdf_dir}/{ds}-converge-speed.pdf',
            x_ticks=x_ticks,
            x_name=x_name,
            y_ticks=y_ticks,
            y_name=y_name,
            x_label='Time (s)',
            y_label='Converge Speed (val-acc/s)',
            label_col=2,
        )


def draw_val_acc(datasets, batch_sizes, run_times, y_lim=None):
    val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
    train_time_dict = print_val_acc('train_time', datasets, batch_sizes)
    # for k, v in val_acc_dict.items():
    #   print(k, '\t', np.max(v))
    # val_accs = {  'ogbn-arxiv': 0.75,
    #               'AmazonCoBuy_computers': 1,
    #               'AmazonCoBuy_photo': 1,
    #               'reddit': 1,
    #             }

    for ds in datasets:
        X, Y = [], []
        for i, bs in enumerate(batch_sizes[ds]):
            Y.append(val_acc_dict[ds + str(bs)])
            # print(ds+str(bs), 'len', len(train_time_dict[ds+str(bs)][3:]))
            # print(train_time_dict[ds+str(bs)])
            # assert(False)
            avg_train_time = np.average(train_time_dict[ds + str(bs)][3:])
            # print(avg_train_time, avg_train_time * len(Y[-1]))
            X.append(np.cumsum([avg_train_time] * len(Y[-1])))
        rename_dict = {
            'reddit': 'reddit',
            'ogbn-arxiv': 'arxiv',
            'AmazonCoBuy_computers': 'computer',
            'AmazonCoBuy_photo': 'photo',
        }
        # labels = [f'{ds}-{bs}' for bs in batch_sizes[ds]]
        # labels = [f'{rename_dict[ds]}-{bs}' for bs in batch_sizes[ds]]
        labels = [f'{bs}' for bs in batch_sizes[ds]]

        x_ticks = np.linspace(0, run_times[ds], 6)
        x_name = [f'{x:.0f}' for x in x_ticks]

        if y_lim is not None:
            y_ticks = np.linspace(y_lim[ds][0], y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        else:
            y_ticks = np.linspace(0, y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        # print(len(X), len(Y))
        # utils.plot_line(X, Y, labels, savefile=f'{os.getcwd()}/overleaf-gnn-eval/exp3-gpu-cache/vary_cache_ratio_{ds}.pdf', x_ticks=x_ticks_dict[ds], x_label='Cache Ratio (%)', y_label='Cache Hit Ratio (%)')
        # utils.plot_line(X, Y, labels, savefile=f'{os.getcwd()}/log/batch-size/pdf/{ds}.pdf', x_ticks=x_ticks, x_name=x_name, y_ticks=y_ticks, y_name=y_name, x_label='Time (s)', y_label='Val ACC (%)')
        plot_line(
            X,
            Y,
            labels,
            savefile=f'{os.getcwd()}/log/batch-size/pdf/{ds}.pdf',
            x_ticks=x_ticks,
            x_name=x_name,
            y_ticks=y_ticks,
            y_name=y_name,
            x_label='Time (s)',
            y_label='Val ACC (%)',
        )


def draw_time_to_accuracy(datasets, batch_sizes, val_acc, y_lim=None):
    val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
    train_time_dict = print_val_acc('train_time', datasets, batch_sizes)

    for ds in datasets:
        X, Y = [], []
        for i, bs in enumerate(batch_sizes[ds]):
            Y.append(val_acc_dict[ds + str(bs)])
            # print(ds+str(bs), 'len', len(train_time_dict[ds+str(bs)][3:]))
            # print(train_time_dict[ds+str(bs)])
            avg_train_time = np.average(train_time_dict[ds + str(bs)][3:])
            # print(avg_train_time, avg_train_time * len(Y[-1]))
            X.append(np.cumsum([avg_train_time] * len(Y[-1])))
        rename_dict = {
            'reddit': 'reddit',
            'ogbn-arxiv': 'arxiv',
            'ogbn-products': 'products',
            'computer': 'computer',
            'photo': 'photo',
        }
        # labels = [f'{ds}-{bs}' for bs in batch_sizes[ds]]
        # labels = [f'{rename_dict[ds]}-{bs}' for bs in batch_sizes[ds]]
        labels = [f'{bs}' for bs in batch_sizes[ds]]

        # print(len(X), len(Y))
        # utils.plot_line(X, Y, labels, savefile=f'{os.getcwd()}/overleaf-gnn-eval/exp3-gpu-cache/vary_cache_ratio_{ds}.pdf', x_ticks=x_ticks_dict[ds], x_label='Cache Ratio (%)', y_label='Cache Hit Ratio (%)')
        # utils.plot_line(X, Y, labels, savefile=f'{os.getcwd()}/log/batch-size/pdf/{ds}.pdf', x_ticks=x_ticks, x_name=x_name, y_ticks=y_ticks, y_name=y_name, x_label='Time (s)', y_label='Val ACC (%)')
        pdf_dir = f'{os.getcwd()}/log/batch-size/pdf'
        utils.create_dir(pdf_dir)

        Y_t = []
        X_t = []
        for x, y in zip(X, Y):
            if np.max(y) < val_acc[ds]:
                Y_t.append(y)
                X_t.append(x)
            else:
                y_t = np.array(y)
                idx = np.where(y_t >= val_acc[ds])[0][0] + 1
                Y_t.append(y[:idx])
                X_t.append(x[:idx])
        assert len(Y) == len(Y_t)
        Y = Y_t
        X = X_t

        max_X = max([max(x) for x in X])
        x_ticks = np.linspace(0, max_X, 6)
        x_name = [f'{x:.0f}' for x in x_ticks]
        print(max_X, x_ticks, x_name)

        if y_lim is not None:
            y_ticks = np.linspace(y_lim[ds][0], y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        else:
            y_ticks = np.linspace(0, y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        print(y_ticks)
        plot_line(
            X,
            Y,
            labels,
            savefile=f'{pdf_dir}/{ds}-time-to-acc.pdf',
            x_ticks=x_ticks,
            x_name=x_name,
            y_ticks=y_ticks,
            y_name=y_name,
            x_label='Time (s)',
            y_label='Val ACC (%)',
        )


def plot_line(
    X,
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
    xscale=None,
    label_col=None,
):
    assert len(X) == len(Y) == len(labels)
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
            'blue',
            'green',
            'orange',
            'purple',
            'red',
            'black',
            'yellow',
            'cyan',
            'pink',
            'magenta',
            'deepskyblue',
            'tomato',
        ]
        # color = ['orange', 'blue', 'green', 'tomato', 'purple', 'deepskyblue', 'red']

    for i in range(len(X)):
        if len(X[i]) == 0:
            continue
        ax.plot(
            X[i], Y[i], marker='', markersize=markersize, color=color[i], alpha=1, label=labels[i], linewidth=linewidth
        )
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
    ax.legend(loc=loc, numpoints=1, ncol=3 if label_col is None else label_col, prop={'size': fontsize - 2})
    # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))
    # leg = ax.gca().get_legend()
    # ltext = leg.get_texts()
    # ax.setp(ltext, fontsize=15)
    # ax.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
    # ax.tight_layout()

    if not savefile:
        savefile = 'plot_line.pdf'
    print(f'save to {savefile}')
    fig.savefig(f'{savefile}', format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    if show:
        plt.show()
    plt.close()


def draw_best_val_acc(datasets, batch_sizes, run_times, y_lim=None):
    val_acc_dict = print_val_acc('val_acc', datasets, batch_sizes)
    for k, v in val_acc_dict.items():
        val_acc_dict[k] = np.max(v)
        print(k, '\t', np.max(v))

    for ds in datasets:
        #    X.append([])
        #    Y.append([])
        X, Y = [], []
        for bs in batch_sizes[ds]:
            X.append(bs)
            Y.append(val_acc_dict[str(ds) + str(bs)])
        X = [np.log2(X)]
        Y = [Y]
        labels = [ds]

        x_ticks = X[0]
        x_name = [f'{x:.0f}' for x in x_ticks]
        x_name[-1] = ''

        if y_lim is not None:
            y_ticks = np.linspace(y_lim[ds][0], y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]
        else:
            y_ticks = np.linspace(0, y_lim[ds][1], 6)
            y_name = [f'{x*100:.0f}' for x in y_ticks]

            # y_ticks = np.linspace(0, 1, 6)
            # y_name = [f'{x*100:.0f}' for x in y_ticks]

        pdf_dir = f'{os.getcwd()}/log/batch-size/pdf'
        utils.create_dir(pdf_dir)
        plot_line(
            X,
            Y,
            labels,
            savefile=f'{pdf_dir}/{ds}-bestacc.pdf',
            x_ticks=x_ticks,
            x_name=x_name,
            y_ticks=y_ticks,
            y_name=y_name,
            x_label='Batch Size (log scale)',
            y_label='Val ACC (%)',
        )


if __name__ == '__main__':
    utils.create_dir('./build')
    os.system('cd build && make -j $(nproc) && cd ..')

    datasets = ['itwiki-2013', 'enwiki-2016', 'hollywood-2011', 'reddit', 'lj-links', 'enwiki-links']
    datasets = ['hollywood-2011', 'reddit']
    datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products']
    datasets = ['reddit', 'ogbn-arxiv', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo']
    datasets = ['reddit', 'ogbn-arxiv']
    datasets = ['AmazonCoBuy_computers', 'AmazonCoBuy_photo']
    datasets = ['ogbn-arxiv']
    batch_sizes = {
        'AmazonCoBuy_computers': (512, 1024, 2048, 4096, 8250),
        'AmazonCoBuy_photo': (512, 1024, 2048, 4590),
        'ogbn-arxiv': (128, 512, 3072, 6144, 12288, 24576, 49152, 90941),
        'ogbn-arxiv': (128, 512),
        'ogbn-arxiv': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 90941),
        'reddit': (512, 2048, 8192, 32768, 131072, 153431),
        'ogbn-products': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196615),
        # 'reddit': (512, 2048, 8192, 32768, 131072),
        'reddit': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
    }

    run_times = {
        'reddit': 300,
        'ogbn-arxiv': 300,
        'ogbn-products': 300,
        'AmazonCoBuy_computers': 50,
        'AmazonCoBuy_photo': 25,
    }

    datasets = ['ogbn-arxiv', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo']
    datasets = ['reddit', 'ogbn-arxiv', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo']
    datasets = ['reddit']
    datasets = [ 'reddit', 'ogbn-products']
    datasets = [ 'reddit', 'ogbn-products']
    datasets = ['ogbn-arxiv', 'ogbn-products', 'reddit']
    datasets = ['ogbn-arxiv']

    # datasets = ['ogbn-arxiv']
    exp2(datasets, batch_sizes, run_times, 0.001)
    # exp2(datasets, batch_sizes, run_times, 0.01)
    # datasets = ['ogbn-products']
    # exp2(datasets, batch_sizes, run_times, 0.001)
    # exp2(datasets, batch_sizes, run_times, 0.01)


    # datasets = ['reddit']
    # exp2(datasets, batch_sizes, run_times, 0.001)
    # exp2(datasets, batch_sizes, run_times, 0.01)
    exit(1)

    y_lims = {
        'reddit': [0.91, 0.96],
        'ogbn-arxiv': [0.67, 0.712],
        'computers': [0.64, 0.9],
        'ogbn-products': [0.64, 0.9],
        'ogbn-products': [0.80, 0.91],
        'photo': [0.85, 0.95],
    }
    draw_best_val_acc(datasets, batch_sizes, run_times, y_lims)
    # datasets = ['reddit', 'ogbn-arxiv']
    # datasets = ['AmazonCoBuy_computers', 'AmazonCoBuy_photo']
    # print_best_val_acc(datasets, batch_sizes, run_times)
    y_lims = {
        'reddit': [0.86, 0.94],
        'ogbn-arxiv': [0.66, 0.715],
        'AmazonCoBuy_computers': [0.64, 0.9],
        'AmazonCoBuy_photo': [0.85, 0.95],
        'ogbn-products': [0.61, 0.92],
    }
    draw_val_acc(datasets, batch_sizes, run_times, y_lims)

    val_acc = {
        'reddit': 0.93,
        'ogbn-arxiv': 0.70,
        'ogbn-products': 0.89,
    }
    y_lims = {
        # 'reddit': [0.86, 0.942],
        'ogbn-arxiv': [0.30, 0.715],
        'reddit': [0.92, 0.94],
        'ogbn-products': [0.61, 0.92],
    }
    draw_time_to_accuracy(datasets, batch_sizes, val_acc, y_lims)

    run_times = {
        'reddit': 5,
        'ogbn-arxiv': 3,
        'ogbn-products': 8,
        'computer': 50,
        'photo': 25,
    }
    #   batch_sizes = {
    #       # 'reddit': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
    #         'reddit': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 153431),
    #       'ogbn-arxiv': (128, 512, 3072, 6144, 12288, 24576, 49152, 90941),
    #       'ogbn-products': (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196615)
    #   }
    y_lims = {
        'reddit': [0, 0.80],
        'ogbn-arxiv': [0, 10.0],
        'ogbn-products': [0, 0.72],
        'AmazonCoBuy_computers': [0.64, 0.9],
        'AmazonCoBuy_photo': [0.85, 0.95],
    }
    draw_converge_speed(datasets, batch_sizes, run_times)
    # draw_converge_speed(datasets, batch_sizes, run_times, y_lims)
