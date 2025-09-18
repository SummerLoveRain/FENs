# pip install SciencePlots
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

# plt.style.use(['science', 'ieee', 'grid'])
# plt.style.use(['science', 'high-vis', 'grid'])
# plt.style.use(['science', 'high-vis'])
plt.style.use(['science', 'ieee', 'no-latex'])


# datas:数据数组[[x, y]]
# data_labes: 每一组数据的标签
# xy_labels: xy轴标签
# title: 图片的标题, 为None则不画
# file_name: 文件保存名字, 默认保存到当前路径下面
# log: 是否以log的方式画图
def plot_line(datas,
              data_labels,
              xy_labels,
              title,
              file_name,
              xlog=False,
              ylog=False,
              marker=False,
              linestyle=False):
        
    linestyles = ['dashdot', 'dotted', 'solid', 'dashed', 'dashdotted', 'dashdotdotted', 'loosely dotted', 'long dash with offset', 'loosely dashed', 'loosely dashdotted', 'loosely dashdotdotted', 'densely dotted', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted']
    linestyle_tuple = {
        'solid': 'solid',
        'dashdot': 'dashdot',
        'loosely dotted': (0, (1, 10)),
        'dotted': (0, (1, 1)),
        'densely dotted': (0, (1, 1)),
        'long dash with offset': (5, (10, 3)),
        'loosely dashed': (0, (5, 10)),
        'dashed': (0, (5, 5)),
        'densely dashed': (0, (5, 1)),
        'loosely dashdotted': (0, (3, 10, 1, 10)),
        'dashdotted': (0, (3, 5, 1, 5)),
        'densely dashdotted': (0, (3, 1, 1, 1)),
        'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
        'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
        'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
    }     

    markers = ['o', 'v', 's', 'p', 'P', '*', 'h', 'X', 'D']    
    markersize=3
    colors = sns.color_palette("colorblind", n_colors=len(data_labels))

    fig, ax = plt.subplots()
    lns = []
    i = 0

    if data_labels is not None:
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            y = data[:, 1]
            # ln = ax.plot(x, y, label=data_label)

            params = {'color': colors[i%(len(colors))]}
            if linestyle:
                params['linestyle'] = linestyle_tuple[linestyles[i%(len(linestyles))]]
            else:
                params['linestyle'] = 'solid'
            if marker:
                params['marker'] = markers[i%(len(markers))]
                params['markersize'] = markersize

            ax.plot(x, y, label=data_label, **params)
            i = i+1
        # ax.legend(fontsize=8)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    else:
        for data in datas:
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y)
    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)


    ax.set_xticks(x)
    ax.set_xticklabels([5, 7, 10, 15])
    
    # xy轴是否以log的方式画图
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    # 设置标题
    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()
    plt.close()


def plot_density(datas,
              data_labels,
              xy_labels,
              title,
              file_name):
    fig, ax = plt.subplots()
    if data_labels is not None:
        colors = sns.color_palette("colorblind", n_colors=len(data_labels))
        i = 0
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            linestyles = ['dashdot', 'dotted', 'solid', 'dashed', 'dashdotted', 'dashdotdotted', 'loosely dotted', 'long dash with offset', 'loosely dashed', 'loosely dashdotted', 'loosely dashdotdotted', 'densely dotted', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted']
            linestyle_tuple = {
                'solid': 'solid',
                'dashdot': 'dashdot',
                'loosely dotted': (0, (1, 10)),
                'dotted': (0, (1, 1)),
                'densely dotted': (0, (1, 1)),
                'long dash with offset': (5, (10, 3)),
                'loosely dashed': (0, (5, 10)),
                'dashed': (0, (5, 5)),
                'densely dashed': (0, (5, 1)),
                'loosely dashdotted': (0, (3, 10, 1, 10)),
                'dashdotted': (0, (3, 5, 1, 5)),
                'densely dashdotted': (0, (3, 1, 1, 1)),
                'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
                'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
                'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
            } 
            # colors = ['steelblue', 'darkred', 'red', 'purple', 'black', 'gray', 'darkblue', 'darkgray']
            sns.kdeplot(x, label=data_label, linestyle=linestyle_tuple[linestyles[i%(len(linestyles))]], color=colors[i%(len(linestyles))])
            i = i+1
        ax.legend()
    else:
        for data in datas:
            x = data[:, 0]
            sns.kdeplot(x)
    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)
    
    # 设置标题
    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)
    # plt.show()
    plt.close()


def plot_line2(datas1,
              data_labels1,
              datas2,
              data_labels2,
              xy_labels,
              title,
              file_name,
              yscale2=None,
              xlog=False,
              ylog1=False,
              ylog2=False,
              marker=False,
              linestyle=False):
    fig, ax = plt.subplots()
    lns = []
    i = 0
    linestyles = ['dashdot', 'dotted', 'solid', 'dashed', 'dashdotted', 'dashdotdotted', 'loosely dotted', 'long dash with offset', 'loosely dashed', 'loosely dashdotted', 'loosely dashdotdotted', 'densely dotted', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted']
    linestyle_tuple = {
        'solid': 'solid',
        'dashdot': 'dashdot',
        'loosely dotted': (0, (1, 10)),
        'dotted': (0, (1, 1)),
        'densely dotted': (0, (1, 1)),
        'long dash with offset': (5, (10, 3)),
        'loosely dashed': (0, (5, 10)),
        'dashed': (0, (5, 5)),
        'densely dashed': (0, (5, 1)),
        'loosely dashdotted': (0, (3, 10, 1, 10)),
        'dashdotted': (0, (3, 5, 1, 5)),
        'densely dashdotted': (0, (3, 1, 1, 1)),
        'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
        'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
        'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
    }     

    markers = ['o', 'v', 's', 'p', 'P', '*', 'h', 'X', 'D']    
    markersize=3
    colors = sns.color_palette("hls", n_colors=len(data_labels1)+len(data_labels2))

    if data_labels1 is not None:
        for data, data_label in zip(datas1, data_labels1):
            x = data[:, 0]
            y = data[:, 1]
            # ln = ax.plot(x, y, label=data_label)

            params = {'color': colors[i%(len(colors))]}
            if linestyle:
                params['linestyle'] = linestyle_tuple[linestyles[i%(len(linestyles))]]
            else:
                params['linestyle'] = 'solid'
            if marker:
                params['marker'] = markers[i%(len(markers))]
                params['markersize'] = markersize

            ln = ax.plot(x, y, label=data_label, **params)
            lns.append(ln)
            i = i+1
        # ax.legend(fontsize=8)
    else:
        for data in datas1:
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y)
    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)

    # xy轴是否以log的方式画图
    if xlog:
        ax.set_xscale('log')
    if ylog1:
        ax.set_yscale('log')

    # 设置标题
    if title is not None:
        ax.set_title(title)
    
    ax2 = ax.twinx()  # this is the important function    if data_labels1 is not None:
    # i = 0
    if data_labels2 is not None:
        for data, data_label in zip(datas2, data_labels2):
            x = data[:, 0]
            y = data[:, 1]

            params = {'color': colors[i%(len(colors))]}
            if linestyle:
                params['linestyle'] = linestyle_tuple[linestyles[i%(len(linestyles))]]
            else:
                params['linestyle'] = 'solid'
            if marker:
                params['marker'] = markers[i%(len(markers))]
                params['markersize'] = markersize


            ln = ax2.plot(x, y, label=data_label, **params)
            lns.append(ln)
            # ax2.plot(x, y, label=data_label, linestyle=linestyles[i])
            i = i+1
        lnss = None
        for l in lns:
            if lnss is None:
                lnss = l
            else:
                lnss = lnss + l
        lbs = [l.get_label() for l in lnss]
        # ax2.legend(lnss, lbs, fontsize=8, loc='center right')
        ax2.legend(lnss, lbs, bbox_to_anchor=(1.20, 1), loc=2, borderaxespad=0)
        # ax2.legend(lnss, lbs, fontsize=8, loc=8)
        # ax2.legend(lnss, lbs, fontsize=8, loc=8, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    else:
        for data in datas2:
            x = data[:, 0]
            y = data[:, 1]
            ax2.plot(x, y, linestyle=linestyle_tuple[linestyles[i%(len(linestyles))]], color=colors[i%(len(linestyles))])
            # ax2.plot(x, y, linestyle=linestyles[i])
            i = i+1
    ax2.set(ylabel=xy_labels[2])
    ax2.autoscale(tight=True)
    if ylog2:
        ax2.set_yscale('log')
    if yscale2 is not None:
        ax2.set_ylim(yscale2[0], yscale2[1])
    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()


def plot_band(datas,
            data_labels,
            xy_labels,
            u_inftys_upper,
            u_inftys_lower,
            band_label,
            title,
            file_name,
            xlog=False,
            ylog=True):
    
    fig, ax = plt.subplots()
    if data_labels is not None:
        colors = sns.color_palette("colorblind", n_colors=len(data_labels))
        i = 0
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            y = data[:, 1]
            linestyles = ['dashdot', 'dotted', 'solid', 'dashed', 'dashdotted', 'dashdotdotted', 'loosely dotted', 'long dash with offset', 'loosely dashed', 'loosely dashdotted', 'loosely dashdotdotted', 'densely dotted', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted']
            linestyle_tuple = {
                'solid': 'solid',
                'dashdot': 'dashdot',
                'loosely dotted': (0, (1, 10)),
                'dotted': (0, (1, 1)),
                'densely dotted': (0, (1, 1)),
                'long dash with offset': (5, (10, 3)),
                'loosely dashed': (0, (5, 10)),
                'dashed': (0, (5, 5)),
                'densely dashed': (0, (5, 1)),
                'loosely dashdotted': (0, (3, 10, 1, 10)),
                'dashdotted': (0, (3, 5, 1, 5)),
                'densely dashdotted': (0, (3, 1, 1, 1)),
                'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
                'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
                'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
            }            
            # colors = ['steelblue', 'darkred', 'red', 'purple', 'black', 'gray', 'darkblue', 'darkgray']
            ax.plot(x, y, label=data_label, linestyle=linestyle_tuple[linestyles[i%(len(linestyles)+1)]], color=colors[i%(len(linestyles)+1)])
            # ax.plot(x, y, label=data_label, linestyle=linestyles[i-7], color=colors[i-7], marker = markers[i-7])
            i = i + 1 
                
        plt.fill_between(x, u_inftys_upper, u_inftys_lower, alpha=0.5, label=band_label)
        # ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    else:
        for data in datas:
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y)
        plt.fill_between(x, u_inftys_upper, u_inftys_lower, alpha=0.5, label=band_label)

    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)

    # xy轴是否以log的方式画图
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
            
    
    # 设置标题
    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()
    plt.close()


def plot_bands(datas,
            data_labels,
            xy_labels,
            u_inftys_uppers,
            u_inftys_lowers,
            band_labels,
            title,
            file_name,
            xlog=False,
            ylog=True):
    
    fig, ax = plt.subplots()
    if data_labels is not None:
        colors = sns.color_palette("colorblind", n_colors=len(data_labels))
        i = 0
        for data, data_label in zip(datas, data_labels):
            x = data[:, 0]
            y = data[:, 1]
            linestyles = ['dashdot', 'dotted', 'solid', 'dashed', 'dashdotted', 'dashdotdotted', 'loosely dotted', 'long dash with offset', 'loosely dashed', 'loosely dashdotted', 'loosely dashdotdotted', 'densely dotted', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted']
            linestyle_tuple = {
                'solid': 'solid',
                'dashdot': 'dashdot',
                'loosely dotted': (0, (1, 10)),
                'dotted': (0, (1, 1)),
                'densely dotted': (0, (1, 1)),
                'long dash with offset': (5, (10, 3)),
                'loosely dashed': (0, (5, 10)),
                'dashed': (0, (5, 5)),
                'densely dashed': (0, (5, 1)),
                'loosely dashdotted': (0, (3, 10, 1, 10)),
                'dashdotted': (0, (3, 5, 1, 5)),
                'densely dashdotted': (0, (3, 1, 1, 1)),
                'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
                'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
                'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
            }            
            # colors = ['steelblue', 'red', 'purple', 'goldenrod', 'gray', 'darkseagreen', 'darkblue', 'darkgray', 'darkred', 'black']
            ax.plot(x, y, label=data_label, linestyle=linestyle_tuple[linestyles[i%(len(linestyles))]], color=colors[i%(len(colors))])
            # ax.plot(x, y, label=data_label, linestyle=linestyles[i-7], color=colors[i-7], marker = markers[i-7])
            if band_labels is not None:    
                plt.fill_between(x, u_inftys_uppers[i], u_inftys_lowers[i], alpha=0.5, label=band_labels[i], color=colors[i%(len(colors))])
            else:
                plt.fill_between(x, u_inftys_uppers[i], u_inftys_lowers[i], alpha=0.5, color=colors[i%(len(colors))])
            i = i + 1 
        # ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    else:
        i = 0
        for data in datas:
            x = data[:, 0]
            y = data[:, 1]
            ax.plot(x, y)
            if band_labels is not None:    
                plt.fill_between(x, u_inftys_uppers[i], u_inftys_lowers[i], alpha=0.5, label=band_labels[i], color=colors[i%(len(colors))])
            else:
                plt.fill_between(x, u_inftys_uppers[i], u_inftys_lowers[i], alpha=0.5, color=colors[i%(len(colors))])
            i = i + 1

    ax.set(xlabel=xy_labels[0])
    ax.set(ylabel=xy_labels[1])
    ax.autoscale(tight=True)

    # xy轴是否以log的方式画图
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
            
    
    # 设置标题
    if title is not None:
        ax.set_title(title)

    fig.savefig(file_name + '.png', dpi=300)

    # plt.show()
    plt.close()