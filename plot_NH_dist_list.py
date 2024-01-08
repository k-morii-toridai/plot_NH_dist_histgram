import os
import sys

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def mk_fig_histgram(array, figsize=6):
    """
    This func() make a histgram under 5Å.

    Parameters
    -----------
    array: CO_bond_dist_1d_list_loaded
    figsize: inch

    Usage
    ------
    mk_fig_histgram(array=CO_bond_dist_1d_list_loaded, figsize=5)

    Returns
    --------
    Figure, .eps, .pdf, .png(:dpi=100, 200, 300)
    """

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(figsize, figsize))
    sns.histplot(array,
                 ax=axes,
                 label='nitrogen-hydrogen distance',
                 color='lightgray',
                 edgecolor=None,)
    # タイトルの設定
    if figsize == 5:
        fontsize = 10
    if figsize == 6:
        fontsize = 13
    axes.set_title('Histgram of distances between nitrogen and hydrogen under 5Å',
                   y=1.05,
                   fontsize=fontsize,)
    # 軸ラベルの設定
    axes.set_xlabel('Distance(Å)', fontsize=15)
    axes.set_ylabel('Count', fontsize=15)
    axes.xaxis.set_label_coords(0.5, -0.08)  # x軸ラベルの位置を調整
    axes.yaxis.set_label_coords(-0.11, 0.5)  # y軸ラベルの位置を調整
    # x軸の範囲設定
    axes.set_xlim(right=5)
    # y軸を指数表示
    axes.ticklabel_format(style="sci",  axis="y", scilimits=(0, 0))
    # 凡例の設定
    axes.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=fontsize-3)  # 凡例を表示
    # save
    f_name = f'figure/xlim_under500pm/size_{figsize}_{figsize}/'
    None if os.path.exists(f_name) else os.makedirs(f_name)
    # .png形式で保存
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_300dpi_s{figsize}.png', dpi=300)
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_200dpi_s{figsize}.png', dpi=200)
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_100dpi_s{figsize}.png', dpi=100)
    # .pdf形式で保存
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_s{figsize}.pdf')
    # .eps形式で保存
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_s{figsize}.eps')
    # show
    # plt.show()


def mk_fig_histgram_under200pm(array, figsize=6):
    """
    This func() make a histgram under 1.1Å.

    Parameters
    -----------
    array: CO_bond_dist_1d_list_under2
    figsize: inch

    Usage
    ------
    mk_fig_histgram_under200pm(array=CO_bond_dist_1d_list_under2, figsize=5)

    Returns
    --------
    Figure, .eps, .pdf, .png(:dpi=100, 200, 300)
    """

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(figsize, figsize))
    sns.histplot(array,
                 ax=axes,
                 label='nitrogen-hydrogen distance',
                 color='lightgray',
                 edgecolor=None,)
    # タイトルの設定
    if figsize == 5:
        fontsize = 10
    if figsize == 6:
        fontsize = 13
    axes.set_title('Histgram of distances between nitrogen and hydrogen under 1.1Å',
                   y=1.05,
                   fontsize=fontsize,)
    # 軸ラベルの設定
    axes.set_xlabel('Distance(Å)', fontsize=15)
    axes.set_ylabel('Count', fontsize=15)
    axes.xaxis.set_label_coords(0.5, -0.08)  # x軸ラベルの位置を調整
    axes.yaxis.set_label_coords(-0.11, 0.5)  # y軸ラベルの位置を調整
    # x軸の範囲設定
    axes.set_xlim(left=0.7, right=1.1)
    # y軸を指数表示
    axes.ticklabel_format(style="sci",  axis="y", scilimits=(0, 0))
    # 凡例の設定
    axes.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=fontsize-3)  # 凡例を表示
    # save
    f_name = f'figure/xlim_under200pm/size_{figsize}_{figsize}/'
    None if os.path.exists(f_name) else os.makedirs(f_name)
    # .png形式で保存（dpi3パターン）
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_under200pm_300dpi_s{figsize}.png', dpi=300)
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_under200pm_200dpi_s{figsize}.png', dpi=200)
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_under200pm_100dpi_s{figsize}.png', dpi=100)
    # .pdf形式で保存
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_under200pm_s{figsize}.pdf')
    # .eps形式で保存
    fig.savefig(f'{f_name}/Histgram_of_distances_between_carbon_and_oxygen_under200pm_s{figsize}.eps')
    # show
    # plt.show()


# input number which size of figure you want to make
args = sys.argv
args[1] = int(args[1])
# make a histgram of CO-bond distances under 5Å
CO_bond_dist_1d_list_loaded = np.load('NH_dist_1d_list.npy', allow_pickle=True)
mk_fig_histgram(array=CO_bond_dist_1d_list_loaded, figsize=args[1])
print(f"A histgram of CO-bond distances under 5Å was made \n\
and saved at figure/under500pm/size_{args[1]}_{args[1]}/")
# make a histgram of CO-bond distances under 2Å
CO_bond_dist_1d_list_under2 = CO_bond_dist_1d_list_loaded[CO_bond_dist_1d_list_loaded <= 1.1]
mk_fig_histgram_under200pm(array=CO_bond_dist_1d_list_under2, figsize=args[1])
print(f"A histgram of CO-bond distances under 3Å was made \n\
and saved at figure/under200pm/size_{args[1]}_{args[1]}/")
