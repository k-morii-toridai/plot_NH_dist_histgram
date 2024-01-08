import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from package_file_conversion.nnlist2df import nnlist2df


def flatten_func(list_2dim):
    return list(itertools.chain.from_iterable(list_2dim))


def get_CO_bond_dist_list(nnlist_path='sample_test_files/POSCAR.nnlist'):
    df_nnlist = nnlist2df(nnlist_path)
    df_nnlist_C_O = df_nnlist[df_nnlist.apply(lambda row: (row['central_atom_symbol'] == 'N') and (row['neighboring_atom_symbol'] == 'H'), axis=1)]
    CO_bond_dist_list = df_nnlist_C_O['rel_distance'].tolist()
    return CO_bond_dist_list


# 元素種C, Oを含むPOSCARファイルパスリストをload
C_O_existed_poscar_folder_path_list = np.load('../get_some_speceis_existed_poscar_path_list/N_H_existed_poscar_folder_path_list.npy', allow_pickle=True)
# CO結合間距離を抽出したいnnlist_5/POSCAR.nnlistのパスリストを作成
C_O_existed_nnlist_5_path_list = [str(folder_p) + '/nnlist_5/POSCAR.nnlist' for folder_p in C_O_existed_poscar_folder_path_list]

# 並列化して処理
try:
    p = Pool(cpu_count() - 1)
    CO_bond_dist_2d_list = list(tqdm(p.imap(get_CO_bond_dist_list, C_O_existed_nnlist_5_path_list), total=len(C_O_existed_nnlist_5_path_list)))
    # flatten
    CO_bond_dist_1d_list = flatten_func(CO_bond_dist_2d_list)
finally:
    p.close()
    p.join()

# 抽出したCO結合間距離の1Dリストをを.npy形式で保存
np.save('NH_dist_1d_list.npy', np.array(CO_bond_dist_1d_list))
