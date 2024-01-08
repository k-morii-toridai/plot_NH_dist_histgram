""" This module is written for convert POSCAR.nnlist to pd.DataFrame. """

import pandas as pd
# 表示オプションを設定
pd.set_option('display.float_format', '{:.6f}'.format)


def nnlist2df(nnlist_path='POSCAR.nnlist'):
    """
    This func converts POSCAR.nnlist to pd.DataFrame.

    Usage:
    -------
    df_nnlist = nnlist2df(nnlist_path='POSCAR.nnlist')

    Parameter:
    ------------
    nnlist_path: str or pathlib.Path

    Return:
    -------
    pd.DataFrame
    """

    # データを読み込みます。列名を指定し、区切り文字として空白を指定します。
    df = pd.read_csv(nnlist_path, sep='\\s+', header=None)
    # 列名を設定します。
    df.columns = ["central_atom_id", "neighboring_atom_id", "rel_distance", "rel_x", "rel_y", "rel_z",
                  "unitcell_x", "unitcell_y", "unitcell_z", "central_atom_symbol", "neighboring_atom_symbol"]
    # cast int64 to str for central_atom_id column
    df['central_atom_id'] = df['central_atom_id'].astype(str)
    # cast int64 to str for neighboring_atom_id column
    df['neighboring_atom_id'] = df['neighboring_atom_id'].astype(str)

    return df
