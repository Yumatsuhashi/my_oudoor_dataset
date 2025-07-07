""""
look_at_csi.py
"""

import sys, re, h5py, numpy as np
DEFAULT_H5 = "/home/matsuhashi/Documents/matsuhashi2/my_outdoor_dataset/out/0627_s4_straight/csi_traj_2.h5"
LABELS_5D  = ["time", "RB", "TxTot", "RxTot", "real/imag"]

# ───────────────────────────────────── helpers
def parse_index_arg(arg: str, max_len: int):
    """

    使い方:
    # 既定 (time=0, rb=0)
    $ python look_at_csi.py

    # time=2, rb=1
    $ python look_at_csi.py file.h5 2 1

    # time = 0,3,7   と   rb = 4,5 をまとめて
    $ python look_at_csi.py file.h5 0,3,7 4,5

    # time = 0〜4 連続 (0:4) , rb = 10,11,12
    $ python look_at_csi.py file.h5 0:4 10-12

    # python3 look_at_csi.py out/0622_s200/csi_traj_1.h5 0:3 1


    """

    if arg is None:
        return [0]
    out = []
    for token in arg.split(','):
        m = re.fullmatch(r'(\d+)[\-\:](\d+)', token)
        if m:  # range
            a, b = map(int, m.groups())
            if token.find('-') != -1:      # inclusive
                rng = range(a, b+1)
            else:                          # slice-like a:b
                rng = range(a, b)
            out.extend(rng)
        else:
            out.append(int(token))
    # 範囲チェック
    for v in out:
        if not 0 <= v < max_len:
            raise ValueError(f"index {v} out of range 0–{max_len-1}")
    return sorted(set(out))

def find_csi_dataset(hf, key="csi"):   # ★デフォルトを変更
    if key in hf:
        return hf[key]
    for v in hf.values():
        if isinstance(v, h5py.Group) and key in v:
            return v[key]
    raise KeyError(f"'{key}' dataset が見つかりません")

def cplx_formatter():
    return {"complex_kind": lambda z: f"{z.real:+.3f}{z.imag:+.3f}j"}

def show_slice(dset, t, rb):
    real = dset[t, rb, ..., 0]
    imag = dset[t, rb, ..., 1]
    mat  = real + 1j*imag
    print(f"\n--- time={t}, RB={rb} ---")
    print(np.array2string(mat, formatter=cplx_formatter(), separator=", "))

# ───────────────────────────────────── main
if __name__ == "__main__":
    # 引数取得
    path   = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_H5
    arg_t  = sys.argv[2] if len(sys.argv) > 2 else None
    arg_rb = sys.argv[3] if len(sys.argv) > 3 else None

    with h5py.File(path, "r") as hf:
        dset = find_csi_dataset(hf)
        if dset.ndim != 5:
            sys.exit("5-D CSI (time,RB,Tx,Rx,2) にだけ対応しています。")

        times = parse_index_arg(arg_t,  dset.shape[0])
        rbs   = parse_index_arg(arg_rb, dset.shape[1])

        # ヘッダ表示
        print(f"File         : {path}")
        print(f"Dataset path : {dset.name}")
        print(f"Shape        : {dset.shape}")
        for lbl, sz in zip(LABELS_5D, dset.shape):
            print(f"{lbl:10}: {sz}")

        # スライス出力
        for t in times:
            for rb in rbs:
                show_slice(dset, t, rb)