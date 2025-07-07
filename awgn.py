#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awgn_ls_estimation.py  –  既存 CSI(.h5) に AWGN を付加し LS 推定でノイズ付き CSI を生成するスクリプト
----------------------------------------------------------------------------------
仕様
  • DATA_DIR 配下の csi_traj_<n>.h5 を順次処理
  • 入力データセットは形状 [time, RB, TxTot, RxTot, 2] (実・虚)
  • 合計送信電力 1 [W] を TxTot 本のアンテナへ等配分 （x ベクトル）
  • 所望 SNR[dB] を満たす AWGN を加え受信信号 y = Hx + n を生成
  • LS 推定 \hat{H}_{LS} = y x^H (x^H x = 1) を計算
  • 以下 2 データセットを出力 HDF5 ファイルへ保存
      - "y_awgn"  : 形状 [time, RB, TxTot, RxTot, 2]
      - "csi_hat" : 同上 (ノイズ付き推定チャネル)
  • 出力ファイル名 : csi_traj_<n>_awgn.h5 （既存なら上書き）
  • 壊れた HDF5 等はスキップし理由を表示
  • --seed で乱数シード指定 (デフォルト 42)
拡張
  • TODO : SNR_LIST_DB を回して複数 SNR を同時生成
"""

import os
import sys
import argparse
from typing import Tuple, List

import h5py
import numpy as np
from tqdm import tqdm

# ───────────────────── ユーザ設定 ─────────────────────
DATA_DIR   = "/home/matsuhashi/Documents/matsuhashi2/my_outdoor_dataset/out/0626_s200_straight/"   # 入力フォルダ
OUTPUT_DIR = "/home/matsuhashi/Documents/matsuhashi2/my_outdoor_dataset/out/0704_s200_straight_awgn/"  # 出力フォルダ
MAX_FILE_IDX = 20            # 探索上限 (csi_traj_0 〜 csi_traj_MAX_FILE_IDX)
DATASET_KEY  = "csi"         # 入力データセット名
SNR_DB       = 20.0          # 所望 SNR[dB]（単一値）
# SNR_LIST_DB = [-5, 0, 5, 10, 15, 20]  # ←複数 SNR 生成時に使用
# ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add AWGN and perform LS channel estimation on CSI HDF5 files.")
    parser.add_argument("--seed", type=int, default=42, help="numpy random seed (default: 42)")
    return parser.parse_args()

# ───────────────────── コア処理関数 ────────────────────

def add_awgn_and_estimate(csi: np.ndarray, snr_db: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """受信信号 y と LS 推定 \hat{H} を生成して返す

    Parameters
    ----------
    csi : np.ndarray
        形状 [T, RB, TxTot, RxTot, 2] 実虚分離の理想 CSI
    snr_db : float
        所望 SNR [dB]
    rng : np.random.Generator
        乱数生成器

    Returns
    -------
    y_out : np.ndarray
        AWGN 付加後受信信号 (実虚分離, 同形状)
    h_hat_out : np.ndarray
        LS 推定チャネル (実虚分離, 同形状)
    """
    # 実虚 → 複素
    h_c = csi[..., 0] + 1j * csi[..., 1]                 # [T,RB,Tx,Rx]

    # 送信信号ベクトル x : 大きさ 1 を TxTot 本で均等配分
    tx_tot = h_c.shape[-2]
    x = np.ones(tx_tot, dtype=np.complex64) / np.sqrt(float(tx_tot))  # [Tx]
    x_h = np.conjugate(x)                                             # row ベクトル

    # 受信信号 z = Hx  (einsum で Tx を畳み込み)
    # h_c [...,Tx,Rx] × x[Tx] → z [...,Rx]
    z = np.einsum('...tr,t->...r', h_c, x)               # [T,RB,Rx]

    # 信号電力 (平均電力) P_sig (保持次元)
    p_sig = np.mean(np.abs(z) ** 2, axis=-1, keepdims=True)  # [T,RB,1]

    # SNR[dB] → 線形
    snr_lin = 10 ** (snr_db / 10.0)

    # 雑音分散 σ² = P_sig / SNR_lin
    sigma2 = p_sig / snr_lin                              # [T,RB,1]

    # AWGN 生成 (複素ガウス, 実部・虚部に σ²/2)
    noise_real = rng.normal(scale=np.sqrt(sigma2 / 2.0), size=z.shape)
    noise_imag = rng.normal(scale=np.sqrt(sigma2 / 2.0), size=z.shape)
    n = (noise_real + 1j * noise_imag).astype(np.complex64)

    # 受信信号 y
    y = z + n                                             # [T,RB,Rx]

    # -------- LS 推定 --------
    # x^H x = 1 なので単純に Outer Product
    # H_hat [...,Tx,Rx] = y[...,None,Rx] * x^H[Tx]
    h_hat = np.einsum('...r,t->...tr', y, x_h)            # [T,RB,Tx,Rx]

    # -------- 実虚分離して出力形式に整形 --------
    # y_full: Tx 軸に複写して元の形状と整合させる
    y_full = np.repeat(y[..., None, :], tx_tot, axis=-2)  # [T,RB,Tx,Rx]

    y_out = np.empty_like(csi, dtype=np.float32)
    h_hat_out = np.empty_like(csi, dtype=np.float32)

    y_out[..., 0] = y_full.real
    y_out[..., 1] = y_full.imag
    h_hat_out[..., 0] = h_hat.real
    h_hat_out[..., 1] = h_hat.imag

    return y_out, h_hat_out

# ───────────────────── メイン処理 ──────────────────────

def process_files(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    skipped: List[str] = []

    print(f"Input Dir : {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Target SNR: {SNR_DB} dB")
    print(f"Seed      : {args.seed}\n")

    for idx in tqdm(range(MAX_FILE_IDX + 1), desc="Processing"):
        in_path  = os.path.join(DATA_DIR,  f"csi_traj_{idx}.h5")
        out_path = os.path.join(OUTPUT_DIR, f"csi_traj_{idx}_awgn.h5")

        # (1) 存在確認
        if not os.path.exists(in_path):
            skipped.append(f"{in_path}: file not found")
            continue

        try:
            with h5py.File(in_path, "r", swmr=True) as fi:
                traj_key = next(iter(fi.keys()))           # 'traj_0' など
                if DATASET_KEY not in fi[traj_key]:
                    skipped.append(f"{in_path}: dataset '{DATASET_KEY}' not found")
                    continue
                csi = fi[traj_key][DATASET_KEY][...].astype(np.float32)

            # (2) AWGN 付加 + LS 推定
            y_awgn, csi_hat = add_awgn_and_estimate(csi, SNR_DB, rng)

            # (3) 保存 (上書き可)
            with h5py.File(out_path, "w") as fo:
                g = fo.create_group(traj_key)
                g.create_dataset("y_awgn",  data=y_awgn,  compression="lzf", shuffle=True)
                g.create_dataset("csi_hat", data=csi_hat, compression="lzf", shuffle=True)

            print(f"[OK] {in_path} → {out_path}")

        except (OSError, KeyError, ValueError) as e:
            skipped.append(f"{in_path}: {type(e).__name__}: {e}")
            continue

    # ---- 結果レポート ----
    print("\n===== skipped files =====")
    if skipped:
        for s in skipped:
            print("SKIP:", s)
    else:
        print("None")

# ───────────────────── エントリポイント ────────────────
if __name__ == "__main__":
    ns = parse_args()
    process_files(ns)
