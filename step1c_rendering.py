"""
step1c_rendering.py
-------------------
sample_1000.csv'deki her kayıt için PTB-XL 1D sinyalini
standart 12-lead 4x3 layout ile 300 DPI PNG'ye dönüştürür.

Çıktı: data/images/<ecg_id>.png  (1000 adet)

Kullanım:
    conda activate efa-ecg
    python step1c_rendering.py

Notlar:
    - records100 (100 Hz) kullanılır — daha hızlı render
    - 4x3 layout: satır1=[I,aVR,V1,V4], satır2=[II,aVL,V2,V5],
                  satır3=[III,aVF,V3,V6]
    - Klinik standart: siyah sinyal, beyaz arka plan, grid
    - Her lead panelinin koordinatları ground_truth ile eşleşir
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI olmadan render
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wfdb
from tqdm import tqdm

# ------------------------------------------------------------------
# Ayarlar
# ------------------------------------------------------------------
PTB_XL_DIR  = "./data/ptb-xl"
SAMPLE_CSV  = "./data/sample_1000.csv"
OUTPUT_DIR  = "./data/images"
GT_JSON     = "./data/ground_truth.json"

DPI         = 300
FIG_W_INCH  = 11.0    # A4 yatay benzeri
FIG_H_INCH  = 8.5
N_COLS      = 4
N_ROWS      = 3
SIGNAL_HZ   = 100     # records100 kullanıyoruz

# Klinik renk şeması
BG_COLOR    = "white"
GRID_COLOR  = "#ffb3b3"   # açık kırmızı grid (klinik EKG kağıdı)
SIGNAL_COLOR = "black"
LEAD_LABEL_COLOR = "#333333"

# 4x3 layout — lead sırası
LAYOUT = [
    ["I",   "aVR", "V1", "V4"],   # satır 0
    ["II",  "aVL", "V2", "V5"],   # satır 1
    ["III", "aVF", "V3", "V6"],   # satır 2
]

# PTB-XL lead sırası (wfdb'den gelir)
PTBXL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

# ------------------------------------------------------------------
# Lead panel koordinatları — ground_truth ile eşleşmesi için
# JSON olarak da kaydedilir
# ------------------------------------------------------------------
def compute_panel_coords(fig_w_px, fig_h_px):
    """
    Her lead panelinin piksel koordinatlarını hesaplar.
    Çıktı: {lead_name: {"x0":, "y0":, "x1":, "y1":}} piksel cinsinden
    """
    panel_w = fig_w_px / N_COLS
    panel_h = fig_h_px / N_ROWS
    coords = {}
    for row_idx, row_leads in enumerate(LAYOUT):
        for col_idx, lead in enumerate(row_leads):
            x0 = col_idx * panel_w
            y0 = row_idx * panel_h
            coords[lead] = {
                "x0": int(x0), "y0": int(y0),
                "x1": int(x0 + panel_w), "y1": int(y0 + panel_h),
                "row": row_idx, "col": col_idx
            }
    return coords


def render_ecg(record_path, ecg_id, output_path):
    """
    Tek bir ECG kaydını 300 DPI PNG olarak render eder.
    """
    # Sinyali oku
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"  HATA: {ecg_id} okunamadı — {e}")
        return False

    signal = record.p_signal          # shape: (n_samples, 12)
    fs     = record.fs                 # örnekleme frekansı

    # Lead → index mapping
    lead_to_idx = {name: i for i, name in enumerate(PTBXL_LEAD_ORDER)}

    # Zaman ekseni (saniye)
    n_samples = signal.shape[0]
    t = np.linspace(0, n_samples / fs, n_samples)

    # Figure oluştur
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(FIG_W_INCH, FIG_H_INCH),
        facecolor=BG_COLOR
    )
    fig.subplots_adjust(
        left=0.04, right=0.99,
        top=0.93, bottom=0.04,
        hspace=0.35, wspace=0.15
    )

    for row_idx, row_leads in enumerate(LAYOUT):
        for col_idx, lead_name in enumerate(row_leads):
            ax = axes[row_idx][col_idx]

            # Lead sinyali
            lead_idx = lead_to_idx.get(lead_name, 0)
            sig = signal[:, lead_idx]

            # NaN/inf temizle
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize et (görsel için, ±2 mV aralığı)
            sig_plot = np.clip(sig, -2.0, 2.0)

            # Grid çiz (klinik EKG kağıdı)
            ax.set_facecolor(BG_COLOR)
            ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="-")

            # Büyük grid (5mm = 0.2 mV ve 0.2 s)
            ax.set_xticks(np.arange(0, t[-1], 0.2), minor=False)
            ax.set_yticks(np.arange(-2.0, 2.1, 0.5), minor=False)
            ax.grid(True, which="major", color=GRID_COLOR,
                    linewidth=0.8, linestyle="-", alpha=0.7)

            # Küçük grid (1mm = 0.04 mV ve 0.04 s)
            ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
            ax.set_yticks(np.arange(-2.0, 2.1, 0.1), minor=True)
            ax.grid(True, which="minor", color=GRID_COLOR,
                    linewidth=0.3, linestyle="-", alpha=0.4)

            # Sinyali çiz
            ax.plot(t, sig_plot, color=SIGNAL_COLOR,
                    linewidth=0.6, rasterized=False)

            # Baseline
            ax.axhline(0, color="#888888", linewidth=0.4, linestyle="--")

            # Eksen limitleri
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(-2.2, 2.2)

            # Lead etiketi
            ax.set_title(lead_name, fontsize=7, fontweight="bold",
                         color=LEAD_LABEL_COLOR, pad=2)

            # Eksen temizle
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")

    # ECG ID başlık
    fig.suptitle(f"ECG ID: {ecg_id}", fontsize=8,
                 color="#555555", y=0.98)

    # Kaydet
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR, format="png")
    plt.close(fig)
    return True


# ------------------------------------------------------------------
# Worker fonksiyonu (multiprocessing için top-level olmalı)
# ------------------------------------------------------------------
def render_worker(args):
    """Tek bir ECG'yi render eden worker — multiprocessing için."""
    ecg_id, filename, output_path = args

    if os.path.exists(output_path):
        return "skipped"

    if not filename:
        return "fail"

    record_path = os.path.join(PTB_XL_DIR, filename)
    ok = render_ecg(record_path, ecg_id, output_path)
    return "ok" if ok else "fail"


# ------------------------------------------------------------------
# Ana döngü
# ------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp

    N_WORKERS = min(8, mp.cpu_count())   # 8 çekirdeğin tamamını kullan

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/3] sample_1000.csv okunuyor...")
    sample_df = pd.read_csv(SAMPLE_CSV, index_col="ecg_id")
    print(f"      {len(sample_df)} kayıt bulundu.")

    # Panel koordinatlarını hesapla ve kaydet
    fig_w_px = int(FIG_W_INCH * DPI)
    fig_h_px = int(FIG_H_INCH * DPI)
    panel_coords = compute_panel_coords(fig_w_px, fig_h_px)
    coords_path = "./data/panel_coords.json"
    with open(coords_path, "w") as f:
        json.dump(panel_coords, f, indent=2)
    print(f"      Panel koordinatları → {coords_path}")

    # İş listesi hazırla
    tasks = []
    for ecg_id, row in sample_df.iterrows():
        output_path = os.path.join(OUTPUT_DIR, f"{ecg_id}.png")
        filename    = row.get("filename_lr", "")
        tasks.append((ecg_id, filename, output_path))

    print(f"\n[2/3] Rendering başlıyor ({len(tasks)} kayıt, "
          f"{DPI} DPI, {N_WORKERS} çekirdek)...")
    print(f"      Çıktı klasörü: {OUTPUT_DIR}\n")

    success = 0
    fail    = 0
    skipped = 0

    with mp.Pool(processes=N_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(render_worker, tasks),
            total=len(tasks),
            desc="Rendering",
            unit="ECG"
        ):
            if result == "ok":
                success += 1
            elif result == "skipped":
                skipped += 1
            else:
                fail += 1

    print(f"\n[3/3] Rendering tamamlandı.")
    print(f"      Başarılı : {success}")
    print(f"      Atlandı  : {skipped} (zaten mevcut)")
    print(f"      Hatalı   : {fail}")
    print(f"\n✓ {success + skipped} PNG dosyası → {OUTPUT_DIR}/")
    print(f"  Panel koordinatları → {coords_path}")