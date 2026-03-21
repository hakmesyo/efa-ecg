"""
step1a_sampling.py
------------------
PTB-XL veri setinden 5 superclass'a göre stratified örnekleme yapar.
Her superclass'tan 200 kayıt seçilir → toplam 1000 kayıt.

Çıktı: data/sample_1000.csv

Kullanım:
    conda activate efa-ecg
    python step1a_sampling.py
"""

import ast
import json
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Ayarlar
# ------------------------------------------------------------------
PTB_XL_DIR   = "./data/ptb-xl"           # PTB-XL klasörü
OUTPUT_DIR   = "./data"                   # Çıktı klasörü
N_PER_CLASS  = 200                        # Her superclass'tan kaç kayıt
RANDOM_SEED  = 42                         # Tekrarlanabilirlik için
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "sample_1000.csv")

# PTB-XL 5 superclass tanımı
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ------------------------------------------------------------------
# 1. ptbxl_database.csv oku
# ------------------------------------------------------------------
print("[1/4] ptbxl_database.csv okunuyor...")
df = pd.read_csv(os.path.join(PTB_XL_DIR, "ptbxl_database.csv"), index_col="ecg_id")
print(f"      Toplam kayıt: {len(df)}")

# ------------------------------------------------------------------
# 2. scp_codes sütununu parse et (string → dict)
# ------------------------------------------------------------------
print("[2/4] SCP kodları parse ediliyor...")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

# ------------------------------------------------------------------
# 3. scp_statements.csv'den superclass bilgisi al
# ------------------------------------------------------------------
print("[3/4] Superclass etiketleri atanıyor...")
scp_df = pd.read_csv(os.path.join(PTB_XL_DIR, "scp_statements.csv"), index_col=0)

# Her kayıt için hangi superclass'a ait olduğunu bul
def get_superclass(scp_codes: dict) -> str | None:
    """
    Kayıttaki SCP kodlarını superclass'lara map eder.
    Birden fazla superclass varsa en yüksek likelihood'a sahip olanı döner.
    NORM için özel durum: sadece NORM kodu varsa NORM döner.
    """
    best_class = None
    best_likelihood = -1

    for code, likelihood in scp_codes.items():
        if code not in scp_df.index:
            continue
        superclass = scp_df.loc[code, "diagnostic_class"]
        if superclass not in SUPERCLASSES:
            continue
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_class = superclass

    return best_class

df["superclass"] = df["scp_codes"].apply(get_superclass)

# Superclass dağılımını göster
print("\n      Superclass dağılımı (tüm veri seti):")
dist = df["superclass"].value_counts()
for sc in SUPERCLASSES:
    count = dist.get(sc, 0)
    print(f"      {sc:6s}: {count:5d} kayıt")
print(f"      Etiketlenemedi: {df['superclass'].isna().sum()} kayıt")

# ------------------------------------------------------------------
# 4. Stratified örnekleme
# ------------------------------------------------------------------
print(f"\n[4/4] Her superclass'tan {N_PER_CLASS} kayıt seçiliyor (seed={RANDOM_SEED})...")

rng = np.random.default_rng(RANDOM_SEED)
sampled_frames = []

for sc in SUPERCLASSES:
    pool = df[df["superclass"] == sc].copy()
    if len(pool) < N_PER_CLASS:
        print(f"      UYARI: {sc} için yeterli kayıt yok "
              f"({len(pool)} < {N_PER_CLASS}). Tamamı alındı.")
        sampled = pool
    else:
        idx = rng.choice(len(pool), size=N_PER_CLASS, replace=False)
        sampled = pool.iloc[idx]
    sampled_frames.append(sampled)
    print(f"      {sc:6s}: {len(sampled)} kayıt seçildi")

sample_df = pd.concat(sampled_frames).sort_index()

# ------------------------------------------------------------------
# Kullanılacak sütunları seç ve kaydet
# ------------------------------------------------------------------
cols = ["superclass", "scp_codes", "filename_lr", "filename_hr",
        "patient_id", "age", "sex", "recording_date"]
cols = [c for c in cols if c in sample_df.columns]
sample_df = sample_df[cols]

os.makedirs(OUTPUT_DIR, exist_ok=True)
sample_df.to_csv(OUTPUT_CSV)

print(f"\n✓ Örnekleme tamamlandı.")
print(f"  Toplam: {len(sample_df)} kayıt")
print(f"  Çıktı : {OUTPUT_CSV}")
print(f"\n  Superclass dağılımı (örneklem):")
for sc in SUPERCLASSES:
    count = (sample_df["superclass"] == sc).sum()
    print(f"  {sc:6s}: {count} kayıt")