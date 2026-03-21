"""
step1b_groundtruth.py
---------------------
sample_1000.csv'deki her kayıt için SCP kodlarından
lead-level ground truth üretir.

Çıktı: data/ground_truth.csv

Kullanım:
    conda activate efa-ecg
    python step1b_groundtruth.py
"""

import ast
import json
import os

import pandas as pd

# ------------------------------------------------------------------
# Ayarlar
# ------------------------------------------------------------------
DATA_DIR      = "./data"
SAMPLE_CSV    = os.path.join(DATA_DIR, "sample_1000.csv")
OUTPUT_CSV    = os.path.join(DATA_DIR, "ground_truth.csv")
OUTPUT_JSON   = os.path.join(DATA_DIR, "ground_truth.json")

# ------------------------------------------------------------------
# SCP kodu → klinik olarak ilgili lead seti mapping
# Makaledeki Tablo II (Appendix B) ile birebir örtüşür
# ------------------------------------------------------------------
SCP_TO_LEADS = {
    # ---- NORM ----
    "NORM":  [],   # baseline, lead yok

    # ---- MI — İnferior ----
    "IMI":   ["II", "III", "aVF"],
    "IPLMI": ["II", "III", "aVF"],   # inferoposterolateral
    "ILMI":  ["II", "III", "aVF"],   # inferolateral
    "IPMI":  ["II", "III", "aVF"],   # inferoposterior

    # ---- MI — Anterior ----
    "AMI":   ["V1", "V2", "V3", "V4"],
    "ASMI":  ["V1", "V2", "V3", "V4"],  # anteroseptal
    "ALMI":  ["V1", "V2", "V3", "V4", "V5", "V6"],  # anterolateral
    "AAMI":  ["V3", "V4", "V5", "V6"],  # anteroapical
    "LMI":   ["I", "aVL", "V5", "V6"],  # lateral

    # ---- MI — Posterior ----
    "PMI":   ["V1", "V2"],   # reciprocal

    # ---- STTC ----
    "STD_":  ["II", "V4", "V5", "V6"],   # ST depression (genel)
    "STE_":  ["II", "III", "aVF"],        # ST elevation (genel)
    "ISCA":  ["I", "aVL", "V5", "V6"],   # ischemia anterior
    "ISCI":  ["II", "III", "aVF"],        # ischemia inferior
    "ISC_":  ["II", "III", "aVF", "V4", "V5", "V6"],
    "INVT":  ["V1", "V2", "V3", "V4"],   # T inversiyonu
    "LVOLT": ["I", "aVL", "V5", "V6"],
    "RVOLT": ["V1", "V2"],
    "TAB_":  ["II", "V5", "V6"],          # T anormallikleri
    "LNGQT": ["II", "V5"],               # uzun QT

    # ---- CD — İletim bozuklukları ----
    "LBBB":  ["V1", "V2", "V5", "V6"],
    "RBBB":  ["V1", "V2"],
    "LAFB":  ["I", "aVL"],               # left anterior fascicular block
    "LPFB":  ["II", "III", "aVF"],       # left posterior fascicular block
    "LPR":   ["II", "aVF"],              # uzun PR
    "IVCD":  ["V1", "V2", "V5", "V6"],  # intraventricular conduction delay
    "AVB":   ["II", "aVF"],              # AV blok
    "1AVB":  ["II", "aVF"],
    "2AVB":  ["II", "aVF"],
    "3AVB":  ["II", "aVF"],
    "WPW":   ["II", "III", "aVF", "V1", "V2"],

    # ---- HYP — Hipertrofi ----
    "LVH":   ["I", "aVL", "V5", "V6"],
    "RVH":   ["V1", "V2"],
    "LAO/LAE": ["II", "aVL"],           # sol atriyal genişleme
    "RAO/RAE": ["II", "V1"],            # sağ atriyal genişleme
    "SEHYP": ["V1", "V2", "V5", "V6"], # septal hipertrofi
}

# Superclass fallback: spesifik SCP kodu eşleşmezse
SUPERCLASS_FALLBACK = {
    "NORM": [],
    "MI":   ["II", "III", "aVF", "V1", "V2", "V3", "V4"],
    "STTC": ["II", "III", "aVF", "V4", "V5", "V6"],
    "CD":   ["V1", "V2", "V5", "V6"],
    "HYP":  ["I", "aVL", "V1", "V2", "V5", "V6"],
}

ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# ------------------------------------------------------------------
# Yardımcı fonksiyon
# ------------------------------------------------------------------
def get_lead_set(scp_codes: dict, superclass: str) -> list[str]:
    """
    SCP kodlarından lead seti üretir.
    - Her SCP kodu için mapping tablosuna bak
    - Bulunan tüm leadleri birleştir (union)
    - Hiç eşleşme yoksa superclass fallback kullan
    """
    leads = set()

    for code in scp_codes.keys():
        code_upper = code.upper().strip()
        if code_upper in SCP_TO_LEADS:
            leads.update(SCP_TO_LEADS[code_upper])

    # Eşleşme bulunamadıysa fallback
    if not leads and superclass in SUPERCLASS_FALLBACK:
        leads = set(SUPERCLASS_FALLBACK[superclass])

    # ALL_LEADS sırasını koru
    return [l for l in ALL_LEADS if l in leads]


def leads_to_binary(lead_list: list[str]) -> dict[str, int]:
    """Lead listesini binary mask dict'e çevirir."""
    return {l: (1 if l in lead_list else 0) for l in ALL_LEADS}


# ------------------------------------------------------------------
# Ana işlem
# ------------------------------------------------------------------
print("[1/3] sample_1000.csv okunuyor...")
sample_df = pd.read_csv(SAMPLE_CSV, index_col="ecg_id")
sample_df["scp_codes"] = sample_df["scp_codes"].apply(ast.literal_eval)
print(f"      {len(sample_df)} kayıt yüklendi.")

print("[2/3] Lead-level ground truth hesaplanıyor...")
records = []
for ecg_id, row in sample_df.iterrows():
    lead_list = get_lead_set(row["scp_codes"], row["superclass"])
    binary    = leads_to_binary(lead_list)

    record = {
        "ecg_id":     ecg_id,
        "superclass": row["superclass"],
        "scp_codes":  json.dumps(row["scp_codes"]),
        "gt_leads":   json.dumps(lead_list),
    }
    # Binary mask sütunları
    for l in ALL_LEADS:
        record[f"gt_{l}"] = binary[l]

    records.append(record)

gt_df = pd.DataFrame(records).set_index("ecg_id")

# ------------------------------------------------------------------
# İstatistik özeti
# ------------------------------------------------------------------
print("\n      Ground truth özeti (ortalama aktif lead sayısı):")
for sc in ["NORM", "MI", "STTC", "CD", "HYP"]:
    sub = gt_df[gt_df["superclass"] == sc]
    lead_cols = [f"gt_{l}" for l in ALL_LEADS]
    avg_leads = sub[lead_cols].sum(axis=1).mean()
    print(f"      {sc:6s}: ort. {avg_leads:.1f} aktif lead")

# ------------------------------------------------------------------
# Kaydet
# ------------------------------------------------------------------
print("\n[3/3] Kaydediliyor...")
gt_df.to_csv(OUTPUT_CSV)
print(f"      CSV  → {OUTPUT_CSV}")

# JSON formatında da kaydet (rendering scripti için kullanışlı)
gt_json = {}
for ecg_id, row in gt_df.iterrows():
    gt_json[str(ecg_id)] = {
        "superclass": row["superclass"],
        "gt_leads":   json.loads(row["gt_leads"]),
    }
with open(OUTPUT_JSON, "w") as f:
    json.dump(gt_json, f, indent=2)
print(f"      JSON → {OUTPUT_JSON}")

print(f"\n✓ Ground truth tamamlandı. {len(gt_df)} kayıt işlendi.")
