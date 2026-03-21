"""
step4_efa.py
------------
Explanation-Attribution Faithfulness Auditor (EFA) skorunu hesaplar.

Girdiler:
    - results/gemini_outputs.csv
    - results/llava_mistral_outputs.csv
    - results/llava_mistral_occlusion.csv
    - data/ground_truth.csv
    - data/panel_coords.json

Çıktı:
    - results/efa_scores.csv

Kullanım:
    conda activate efa-ecg
    python step4_efa.py
"""

import os
import json
import re
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

# ------------------------------------------------------------------
# Ayarlar
# ------------------------------------------------------------------
RESULTS_DIR  = "./results"
DATA_DIR     = "./data"
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "efa_scores.csv")
ALPHA        = 0.5   # F = alpha * F_vis + (1-alpha) * F_txt

# Confidence threshold'ları (Danger Zone için)
TAU_C = 0.75   # 75. percentil → sonradan hesaplanacak
TAU_F = 0.25   # 25. percentil → sonradan hesaplanacak

ALL_LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# ------------------------------------------------------------------
# Lead NER parser
# ------------------------------------------------------------------
# Composite anatomical → lead mapping
ANATOMICAL_MAP = {
    'inferior':    ['II', 'III', 'aVF'],
    'anterior':    ['V1', 'V2', 'V3', 'V4'],
    'lateral':     ['I', 'aVL', 'V5', 'V6'],
    'posterior':   ['V1', 'V2'],
    'precordial':  ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
    'limb':        ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'],
    'septal':      ['V1', 'V2'],
    'apical':      ['V3', 'V4', 'V5'],
    'high lateral':['I', 'aVL'],
}

# Explicit lead patterns
LEAD_PATTERNS = {
    'I':   [r'\blead\s+I\b', r'\bI\b(?!\w)'],
    'II':  [r'\blead\s+II\b', r'\bII\b(?!\w)'],
    'III': [r'\blead\s+III\b', r'\bIII\b(?!\w)'],
    'aVR': [r'\baVR\b', r'\bAVR\b'],
    'aVL': [r'\baVL\b', r'\bAVL\b'],
    'aVF': [r'\baVF\b', r'\bAVF\b'],
    'V1':  [r'\bV1\b', r'\bV\s*1\b'],
    'V2':  [r'\bV2\b', r'\bV\s*2\b'],
    'V3':  [r'\bV3\b', r'\bV\s*3\b'],
    'V4':  [r'\bV4\b', r'\bV\s*4\b'],
    'V5':  [r'\bV5\b', r'\bV\s*5\b'],
    'V6':  [r'\bV6\b', r'\bV\s*6\b'],
}

def extract_leads_from_text(text: str) -> set:
    """NER: metinden lead referanslarını çıkar."""
    if not isinstance(text, str) or not text:
        return set()
    
    found = set()
    text_lower = text.lower()
    
    # Explicit lead patterns
    for lead, patterns in LEAD_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.add(lead)
                break
    
    # V1-V6 range patterns (e.g., "V1-V4", "V2 through V5")
    range_pat = re.findall(r'V(\d)\s*[-–through]+\s*V(\d)', text, re.IGNORECASE)
    for start, end in range_pat:
        for i in range(int(start), int(end)+1):
            if f'V{i}' in ALL_LEADS:
                found.add(f'V{i}')
    
    # Anatomical composite terms
    for term, leads in ANATOMICAL_MAP.items():
        if term in text_lower:
            found.update(leads)
    
    return found


# ------------------------------------------------------------------
# EFA metrik fonksiyonları
# ------------------------------------------------------------------
def compute_f_txt(response_text: str, gt_leads: list) -> tuple:
    """
    Textual faithfulness: NER ile çıkarılan lead'lerin GT ile overlap'i.
    Döndürür: (precision, recall, f1)
    """
    if not gt_leads or not isinstance(response_text, str):
        return 0.0, 0.0, 0.0
    
    pred_leads = extract_leads_from_text(response_text)
    gt_set = set(gt_leads)
    
    if not pred_leads:
        return 0.0, 0.0, 0.0
    
    tp = len(pred_leads & gt_set)
    
    precision = tp / len(pred_leads) if pred_leads else 0.0
    recall    = tp / len(gt_set) if gt_set else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    
    return precision, recall, f1


def compute_f_vis_occlusion(occ_row: pd.Series, gt_leads: list) -> float:
    """
    Visual faithfulness (occlusion): Yüksek occlusion skoru alan lead'lerin
    GT ile IoU'su.
    """
    if not gt_leads:
        return 0.0
    
    # Occlusion skorlarını al
    occ_scores = {}
    for lead in ALL_LEADS:
        col = f'occ_{lead}'
        if col in occ_row.index:
            occ_scores[lead] = float(occ_row[col]) if pd.notna(occ_row[col]) else 0.0
        else:
            occ_scores[lead] = 0.0
    
    if not occ_scores or max(occ_scores.values()) == 0:
        return 0.0
    
    # Threshold: ortalama üzerindeki leadler "attended" sayılır
    mean_score = np.mean(list(occ_scores.values()))
    attended = set(l for l, s in occ_scores.items() if s > mean_score)
    gt_set = set(gt_leads)
    
    if not attended:
        return 0.0
    
    intersection = len(attended & gt_set)
    union = len(attended | gt_set)
    
    return intersection / union if union > 0 else 0.0


def compute_efa(f_vis: float, f_txt: float, alpha: float = 0.5) -> float:
    """EFA composite score."""
    return alpha * f_vis + (1 - alpha) * f_txt

HIGH_WORDS = ['acute','stemi','infarction','fibrillation','flutter','block','hypertrophy']
LOW_WORDS  = ['normal','no signs','unremarkable','within normal']

def proxy_confidence(text: str) -> float:
    """LLaVA response'undan proxy confidence hesapla."""
    if not isinstance(text, str) or not text:
        return 0.65
    t = text.lower()
    h = sum(1 for w in HIGH_WORDS if w in t)
    if h >= 2: return 0.85
    if h == 1: return 0.75
    if any(w in t for w in LOW_WORDS): return 0.60
    return 0.65
LOW_WORDS  = ['normal','no signs','unremarkable','within normal']

def proxy_confidence(text: str) -> float:
    """LLaVA response'undan proxy confidence hesapla."""
    if not isinstance(text, str) or not text:
        return 0.65
    t = text.lower()
    h = sum(1 for w in HIGH_WORDS if w in t)
    if h >= 2: return 0.85
    if h == 1: return 0.75
    if any(w in t for w in LOW_WORDS): return 0.60
    return 0.65


# ------------------------------------------------------------------
# Veri yükle
# ------------------------------------------------------------------
print("[1/4] Veriler yükleniyor...")

gt_df = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"), index_col="ecg_id")
gt_df["gt_leads"] = gt_df["gt_leads"].apply(json.loads)

gemini_df = pd.read_csv(os.path.join(RESULTS_DIR, "gemini_outputs.csv"), quoting=1)
gemini_df = gemini_df.set_index("ecg_id")

mistral_df = pd.read_csv(os.path.join(RESULTS_DIR, "llava_mistral_outputs.csv"))
mistral_df = mistral_df.set_index("ecg_id")

# Occlusion (varsa yükle)
occ_path = os.path.join(RESULTS_DIR, "llava_mistral_occlusion.csv")
if os.path.exists(occ_path):
    occ_df = pd.read_csv(occ_path).set_index("ecg_id")
    print(f"   Occlusion: {len(occ_df)} kayıt yüklendi")
else:
    occ_df = pd.DataFrame()
    print("   Occlusion: henüz yok, F_vis=0 kullanılacak")

print(f"   Gemini: {len(gemini_df)} kayıt")
print(f"   LLaVA-Mistral: {len(mistral_df)} kayıt")
print(f"   Ground truth: {len(gt_df)} kayıt")

# ------------------------------------------------------------------
# EFA hesapla
# ------------------------------------------------------------------
print("\n[2/4] EFA skorları hesaplanıyor...")

records = []

for ecg_id, gt_row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="EFA"):
    gt_leads = gt_row["gt_leads"]
    superclass = gt_row["superclass"]

    # ---- Gemini ----
    if ecg_id in gemini_df.index:
        g_row = gemini_df.loc[ecg_id]
        g_txt = g_row.get("response_text", "")
        g_conf = float(g_row.get("confidence", 0.5)) if pd.notna(g_row.get("confidence")) else 0.5

        g_prec, g_rec, g_f_txt = compute_f_txt(g_txt, gt_leads)

        # Gemini occlusion map varsa
        g_f_vis = 0.0
        if ecg_id in occ_df.index:
            g_f_vis = compute_f_vis_occlusion(occ_df.loc[ecg_id], gt_leads)

        g_efa = compute_efa(g_f_vis, g_f_txt)

        records.append({
            "ecg_id":      ecg_id,
            "model":       "gemini-2.5-flash",
            "superclass":  superclass,
            "f_vis":       round(g_f_vis, 4),
            "f_txt":       round(g_f_txt, 4),
            "f_txt_prec":  round(g_prec, 4),
            "f_txt_rec":   round(g_rec, 4),
            "efa":         round(g_efa, 4),
            "confidence":  round(g_conf, 4),
        })

    # ---- LLaVA-Mistral ----
    if ecg_id in mistral_df.index:
        m_row = mistral_df.loc[ecg_id]
        m_txt = m_row.get("response_text", "")
        m_conf = proxy_confidence(m_txt)

        m_prec, m_rec, m_f_txt = compute_f_txt(m_txt, gt_leads)

        # Occlusion map varsa
        m_f_vis = 0.0
        if ecg_id in occ_df.index:
            m_f_vis = compute_f_vis_occlusion(occ_df.loc[ecg_id], gt_leads)

        m_efa = compute_efa(m_f_vis, m_f_txt)

        records.append({
            "ecg_id":      ecg_id,
            "model":       "llava-mistral-7b",
            "superclass":  superclass,
            "f_vis":       round(m_f_vis, 4),
            "f_txt":       round(m_f_txt, 4),
            "f_txt_prec":  round(m_prec, 4),
            "f_txt_rec":   round(m_rec, 4),
            "efa":         round(m_efa, 4),
            "confidence":  round(m_conf, 4),
        })

efa_df = pd.DataFrame(records)

# ------------------------------------------------------------------
# Danger Zone hesapla
# ------------------------------------------------------------------
print("\n[3/4] Danger Zone hesaplanıyor...")

for model_name in efa_df["model"].unique():
    mask = efa_df["model"] == model_name
    sub = efa_df[mask]

    tau_c = sub["confidence"].quantile(0.75)
    tau_f = sub["efa"].quantile(0.25)

    danger = (sub["confidence"] >= tau_c) & (sub["efa"] <= tau_f)
    efa_df.loc[mask, "danger_zone"] = danger.astype(int)
    efa_df.loc[mask, "tau_c"] = round(tau_c, 4)
    efa_df.loc[mask, "tau_f"] = round(tau_f, 4)

# ------------------------------------------------------------------
# Kaydet ve özet
# ------------------------------------------------------------------
print("\n[4/4] Kaydediliyor...")
os.makedirs(RESULTS_DIR, exist_ok=True)
efa_df.to_csv(OUTPUT_CSV, index=False)
print(f"   Çıktı: {OUTPUT_CSV}")
print(f"   Toplam: {len(efa_df)} kayıt")

print("\n" + "="*60)
print("EFA ÖZET RAPORU")
print("="*60)

for model_name in sorted(efa_df["model"].unique()):
    sub = efa_df[efa_df["model"] == model_name]
    dz = sub["danger_zone"].mean() * 100

    print(f"\n{model_name}")
    print(f"  Macro EFA    : {sub['efa'].mean():.3f}")
    print(f"  Macro F_vis  : {sub['f_vis'].mean():.3f}")
    print(f"  Macro F_txt  : {sub['f_txt'].mean():.3f}")
    print(f"  Confidence   : {sub['confidence'].mean():.3f}")
    print(f"  Danger Zone  : {dz:.1f}%")
    print(f"  Superclass breakdown:")

    for sc in ["NORM", "MI", "STTC", "CD", "HYP"]:
        sc_sub = sub[sub["superclass"] == sc]
        if len(sc_sub) > 0:
            print(f"    {sc:6s}: EFA={sc_sub['efa'].mean():.3f}  "
                  f"DZ={sc_sub['danger_zone'].mean()*100:.1f}%  "
                  f"(n={len(sc_sub)})")

print("\n✓ Tamamlandı!")