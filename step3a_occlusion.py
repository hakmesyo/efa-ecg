"""
step3a_occlusion.py
-------------------
LLaVA-v1.6-Mistral-7B ile lead-structured occlusion attribution.

Gereksinimler:
    - GPU (min 16 GB VRAM, FP16)
    - torch>=2.2.0+cu118
    - transformers>=4.44.0

Kullanim:
    python step3a_occlusion.py
    python step3a_occlusion.py --n_ecg 200
    python step3a_occlusion.py --per_class 40

Cikti:
    results/llava_mistral_occlusion.csv
    results/occ_maps/<ecg_id>_occ.npy
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# ------------------------------------------------------------------
# Argümanlar
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--images_dir',  default='./data/images')
parser.add_argument('--sample_csv',  default='./data/sample_1000.csv')
parser.add_argument('--gt_json',     default='./data/ground_truth.json')
parser.add_argument('--coords_json', default='./data/panel_coords.json')
parser.add_argument('--output_dir',  default='./results')
parser.add_argument('--n_ecg',       type=int, default=None,
                    help='Kac ECG isleneceği (varsayilan: tumu)')
parser.add_argument('--per_class',   type=int, default=None,
                    help='Superclass basi kac ECG')
args = parser.parse_args()

LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
OUTPUT_CSV  = os.path.join(args.output_dir, 'llava_mistral_occlusion.csv')
OCC_MAP_DIR = os.path.join(args.output_dir, 'occ_maps')
os.makedirs(OCC_MAP_DIR, exist_ok=True)

MODEL_NAME = 'llava-hf/llava-v1.6-mistral-7b-hf'

PROMPT = (
    "You are an expert cardiologist analyzing a 12-lead ECG image.\n"
    "The ECG is displayed in standard 4x3 layout:\n"
    "- Row 1: leads I, aVR, V1, V4\n"
    "- Row 2: leads II, aVL, V2, V5\n"
    "- Row 3: leads III, aVF, V3, V6\n\n"
    "Please provide a structured interpretation:\n"
    "1. PRIMARY DIAGNOSIS: Most likely diagnosis.\n"
    "2. KEY FINDINGS: ECG findings with specific lead names.\n"
    "3. CONFIDENCE: [Very High / High / Moderate / Low / Very Low]"
)

# ------------------------------------------------------------------
# Veri yükle
# ------------------------------------------------------------------
print("[1/4] Veriler yukleniyor...")
sample_df = pd.read_csv(args.sample_csv)
if 'ecg_id' in sample_df.columns:
    sample_df = sample_df.set_index('ecg_id')

with open(args.coords_json) as f:
    panel_coords = json.load(f)

# Seçim
if args.per_class:
    selected = []
    for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        ids = sample_df[sample_df['superclass'] == sc].index.tolist()
        selected.extend(ids[:args.per_class])
elif args.n_ecg:
    selected = sample_df.index.tolist()[:args.n_ecg]
else:
    selected = sample_df.index.tolist()

# Daha önce işlenenler
completed_ids = set()
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    completed_ids = set(done_df['ecg_id'].astype(str).tolist())
    print(f"   Daha once tamamlanan: {len(completed_ids)} kayit")

todo = [eid for eid in selected if str(eid) not in completed_ids]
print(f"   Secilen: {len(selected)} ECG")
print(f"   Islenecek: {len(todo)} ECG")

if not todo:
    print("Tum kayitlar zaten islendi.")
    exit(0)

# ------------------------------------------------------------------
# Model yükle
# ------------------------------------------------------------------
print("\n[2/4] Model yukleniyor...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("UYARI: GPU bulunamadi, CPU kullaniliyor (cok yavas olacak)")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    size={'shortest_edge': 336}
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map='auto'
)
model.eval()

if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {used:.1f} / {total:.1f} GB")

# ------------------------------------------------------------------
# Confidence skoru
# ------------------------------------------------------------------
def get_confidence_score(image):
    conv = f"[INST] <image>\n{PROMPT} [/INST]"
    inputs = processor(
        text=conv,
        images=image,
        return_tensors='pt'
    ).to(device, torch.float16)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )

    response = processor.decode(
        out[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()

    high_words = ['acute', 'stemi', 'infarction', 'fibrillation',
                  'flutter', 'block', 'hypertrophy']
    low_words  = ['normal', 'no signs', 'unremarkable', 'within normal']

    h = sum(1 for w in high_words if w in response)
    if h >= 2: return 0.85
    if h == 1: return 0.75
    if any(w in response for w in low_words): return 0.60
    return 0.65


def apply_occlusion(image, coords, lead):
    img_arr = np.array(image).copy()
    if lead not in coords:
        return image
    x1, y1, x2, y2 = [int(v) for v in coords[lead]]
    img_arr[y1:y2, x1:x2] = 128  # neutral gray fill
    return Image.fromarray(img_arr)


def compute_occlusion_scores(image, coords):
    baseline_conf = get_confidence_score(image)
    scores = {}
    for lead in LEADS:
        if lead not in coords:
            scores[f'occ_{lead}'] = 0.0
            continue
        occ_image = apply_occlusion(image, coords, lead)
        occ_conf  = get_confidence_score(occ_image)
        scores[f'occ_{lead}'] = float(baseline_conf - occ_conf)
    return scores


# ------------------------------------------------------------------
# Ana döngü
# ------------------------------------------------------------------
print("\n[3/4] Occlusion attribution hesaplaniyor...")
results = []

for ecg_id in tqdm(todo, desc='Occlusion'):
    ecg_id_str = str(ecg_id)
    superclass = sample_df.loc[int(ecg_id), 'superclass'] \
        if int(ecg_id) in sample_df.index else 'UNKNOWN'

    image_path = os.path.join(args.images_dir, f'{ecg_id}.png')
    if not os.path.exists(image_path):
        continue

    try:
        image  = Image.open(image_path).convert('RGB')
        coords = panel_coords.get(ecg_id_str,
                 panel_coords.get(str(int(ecg_id)), {}))

        occ_scores = compute_occlusion_scores(image, coords)

        # numpy array kaydet
        occ_array = np.array([occ_scores[f'occ_{l}'] for l in LEADS])
        np.save(os.path.join(OCC_MAP_DIR, f'{ecg_id}_occ.npy'), occ_array)

        row = {'ecg_id': ecg_id, 'superclass': superclass,
               'error': None, **occ_scores}

    except Exception as e:
        row = {'ecg_id': ecg_id, 'superclass': superclass,
               'error': str(e)[:200],
               **{f'occ_{l}': None for l in LEADS}}
        print(f"\nHATA {ecg_id}: {str(e)[:80]}")

    results.append(row)

    # Her 20 kayıtta bir kaydet
    if len(results) % 20 == 0:
        df_new = pd.DataFrame(results)
        if os.path.exists(OUTPUT_CSV):
            pd.concat([pd.read_csv(OUTPUT_CSV), df_new],
                      ignore_index=True).to_csv(OUTPUT_CSV, index=False)
        else:
            df_new.to_csv(OUTPUT_CSV, index=False)
        results = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Son batch
if results:
    df_new = pd.DataFrame(results)
    if os.path.exists(OUTPUT_CSV):
        pd.concat([pd.read_csv(OUTPUT_CSV), df_new],
                  ignore_index=True).to_csv(OUTPUT_CSV, index=False)
    else:
        df_new.to_csv(OUTPUT_CSV, index=False)

# ------------------------------------------------------------------
# Özet
# ------------------------------------------------------------------
print("\n[4/4] Ozet")
df = pd.read_csv(OUTPUT_CSV)
print(f"   Toplam kayit  : {len(df)}")
print(f"   Hatali        : {df['error'].notna().sum()}")
print(f"   Cikti         : {OUTPUT_CSV}")
print(f"   Occ maps      : {OCC_MAP_DIR}/")
print("\nTamamlandi!")