"""
step2a_gemini_inference.py
--------------------------
Her ECG PNG'sini Gemini 2.5 Flash API'ye göndererek
standart prompt ile doğal dil açıklaması üretir.
10 paralel worker ile ~12 dakikada 1000 kayıt işler.

Çıktı: results/gemini_outputs.csv

Kullanım:
    conda activate efa-ecg
    python step2a_gemini_inference.py

Notlar:
    - Yarıda kesilirse kaldığı yerden devam eder
    - 429 alırsa otomatik bekler ve tekrar dener
"""

import os
import time
import base64
import threading
import concurrent.futures

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ------------------------------------------------------------------
# Ayarlar
# ------------------------------------------------------------------
SAMPLE_CSV  = "./data/sample_1000.csv"
IMAGES_DIR  = "./data/images"
OUTPUT_DIR  = "./results"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "gemini_outputs.csv")

MODEL_NAME  = "gemini-2.5-flash"
API_KEY     = "AIzaSyAHRH6f1KdC2ETY97w9XlE2FuykNmlyNxU"   # <- kendi API key'ini buraya yapistir

N_WORKERS   = 10     # paralel thread sayisi
MAX_RETRIES = 5
RETRY_SLEEP = 30     # 429 hatasinda bekle (saniye)

# ------------------------------------------------------------------
# Standart prompt (Appendix D ile birebir)
# ------------------------------------------------------------------
ECG_PROMPT = """You are an expert cardiologist analyzing a 12-lead ECG image.
The ECG is displayed in standard 4x3 layout:
- Row 1: leads I, aVR, V1, V4
- Row 2: leads II, aVL, V2, V5
- Row 3: leads III, aVF, V3, V6

Please provide a structured interpretation of this ECG. Your response must include:

1. PRIMARY DIAGNOSIS: State the most likely diagnosis clearly.
2. KEY FINDINGS: Describe the specific ECG findings that support your diagnosis.
   For each finding, explicitly name the lead(s) where it is observed.
3. LEAD-SPECIFIC OBSERVATIONS: For each relevant lead, describe what you observe.
4. CONFIDENCE: State your confidence level as one of:
   [Very High / High / Moderate / Low / Very Low]
5. CLINICAL RECOMMENDATION: Brief clinical action recommendation.

Be specific about lead names (I, II, III, aVR, aVL, aVF, V1-V6).
Do not use vague anatomical references without specifying the leads."""

# ------------------------------------------------------------------
# Confidence parser
# ------------------------------------------------------------------
CONFIDENCE_MAP = {
    "very high": 0.95,
    "very low":  0.15,
    "high":      0.80,
    "moderate":  0.60,
    "low":       0.35,
}

def extract_confidence(text):
    t = text.lower()
    for key, val in CONFIDENCE_MAP.items():
        if key in t:
            return val
    return 0.50

# ------------------------------------------------------------------
# Thread-safe CSV writer
# ------------------------------------------------------------------
write_lock = threading.Lock()

def append_result(result):
    with write_lock:
        df_new = pd.DataFrame([result])
        if os.path.exists(OUTPUT_CSV):
            df_new.to_csv(OUTPUT_CSV, mode="a", header=False,
                          index=False, quoting=1, lineterminator="\n")
        else:
            df_new.to_csv(OUTPUT_CSV, mode="w", header=True,
                          index=False, quoting=1, lineterminator="\n")

# ------------------------------------------------------------------
# Tek kayit icin inference
# ------------------------------------------------------------------
def run_inference(args):
    ecg_id, superclass, mdl = args
    image_path = os.path.join(IMAGES_DIR, f"{ecg_id}.png")
    if not os.path.exists(image_path):
        return
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = mdl.generate_content(
                [ECG_PROMPT, {"mime_type": "image/png", "data": img_b64}],
                generation_config={"temperature": 0.1},
            )
            text = response.text
            result = {
                "ecg_id":        ecg_id,
                "model":         MODEL_NAME,
                "response_text": text,
                "confidence":    extract_confidence(text),
                "input_tokens":  response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "error":         "",
                "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
                "superclass":    superclass,
            }
            append_result(result)
            return
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = RETRY_SLEEP * attempt
                tqdm.write(f"  429 (ECG {ecg_id}), {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                tqdm.write(f"  HATA (ECG {ecg_id}): {err[:80]}")
                append_result({
                    "ecg_id": ecg_id, "model": MODEL_NAME,
                    "response_text": "", "confidence": None,
                    "input_tokens": None, "output_tokens": None,
                    "error": err[:200],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "superclass": superclass,
                })
                return

# ------------------------------------------------------------------
# Ana dongü
# ------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
print(f"Gemini API baglantisi kuruldu. Model: {MODEL_NAME}")

print("[1/3] sample_1000.csv okunuyor...")
sample_df = pd.read_csv(SAMPLE_CSV, index_col="ecg_id")

completed_ids = set()
if os.path.exists(OUTPUT_CSV):
    existing = pd.read_csv(OUTPUT_CSV, quoting=1)
    completed_ids = set(existing["ecg_id"].tolist())
    print(f"      Daha once tamamlanan: {len(completed_ids)} kayit - atlanacak.")

todo = [(ecg_id, row["superclass"], model)
        for ecg_id, row in sample_df.iterrows()
        if ecg_id not in completed_ids]

print(f"\n[2/3] Inference basliyor... Kalan: {len(todo)}, Worker: {N_WORKERS}")
print(f"      Tahmini sure: ~{len(todo)*1.5/60:.0f} dakika\n")

with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    list(tqdm(executor.map(run_inference, todo), total=len(todo),
              desc="Gemini Inference", unit="ECG"))

print(f"\n[3/3] Tamamlandi.")
final_df = pd.read_csv(OUTPUT_CSV, quoting=1)
print(f"      Toplam: {len(final_df)}, Hatali: {(final_df['error']!='').sum()}")
print(f"      Ort. confidence: {final_df['confidence'].mean():.3f}")
print(f"\nCikti -> {OUTPUT_CSV}")