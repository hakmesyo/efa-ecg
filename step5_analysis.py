"""
step5_analysis.py
-----------------
Occlusion tamamlandıktan sonra çalıştır.
1. EFA skorlarını güncelle (F_vis dahil)
2. Confidence-Faithfulness korelasyon hesapla
3. Danger Zone prevalence hesapla
4. Özet rapor üret

Kullanım:
    conda activate efa-ecg
    python step5_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "./results"
DATA_DIR    = "./data"

# ------------------------------------------------------------------
# 1. Güncel EFA skorlarını yükle
# ------------------------------------------------------------------
print("[1/4] EFA skorları yükleniyor...")
efa_df = pd.read_csv(os.path.join(RESULTS_DIR, "efa_scores.csv"))
print(f"   {len(efa_df)} kayıt")

# Occlusion varsa F_vis güncellenmiş mi kontrol et
f_vis_mean = efa_df['f_vis'].mean()
if f_vis_mean == 0:
    print("   UYARI: F_vis hala 0 — step4_efa.py'yi occlusion sonrası tekrar çalıştır!")
else:
    print(f"   F_vis ortalama: {f_vis_mean:.3f} (occlusion dahil)")

# ------------------------------------------------------------------
# 2. Confidence-Faithfulness Korelasyon
# ------------------------------------------------------------------
print("\n[2/4] Confidence-Faithfulness korelasyon hesaplanıyor...")

corr_results = []
for model in efa_df['model'].unique():
    sub = efa_df[efa_df['model'] == model].dropna(subset=['confidence', 'efa'])

    if len(sub) < 10:
        continue

    pearson_r, pearson_p = stats.pearsonr(sub['confidence'], sub['efa'])
    spearman_r, spearman_p = stats.spearmanr(sub['confidence'], sub['efa'])

    # Bootstrap CI (95%)
    n_boot = 1000
    boot_rs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(len(sub), size=len(sub), replace=True)
        boot_sub = sub.iloc[idx]
        try:
            r, _ = stats.pearsonr(boot_sub['confidence'], boot_sub['efa'])
            boot_rs.append(r)
        except:
            pass
    ci_low  = np.percentile(boot_rs, 2.5)
    ci_high = np.percentile(boot_rs, 97.5)

    corr_results.append({
        'model':       model,
        'n':           len(sub),
        'pearson_r':   round(pearson_r, 3),
        'pearson_p':   round(pearson_p, 4),
        'spearman_r':  round(spearman_r, 3),
        'spearman_p':  round(spearman_p, 4),
        'ci_95_low':   round(ci_low, 3),
        'ci_95_high':  round(ci_high, 3),
    })

    print(f"\n   {model} (n={len(sub)}):")
    print(f"   Pearson  r = {pearson_r:.3f}  (p={pearson_p:.4f})")
    print(f"   Spearman ρ = {spearman_r:.3f}  (p={spearman_p:.4f})")
    print(f"   95% CI     = [{ci_low:.3f}, {ci_high:.3f}]")

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(os.path.join(RESULTS_DIR, "correlation_results.csv"), index=False)

# ------------------------------------------------------------------
# 3. Danger Zone Prevalence
# ------------------------------------------------------------------
print("\n[3/4] Danger Zone analizi...")

dz_results = []
for model in efa_df['model'].unique():
    sub = efa_df[efa_df['model'] == model]
    overall_dz = sub['danger_zone'].mean() * 100

    print(f"\n   {model}: Overall DZ = {overall_dz:.1f}%")
    for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        sc_sub = sub[sub['superclass'] == sc]
        dz_rate = sc_sub['danger_zone'].mean() * 100
        print(f"   {sc:6s}: {dz_rate:.1f}%  (n={len(sc_sub)})")
        dz_results.append({
            'model': model,
            'superclass': sc,
            'dz_rate': round(dz_rate, 1),
            'n': len(sc_sub),
            'mean_efa': round(sc_sub['efa'].mean(), 3),
            'mean_conf': round(sc_sub['confidence'].mean(), 3),
        })

dz_df = pd.DataFrame(dz_results)
dz_df.to_csv(os.path.join(RESULTS_DIR, "danger_zone_results.csv"), index=False)

# ------------------------------------------------------------------
# 4. Özet Rapor
# ------------------------------------------------------------------
print("\n[4/4] Özet Rapor")
print("="*70)
print(f"{'Model':<25} {'EFA':>6} {'F_vis':>6} {'F_txt':>6} {'Conf':>6} {'DZ%':>6}")
print("-"*70)

for model in sorted(efa_df['model'].unique()):
    sub = efa_df[efa_df['model'] == model]
    print(f"{model:<25} "
          f"{sub['efa'].mean():>6.3f} "
          f"{sub['f_vis'].mean():>6.3f} "
          f"{sub['f_txt'].mean():>6.3f} "
          f"{sub['confidence'].mean():>6.3f} "
          f"{sub['danger_zone'].mean()*100:>6.1f}%")

print("\nSuperclass x Model EFA:")
print(f"{'':10}", end="")
for model in sorted(efa_df['model'].unique()):
    short = model.split('-')[0][:8]
    print(f"{short:>10}", end="")
print()

for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
    print(f"{sc:<10}", end="")
    for model in sorted(efa_df['model'].unique()):
        val = efa_df[(efa_df['model']==model) & (efa_df['superclass']==sc)]['efa'].mean()
        print(f"{val:>10.3f}", end="")
    print()

print("\n✓ Çıktılar:")
print(f"  {RESULTS_DIR}/efa_scores.csv")
print(f"  {RESULTS_DIR}/correlation_results.csv")
print(f"  {RESULTS_DIR}/danger_zone_results.csv")
