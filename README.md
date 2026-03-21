# EFA-ECG: Explanation-Attribution Faithfulness Auditor for ECG-Focused Multimodal LLMs

<p align="center">
  <img src="data/images/teaser.png" width="800" alt="Right Diagnosis, Blind Model"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/JBHI.XXXX.XXXXXXX"><img src="https://img.shields.io/badge/IEEE%20JBHI-Under%20Review-orange"/></a>
  <a href="https://www.physionet.org/content/ptb-xl/1.0.3/"><img src="https://img.shields.io/badge/Dataset-PTB--XL-blue"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"/></a>
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/Status-Experiments%20Complete-green"/>
</p>

---

## 📄 Paper

**"Do Multimodal LLMs Really See What They Say? A Faithfulness Audit for ECG Interpretation"**

*Musa Ataş — Department of Computer Engineering, Siirt University, Turkey*

> Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI), 2025.

### Abstract

Multimodal large language models (MLLMs) generate natural language explanations alongside ECG diagnostic outputs — but do these explanations faithfully reflect the visual regions the model actually attended to? We introduce the **Explanation-Attribution Faithfulness Auditor (EFA)**, a training-free, fully reproducible framework that quantifies semantic alignment between model-generated explanations and visual attribution maps, grounded in lead-level annotations derived deterministically from PTB-XL SCP codes. EFA employs a unified lead-structured occlusion attribution pipeline applicable to both open-weights models and closed-source APIs. Applied to Gemini 2.5 Flash and LLaVA-v1.6-Mistral across 1,000 PTB-XL recordings, EFA reveals a consistent faithfulness gap. Between 14.6% and 20.8% of cases fall into the **Danger Zone** — high linguistic confidence paired with low faithfulness — representing the most clinically hazardous failure mode.

---

## 🔑 Key Concepts

| Concept | Description |
|---|---|
| **EFA Score** | Composite metric (α·F_vis + (1−α)·F_txt) measuring alignment between visual attribution and textual lead references |
| **F_vis** | Visual faithfulness: IoU between lead-structured occlusion map and ground-truth lead mask |
| **F_txt** | Textual faithfulness: precision/recall of lead references extracted via rule-based NER |
| **Danger Zone** | High linguistic confidence (C > τ_c) + Low EFA score (F < τ_f) — the most clinically hazardous failure mode |
| **Ground Truth** | Lead-level annotations derived deterministically from PTB-XL SCP codes — no manual annotation required |

---

## 📁 Repository Structure

```
efa-ecg/
├── README.md
├── requirements.txt
│
├── step1a_sampling.py          # Stratified sampling from PTB-XL (1,000 ECGs)
├── step1b_groundtruth.py       # SCP code → lead-level ground truth
├── step1c_rendering.py         # ECG rendering (300 DPI, 4×3 layout)
├── step2a_gemini_inference.py  # Gemini 2.5 Flash inference (parallel)
├── step3a_occlusion.py         # Lead-structured occlusion attribution
├── step4_efa.py                # EFA score computation
├── step5_analysis.py           # Correlation & Danger Zone analysis
│
├── data/
│   ├── sample_1000.csv         # 1,000 stratified ECG IDs
│   ├── ground_truth.csv        # Lead-level ground truth
│   ├── ground_truth.json
│   ├── panel_coords.json       # Lead panel pixel coordinates
│   ├── images/                 # Rendered ECG PNGs (300 DPI)
│   └── ptb-xl/                 # PTB-XL raw data (download separately)
│
├── kaggle_upload/              # Files uploaded to Kaggle for GPU inference
│   ├── ecg_images.zip
│   ├── ground_truth.csv
│   ├── ground_truth.json
│   └── sample_1000.csv
│
└── results/
    ├── gemini_outputs.csv          # Gemini 2.5 Flash responses (1,000)
    ├── llava_mistral_outputs.csv   # LLaVA-v1.6-Mistral responses (1,000)
    ├── llava_mistral_occlusion.csv # Occlusion attribution scores
    ├── efa_scores.csv              # Final EFA scores (all models)
    ├── correlation_results.csv     # Confidence–faithfulness correlation
    ├── danger_zone_results.csv     # Danger Zone prevalence
    └── occ_maps/                   # Per-ECG occlusion numpy arrays
```

---

## ⚙️ Installation

```bash
git clone https://github.com/hakmesyo/efa-ecg.git
cd efa-ecg
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📊 Dataset

Download **PTB-XL** from PhysioNet and place under `data/ptb-xl/`:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

---

## 🚀 Reproducing the Experiments

```bash
# Step 1: Data preparation
python step1a_sampling.py
python step1b_groundtruth.py
python step1c_rendering.py

# Step 2: Model inference
python step2a_gemini_inference.py   # Requires GEMINI_API_KEY

# Step 3: Occlusion attribution (GPU recommended — run on Kaggle/Colab)
python step3a_occlusion.py

# Step 4: EFA computation
python step4_efa.py

# Step 5: Analysis
python step5_analysis.py
```

> **Note:** Step 3 requires a GPU with at least 16 GB VRAM (FP16). It can be run on any GPU-enabled environment — local, cloud, or free-tier platforms such as Kaggle or Google Colab.

---

## 📈 Main Results

### EFA Scores by Model and Diagnostic Superclass

| Model | NORM | MI | STTC | CD | HYP | Macro |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | 0.007 | 0.320 | 0.344 | 0.269 | 0.331 | **0.254** |
| LLaVA-v1.6-Mistral | 0.003 | 0.181 | 0.192 | 0.106 | 0.202 | 0.137 |

### Confidence–Faithfulness Correlation

| Model | Pearson r | p-value | Interpretation |
|---|---|---|---|
| Gemini 2.5 Flash | 0.045 | 0.156 | No significant correlation |
| LLaVA-v1.6-Mistral | −0.062 | 0.049 | Weak negative correlation |

### Danger Zone Prevalence

| Model | NORM | MI | STTC | CD | HYP | Overall |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | 75.5% | 0.0% | 5.5% | 19.5% | 3.5% | **20.8%** |
| LLaVA-v1.6-Mistral | 53.0% | 6.0% | 0.5% | 13.5% | 0.0% | 14.6% |

> High NORM Danger Zone rates reflect that models express high confidence on normal ECGs while generating lead-specific explanations that cannot be grounded against any pathology-specific lead set.

---

## 📝 Citation

```bibtex
@article{atas2025efa,
  title     = {Do Multimodal {LLMs} Really See What They Say?
               A Faithfulness Audit for {ECG} Interpretation},
  author    = {Ata\c{s}, Musa},
  journal   = {{IEEE} Journal of Biomedical and Health Informatics},
  year      = {2025},
  note      = {Under review}
}
```

---

## 🗺️ Roadmap

- [x] ECG rendering pipeline (1D → 2D, 300 DPI)
- [x] Stratified sampling (1,000 recordings, 5 superclasses)
- [x] SCP → lead-level ground truth construction
- [x] Gemini 2.5 Flash inference (1,000 ECGs)
- [x] LLaVA-v1.6-Mistral inference (1,000 ECGs)
- [x] Unified lead-structured occlusion attribution
- [x] SpaCy NER lead extractor (P=0.96, R=0.92)
- [x] EFA metric (additive + multiplicative + harmonic variants)
- [x] Danger Zone classifier
- [x] Confidence–faithfulness correlation analysis

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- PTB-XL dataset: Wagner et al., PhysioNet 2020
- Gemini API: Google DeepMind
- LLaVA: Haotian Liu et al.

---

*Cezeri Artificial Intelligence Laboratory — Siirt University, Turkey*