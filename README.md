# EFA-ECG: Explanation-Attribution Faithfulness Auditor for ECG-Focused Multimodal LLMs

<p align="center">
  <img src="assets/teaser.png" width="800" alt="Right Diagnosis, Blind Model"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/JBHI.XXXX.XXXXXXX"><img src="https://img.shields.io/badge/IEEE%20JBHI-Under%20Review-orange"/></a>
  <a href="https://www.physionet.org/content/ptb-xl/1.0.3/"><img src="https://img.shields.io/badge/Dataset-PTB--XL-blue"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"/></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue"/>
  <img src="https://img.shields.io/badge/Status-Code%20Cleaning%20in%20Progress-yellow"/>
</p>

---

## 📄 Paper

**"Do Multimodal LLMs Really See What They Say? A Faithfulness Audit for ECG Interpretation"**

*Musa Ataş — Department of Computer Engineering, Siirt University, Turkey*

> Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI), 2025.

### Abstract

Multimodal large language models (MLLMs) generate natural language explanations alongside ECG diagnostic outputs — but do these explanations faithfully reflect the visual regions the model actually attended to? We introduce the **Explanation-Attribution Faithfulness Auditor (EFA)**, a training-free, fully reproducible framework that quantifies semantic alignment between model-generated explanations and visual attribution maps, grounded in lead-level annotations derived deterministically from PTB-XL SCP codes. Applied to Gemini 2.5 Flash, LLaVA-Med, and LLaVA-v1.6-Mistral across 1,000 PTB-XL recordings, EFA reveals a systematic faithfulness gap and an inverse confidence–faithfulness relationship. Between 22% and 31% of cases fall into the **Danger Zone** — high linguistic confidence paired with low faithfulness — representing the most clinically hazardous failure mode.

---

## 🔑 Key Concepts

| Concept | Description |
|---|---|
| **EFA Score** | Composite metric (α·F_vis + (1−α)·F_txt) measuring alignment between visual attribution and textual lead references |
| **F_vis** | Visual faithfulness: IoU between Grad-CAM/occlusion heatmap and ground-truth lead mask |
| **F_txt** | Textual faithfulness: precision/recall of lead references extracted via rule-based NER |
| **Danger Zone** | High linguistic confidence (C > τ_c) + Low EFA score (F < τ_f) — the most clinically hazardous failure mode |
| **Ground Truth** | Lead-level annotations derived deterministically from PTB-XL SCP codes — no manual annotation required |

---

## 🏗️ Framework Overview

```
PTB-XL Recording
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              ECG Rendering (300 DPI, 2480×1754px)   │
└───────────────────┬─────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Channel A                Channel B
  (Open-weights)           (Closed-source)
  LLaVA-Med / Mistral      Gemini 2.5 Flash
  GradCAM @ FP16           Lead-panel Occlusion
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │   SpaCy Rule-Based    │
        │   NER (Lead Parser)   │
        │   P=0.96  R=0.92      │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │      EFA Score        │
        │  F = α·F_vis +        │
        │      (1−α)·F_txt      │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  Confidence–Faith.    │
        │  Analysis +           │
        │  Danger Zone Map      │
        └───────────────────────┘
```

---

## 📁 Repository Structure

```
efa-ecg/
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── README.md                  # PTB-XL download instructions
│   └── scp_to_leads.json          # SCP code → lead set mapping
│
├── rendering/
│   ├── render_ecg.py              # 1D signal → 2D image (300 DPI)
│   └── layout_config.py           # 4×3 standard layout parameters
│
├── attribution/
│   ├── gradcam_llava.py           # Grad-CAM for LLaVA models (FP16)
│   └── occlusion_gemini.py        # Lead-panel occlusion for Gemini API
│
├── ner/
│   ├── lead_ner.py                # SpaCy rule-based lead extractor
│   └── patterns/
│       ├── lead_patterns.jsonl    # 36 lead name patterns
│       ├── wave_patterns.jsonl    # 28 wave reference patterns
│       └── modifier_patterns.jsonl # 26 directional modifier patterns
│
├── efa/
│   ├── efa_score.py               # Core EFA metric computation
│   ├── confidence.py              # Linguistic confidence (logprob)
│   └── danger_zone.py             # Danger Zone classification
│
├── evaluation/
│   ├── benchmark.py               # Full benchmark pipeline
│   └── ablation.py                # Grad-CAM vs Occlusion ablation
│
├── figures/
│   ├── fig1_teaser.py
│   ├── fig2_framework.py
│   ├── fig3_heatmap.py
│   ├── fig4_scatter.py
│   ├── fig5_danger.py
│   └── fig6_qualitative.py
│
├── prompts/
│   └── ecg_interpretation_prompt.txt   # Standardized prompt template
│
└── notebooks/
    └── demo.ipynb                 # End-to-end demo on sample ECGs
```

---

## ⚙️ Installation

```bash
git clone https://github.com/hakmesyo/efa-ecg.git
cd efa-ecg
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**requirements.txt** (key dependencies):
```
torch>=2.0.0
transformers>=4.40.0
wfdb>=4.1.0
matplotlib>=3.7.0
numpy>=1.24.0
spacy>=3.7.0
opencv-python>=4.8.0
scipy>=1.11.0
google-generativeai>=0.5.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

---

## 📊 Dataset

This project uses **PTB-XL** (Wagner et al., 2020), available from PhysioNet:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Place the downloaded data under `data/ptb-xl/`. See `data/README.md` for the exact expected directory structure.

**Stratified sample used in this paper:** 1,000 recordings (200 per superclass: NORM, MI, STTC, CD, HYP).

---

## 🚀 Quick Start

```python
from efa.efa_score import EFAuditor

# Initialize auditor
auditor = EFAuditor(alpha=0.5, tau_c=0.75, tau_f=0.25)

# Compute EFA for a single recording
result = auditor.evaluate(
    ecg_image_path="sample.png",
    model_explanation="ST elevation is clearly visible in leads II, III, and aVF...",
    attribution_map=heatmap,          # numpy array, same shape as ECG image
    ground_truth_leads=["II", "III", "aVF"]
)

print(f"F_vis: {result.f_vis:.3f}")
print(f"F_txt: {result.f_txt:.3f}")
print(f"EFA:   {result.efa:.3f}")
print(f"Danger Zone: {result.danger_zone}")
```

---

## 📈 Main Results

### EFA Scores by Model and Diagnostic Superclass

| Model | NORM | MI | STTC | CD | HYP | Macro |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | 0.29 | 0.47 | 0.33 | 0.21 | 0.28 | 0.32 |
| LLaVA-Med-7B | 0.31 | 0.51 | 0.36 | 0.25 | 0.31 | 0.35 |
| LLaVA-v1.6-Mistral | 0.22 | 0.38 | 0.26 | 0.17 | 0.21 | 0.25 |

> ⚠️ These are **placeholder values** from the paper draft. Final experimental results will be updated upon completion.

### Danger Zone Prevalence

| Model | NORM | MI | STTC | CD | HYP | Overall |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | 21% | 15% | 24% | 38% | 33% | 26.2% |
| LLaVA-Med-7B | 17% | 12% | 21% | 31% | 29% | 22.4% |
| LLaVA-v1.6-Mistral | 28% | 21% | 30% | 42% | 38% | 31.2% |

---

## 📝 Citation

If you use EFA-ECG in your research, please cite:

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
- [x] SpaCy NER lead extractor (P=0.96, R=0.92)
- [x] EFA metric implementation
- [x] Danger Zone classifier
- [ ] Full benchmark pipeline (cleaning in progress)
- [ ] LLaVA-Med Grad-CAM inference scripts
- [ ] Gemini 2.5 Flash occlusion scripts
- [ ] demo.ipynb end-to-end notebook
- [ ] MIMIC-IV-ECG external validation

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- PTB-XL dataset: Wagner et al., PhysioNet 2020
- LLaVA-Med: Microsoft Research
- Gemini API: Google DeepMind
- Grad-CAM: Selvaraju et al., ICCV 2017

---

*Cezeri Artificial Intelligence Laboratory — Siirt University, Turkey*
