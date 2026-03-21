# EFA-ECG: Explanation-Attribution Faithfulness Auditor for ECG-Focused Multimodal LLMs

<p align="center">
  <img src="data/images/teaser.png" width="800" alt="Right Diagnosis, Blind Model"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/JBHI.XXXX.XXXXXXX"><img src="https://img.shields.io/badge/IEEE%20JBHI-Under%20Review-orange"/></a>
  <a href="https://www.kaggle.com/code/cezeriotonomo/efa-ecg"><img src="https://img.shields.io/badge/Kaggle-Reproduce%20in%201--Click-20BEFF?logo=kaggle"/></a>
  <a href="https://www.physionet.org/content/ptb-xl/1.0.3/"><img src="https://img.shields.io/badge/Dataset-PTB--XL-blue"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"/></a>
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/Models-5-brightgreen"/>
</p>

---

## Paper

**"Do Multimodal LLMs Really See What They Say? A Faithfulness Audit for ECG Interpretation"**

*Musa Ataş — Department of Computer Engineering, Siirt University, Turkey*

> Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI), 2025.

### Abstract

Multimodal large language models (MLLMs) generate natural language explanations alongside ECG diagnostic outputs — but do these explanations faithfully reflect the visual regions the model actually attended to? We introduce the **Explanation-Attribution Faithfulness Auditor (EFA)**, a training-free, fully reproducible framework that quantifies semantic alignment between model-generated explanations and visual attribution maps, grounded in lead-level annotations derived deterministically from PTB-XL SCP codes. EFA employs a unified lead-structured occlusion attribution pipeline applicable to both open-weights models and closed-source APIs. Applied to five state-of-the-art MLLMs — Gemini 2.5 Flash, Claude Sonnet 4, LLaVA-v1.6-Mistral-7B, Qwen2.5-VL-7B, and InternVL2-8B — across 250 stratified PTB-XL recordings, EFA reveals a consistent **faithfulness gap**: models frequently produce plausible-sounding explanations that are poorly grounded in their actual visual evidence. A substantial proportion of cases fall into the **Danger Zone** — high linguistic confidence paired with low faithfulness — representing the most clinically hazardous failure mode.

---

## Key Concepts

| Concept | Description |
|---|---|
| **EFA Score** | Composite metric (α·F_vis + (1−α)·F_txt) measuring alignment between visual attribution and textual lead references |
| **F_vis** | Visual faithfulness: IoU between top-k occluded leads and ground-truth lead set |
| **F_txt** | Textual faithfulness: Jaccard IoU between NER-extracted lead references and ground-truth leads |
| **Danger Zone** | High linguistic confidence + Low EFA score — the most clinically hazardous failure mode |
| **Ground Truth** | Lead-level annotations derived deterministically from PTB-XL SCP codes — no manual annotation required |

---

## One-Click Reproduction (Recommended)

The entire experiment pipeline — data download, model inference, occlusion attribution, EFA computation, and statistical analysis — runs in a **single Kaggle notebook**.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/cezeriotonomo/efa-ecg)

### Steps

1. Click the badge above (or open the notebook on Kaggle)
2. Go to **Add-ons → Secrets** and add your API keys:
   - `CLAUDE_API_KEY` — from [console.anthropic.com](https://console.anthropic.com/settings/api-keys)
   - `GEMINI_API_KEY` — from [aistudio.google.com](https://aistudio.google.com/apikey) *(optional: pre-computed results are included)*
3. Select **GPU T4** or **GPU P100** as accelerator
4. Ensure **Internet** is enabled
5. **Run All** — results will match the paper

> **Note:** Pre-computed Gemini 2.5 Flash results are automatically downloaded from this repository's GitHub Release. Re-running Gemini inference is optional and requires a valid API key. Claude Sonnet 4 inference requires a valid API key with credits (~$3-4 for 250 recordings). Local models (LLaVA, Qwen, InternVL) run on Kaggle GPU at no cost.

---

## Evaluated Models

| Model | Type | Access | Quantization |
|---|---|---|---|
| **Gemini 2.5 Flash** | Closed-source | Google API | — |
| **Claude Sonnet 4** | Closed-source | Anthropic API | — |
| **LLaVA-v1.6-Mistral-7B** | Open-weights | Local (Kaggle GPU) | INT4 (bitsandbytes) |
| **Qwen2.5-VL-7B** | Open-weights | Local (Kaggle GPU) | INT4 (bitsandbytes) |
| **InternVL2-8B** | Open-weights | Local (Kaggle GPU) | INT4 (bitsandbytes) |

Open-weights models are evaluated at INT4 precision, reflecting realistic clinical deployment conditions where high-end GPU infrastructure is not universally available.

---

## Methodology Improvements (v2)

This version addresses reviewer feedback with the following changes:

| Aspect | v1 | v2 |
|---|---|---|
| Models evaluated | 2 | **5** |
| F_txt metric | Recall only | **Jaccard IoU** (primary), F1 and Recall (secondary) |
| Occlusion fill | Gray (128) | **White (255)** + gray vs white ablation study |
| Per-model occlusion | Shared across models | **Independent per model** |
| NORM edge case | Undefined (division by zero) | **Explicitly handled** |
| α sensitivity | Claimed but not shown | **Reported for α ∈ {0.3, 0.5, 0.7}** |
| MI subclass analysis | Not reported | **Inferior / Anterior / Lateral breakdown** |
| EFA variants | Additive only | **Additive + Multiplicative + Harmonic** |

---

## Dataset

All experiments use **PTB-XL**, the largest publicly available clinical 12-lead ECG dataset (21,799 recordings). A stratified subset of 250 recordings (50 per diagnostic superclass) is used for the main evaluation. Pre-rendered ECG images (300 DPI, 4×3 layout) are available in the GitHub Release.

### GitHub Release Contents

All data files are hosted as [GitHub Release assets](https://github.com/hakmesyo/efa-ecg/releases/tag/ecg_images):

| File | Description |
|---|---|
| `ecg_images.zip` | 1,000 pre-rendered 12-lead ECG images (PNG, 300 DPI) |
| `sample_1000.csv` | Stratified sample metadata with SCP codes |
| `ground_truth.csv` | Lead-level ground truth derived from SCP codes |
| `panel_coords.json` | Pixel coordinates for each lead panel |
| `gemini_outputs.csv` | Pre-computed Gemini 2.5 Flash inference results |

The Kaggle notebook automatically downloads these files at runtime.

---

## Main Results

> **Note:** Results below are from v2 experiments (5 models, 250 recordings). Tables will be updated upon completion of all experiments.

### EFA Scores by Model and Diagnostic Superclass

| Model | NORM | MI | STTC | CD | HYP | Macro |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | — | — | — | — | — | — |
| Claude Sonnet 4 | — | — | — | — | — | — |
| LLaVA-v1.6-Mistral-7B | — | — | — | — | — | — |
| Qwen2.5-VL-7B | — | — | — | — | — | — |
| InternVL2-8B | — | — | — | — | — | — |

*Results will be populated after completing all model evaluations.*

---

## Repository Structure

```
efa-ecg/
├── README.md                      ← This file
├── requirements.txt               ← Dependencies for local execution
├── notebooks/
│   └── efa_ecg_v2_kaggle.ipynb    ← Main experiment notebook (recommended)
├── step1a_sampling.py             ← Reference: stratified sampling
├── step1b_groundtruth.py          ← Reference: SCP → lead mapping
├── step1c_rendering.py            ← Reference: 1D signal → 2D image
├── step2a_gemini_inference.py     ← Reference: Gemini API inference
├── step3a_occlusion.py            ← Reference: occlusion attribution
├── step4_efa.py                   ← Reference: EFA computation
└── step5_analysis.py              ← Reference: statistical analysis
```

> The individual Python scripts are provided for reference only. The recommended way to reproduce all results is the **Kaggle notebook**.

---

## Citation

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

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- PTB-XL dataset: Wagner et al., PhysioNet 2020
- Gemini API: Google DeepMind
- Claude API: Anthropic
- LLaVA: Haotian Liu et al.
- Qwen2.5-VL: Alibaba DAMO Academy
- InternVL2: OpenGVLab, Shanghai AI Laboratory

---

## Author

**Prof. Dr. Musa Ataş**
Department of Computer Engineering, Siirt University, Turkey

[musa.atas@siirt.edu.tr](mailto:musa.atas@siirt.edu.tr) · [hakmesyo@gmail.com](mailto:hakmesyo@gmail.com) · [github.com/hakmesyo](https://github.com/hakmesyo)

---

*Cezeri Artificial Intelligence Laboratory — Siirt University, Turkey*