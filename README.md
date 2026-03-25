# EFA-ECG: Explanation-Attribution Faithfulness Auditor for ECG-Focused Multimodal LLMs

<p align="center">
  <img src="https://github.com/hakmesyo/efa-ecg/releases/download/ecg_images/teaser.png" width="800" alt="Right Diagnosis, Blind Model"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/JBHI.XXXX.XXXXXXX"><img src="https://img.shields.io/badge/IEEE%20JBHI-Under%20Review-orange"/></a>
  <a href="https://www.kaggle.com/code/cezeriotonomo/efa-ecg"><img src="https://img.shields.io/badge/Kaggle-Reproduce%20in%201--Click-20BEFF?logo=kaggle"/></a>
  <a href="https://www.physionet.org/content/ptb-xl/1.0.3/"><img src="https://img.shields.io/badge/Dataset-PTB--XL-blue"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"/></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue"/>
  <img src="https://img.shields.io/badge/Models-5-brightgreen"/>
</p>

---

## Paper

**"Do Multimodal LLMs Really See What They Say? A Faithfulness Audit for ECG Interpretation"**

*Musa Ataş — Department of Computer Engineering, Siirt University, Turkey*

> Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI), 2025.

### Abstract

Multimodal large language models (MLLMs) generate natural language explanations alongside ECG diagnostic outputs — but do these explanations faithfully reflect the visual regions the model actually attended to? We introduce the **Explanation-Attribution Faithfulness Auditor (EFA)**, a training-free, fully reproducible framework that quantifies semantic alignment between model-generated explanations and visual attribution maps, grounded in lead-level annotations derived deterministically from PTB-XL SCP codes. Applied to five state-of-the-art MLLMs — Gemini 2.5 Flash, Claude Sonnet 4, LLaVA-1.5-7B, Qwen2.5-VL-7B, and Gemma3-4B — across 250 stratified PTB-XL recordings, EFA reveals a consistent **faithfulness gap**: all models score below a random attribution baseline on pathological classes. A substantial proportion of pathological cases fall into the **Danger Zone** (high linguistic confidence paired with low faithfulness; 8.0%–18.5% across models) — the most clinically hazardous failure mode, as it provides no signal to the clinician that the explanation is unfaithful.

---

## Key Concepts

| Concept | Description |
|---|---|
| **EFA Score** | Composite metric: `α·F_vis + (1-α)·F_txt`, default α=0.5 |
| **F_vis** | Visual faithfulness: IoU between occlusion-sensitive leads and ground-truth lead set |
| **F_txt** | Textual faithfulness: Jaccard IoU between NER-extracted lead references and ground-truth leads |
| **Danger Zone** | High confidence (τ_c ≥ 0.40) + Low EFA (τ_f < 0.10) — clinically hazardous failure mode |
| **Ground Truth** | Lead-level annotations derived deterministically from PTB-XL SCP codes |
| **Auditability Barrier** | Closed-source models that disable logprobs cannot be subjected to occlusion-based auditing |

---

## One-Click Reproduction

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/cezeriotonomo/efa-ecg)

> **Use the latest notebook: [`efa-ecg-v3.ipynb`](efa-ecg-v3.ipynb)**

1. Open the notebook on Kaggle (or upload `efa-ecg-v3.ipynb` manually)
2. Add your API keys in **Add-ons → Secrets**:
   - `GEMINI_API_KEY` (from [aistudio.google.com](https://aistudio.google.com))
   - `CLAUDE_API_KEY` (from [console.anthropic.com](https://console.anthropic.com/settings/api-keys))
3. Select **GPU T4 x2**, ensure **Internet** is on
4. **Run All** — pre-computed results auto-download from GitHub Release; API inference is skipped by default

> API inference costs ~$3.50 (Claude) and ~$1.00 (Gemini) for 250 recordings. All open-weights models (LLaVA, Qwen, Gemma3) run free on Kaggle GPU. Pre-computed results for all 5 models are available in the GitHub Release.

---

## Evaluated Models

| Model | Type | Provider | Access | Version |
|---|---|---|---|---|
| **Gemini 2.5 Flash** | Closed-source | Google | API | `gemini-2.5-flash-preview-04-17` |
| **Claude Sonnet 4** | Closed-source | Anthropic | API | `claude-sonnet-4-20250514` |
| **LLaVA-1.5-7B** | Open-weights | UW–Madison | Kaggle GPU (INT4) | LLaVA-1.5 |
| **Qwen2.5-VL-7B** | Open-weights | Alibaba | Kaggle GPU (INT4) | Qwen2.5-VL |
| **Gemma3-4B** | Open-weights | Google | Kaggle GPU (INT4) | Gemma 3 |

---

## Main Results

| Model | EFA (macro) | F_vis | F_txt | DZ% (path.) |
|---|---|---|---|---|
| Gemini 2.5 Flash | **0.234** | 0.109 | 0.359 | 8.0% |
| Claude Sonnet 4 | 0.222 | 0.085 | 0.359 | 9.5% |
| LLaVA-1.5-7B | 0.213 | N/A† | 0.317 | 11.5% |
| Qwen2.5-VL-7B | 0.208 | 0.087 | 0.330 | 11.0% |
| Gemma3-4B | 0.159 | 0.089 | 0.229 | 18.5% |
| **Random baseline** | **0.315** | — | — | — |

> †LLaVA F_vis is N/A due to zero confidence variance (all responses "definite"). DZ%: Danger Zone prevalence on pathological classes only (τ_c ≥ 0.40, τ_f < 0.10; NORM excluded as methodological artefact).

---

## Changelog

### v3 (current)
- NORM class excluded from Danger Zone analysis (methodological artefact — no GT lead set defined)
- Absolute thresholds (τ_c = 0.40, τ_f = 0.10) replace percentile-based thresholds
- LLaVA-1.5-7B F_vis reported as N/A with explicit caveat
- Over-citation ablation analysis added (Table V in paper)
- Lead citation frequency analysis added (Table VI + Appendix E)
- API versions pinned for reproducibility

### v2
- 5 models (was 2 in v1)
- F_txt uses Jaccard IoU (was Recall)
- White (255) occlusion fill (was gray 128) + ablation study
- Per-model independent occlusion attribution
- NORM edge case handled explicitly
- α sensitivity analysis {0.3, 0.5, 0.7}
- MI subclass breakdown (inferior / anterior / lateral)
- Additive + Multiplicative + Harmonic EFA variants

---

## GitHub Release

[All data and pre-computed results](https://github.com/hakmesyo/efa-ecg/releases/tag/ecg_images) are auto-downloaded by the notebook:

`ecg_images.zip` · `sample_1000.csv` · `ground_truth.csv` · `panel_coords.json` · `gemini_outputs.csv` · `claude_outputs.csv` · `llava_outputs.csv` · `qwen_outputs.csv` · `gemma3_outputs.csv` · `gemini_occlusion.csv` · `claude_occlusion.csv` · `llava_occlusion.csv` · `qwen_occlusion.csv` · `gemma3_occlusion.csv` · `ablation_occlusion.csv`

---

## Repository Structure

```
efa-ecg/
├── README.md
├── requirements.txt
├── efa-ecg-v3.ipynb               ← Main notebook (recommended)
├── step1a_sampling.py             ← Reference only
├── step1b_groundtruth.py          ← Reference only
├── step1c_rendering.py            ← Reference only
├── step2a_gemini_inference.py     ← Reference only
├── step3a_occlusion.py            ← Reference only
├── step4_efa.py                   ← Reference only
└── step5_analysis.py              ← Reference only
```

---

## Citation

```bibtex
@article{atas2025efa,
  title   = {Do Multimodal {LLMs} Really See What They Say?
             A Faithfulness Audit for {ECG} Interpretation},
  author  = {Ata\c{s}, Musa},
  journal = {{IEEE} Journal of Biomedical and Health Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## Author

**Prof. Dr. Musa Ataş** · Siirt University, Turkey · [musa.atas@siirt.edu.tr](mailto:musa.atas@siirt.edu.tr) · [github.com/hakmesyo](https://github.com/hakmesyo)

*Cezeri Artificial Intelligence Laboratory*
