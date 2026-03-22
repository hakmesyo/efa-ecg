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
  <img src="https://img.shields.io/badge/Models-6-brightgreen"/>
</p>

---

## Paper

**"Do Multimodal LLMs Really See What They Say? A Faithfulness Audit for ECG Interpretation"**

*Musa Ataş — Department of Computer Engineering, Siirt University, Turkey*

> Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI), 2025.

### Abstract

Multimodal large language models (MLLMs) generate natural language explanations alongside ECG diagnostic outputs — but do these explanations faithfully reflect the visual regions the model actually attended to? We introduce the **Explanation-Attribution Faithfulness Auditor (EFA)**, a training-free, fully reproducible framework that quantifies semantic alignment between model-generated explanations and visual attribution maps, grounded in lead-level annotations derived deterministically from PTB-XL SCP codes. Applied to six state-of-the-art MLLMs — including Gemini 2.5 Flash, Claude Sonnet 4, Gemma3-4B, Qwen2.5-VL-7B, InternVL2-8B, and LLaVA-1.5-7B — across 250 stratified PTB-XL recordings, EFA reveals a consistent **faithfulness gap**. A substantial proportion of cases fall into the **Danger Zone** — high linguistic confidence paired with low faithfulness — representing the most clinically hazardous failure mode.

---

## Key Concepts

| Concept | Description |
|---|---|
| **EFA Score** | Composite metric measuring alignment between visual attribution and textual lead references |
| **F_vis** | Visual faithfulness: IoU between top-k occluded leads and ground-truth lead set |
| **F_txt** | Textual faithfulness: Jaccard IoU between NER-extracted lead references and ground-truth leads |
| **Danger Zone** | High confidence + Low faithfulness — no signal to the clinician that the explanation is unfaithful |
| **Ground Truth** | Lead-level annotations derived deterministically from PTB-XL SCP codes |

---

## One-Click Reproduction

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/cezeriotonomo/efa-ecg)

1. Open the notebook on Kaggle
2. Add `CLAUDE_API_KEY` in **Add-ons → Secrets** (from [console.anthropic.com](https://console.anthropic.com/settings/api-keys))
3. Select **GPU T4 x2**, ensure **Internet** is on
4. **Run All** — pre-computed results auto-download from GitHub Release

> Claude inference costs ~$3.50 for 250 recordings. All other models run free on Kaggle GPU or use pre-computed results.

---

## Evaluated Models

| Model | Type | Provider | Access |
|---|---|---|---|
| **Gemini 2.5 Flash** | Closed-source | Google | API |
| **Claude Sonnet 4** | Closed-source | Anthropic | API |
| **Gemma3-4B** | Open-weights | Google | Kaggle GPU (INT4) |
| **Qwen2.5-VL-7B** | Open-weights | Alibaba | Kaggle GPU (INT4) |
| **InternVL2-8B** | Open-weights | Shanghai AI Lab | Kaggle GPU (INT4) |
| **LLaVA-1.5-7B** | Open-weights | UW–Madison | Kaggle GPU (INT4) |

---

## Methodology v2 Improvements

| Aspect | v1 | v2 |
|---|---|---|
| Models | 2 | **6** |
| F_txt | Recall | **Jaccard IoU** |
| Occlusion fill | Gray (128) | **White (255)** + ablation |
| Per-model occlusion | Shared | **Independent** |
| NORM edge case | Undefined | **Handled** |
| α sensitivity | Not shown | **{0.3, 0.5, 0.7}** |
| MI subclass | Not reported | **Inferior/Anterior/Lateral** |
| EFA variants | Additive | **Additive + Multiplicative + Harmonic** |
| Image size | Variable | **800×600 standardized** |

---

## GitHub Release

[All data and results](https://github.com/hakmesyo/efa-ecg/releases/tag/ecg_images) are auto-downloaded by the notebook:

`ecg_images.zip` · `sample_1000.csv` · `ground_truth.csv` · `panel_coords.json` · `gemini_outputs.csv` · `claude_outputs.csv` · `llava_outputs.csv` · `qwen_outputs.csv` · `internvl_outputs.csv` · `gemma3_outputs.csv`

---

## Repository Structure

```
efa-ecg/
├── README.md
├── requirements.txt
├── efa_ecg_v2_kaggle.ipynb         ← Recommended: single notebook
├── step1a_sampling.py              ← Reference only
├── step1b_groundtruth.py			← Reference only
├── step1c_rendering.py				← Reference only
├── step2a_gemini_inference.py		← Reference only
├── step3a_occlusion.py				← Reference only
├── step4_efa.py					← Reference only
└── step5_analysis.py				← Reference only
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
