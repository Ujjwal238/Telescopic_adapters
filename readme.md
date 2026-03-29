<div align="center">

<h1 align="center">Telescopic Adapters</h1>

<h3>Improvise, Adapt, Overcome — Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging</h3>

[Ujjwal Mishra](https://scholar.google.co.in/citations?user=Ggw7z6sAAAAJ&hl=en)<sup>1 </sup>, [Vinita Shukla]()<sup>1 </sup>, [Praful Hambarde](https://scholar.google.co.in/citations?user=fHMMcBYAAAAJ&hl=en)<sup>1</sup>, [Amit Shukla](https://scholar.google.ae/citations?user=pjJwY5oAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup> Centre for Artificial Intelligence and Robotics, Indian Institute of Technology Mandi, India



[![WACV 2026](https://img.shields.io/badge/WACV-2026-blue.svg)](https://openaccess.thecvf.com/content/WACV2026/html/Mishra_Improvise_Adapt_Overcome_--_Telescopic_Adapters_for_Efficient_Fine-tuning_of_WACV_2026_paper.html)
[![arXiv paper](https://img.shields.io/badge/arXiv-2512.13855-b31b1b.svg)](https://arxiv.org/abs/2512.13855)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Weights-yellow)](https://huggingface.co/spaces/Ujjwal101/Telescopic_Adapters/tree/main)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://ujjwal238.github.io/Telescopic_adapters/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Ujjwal238.Telescopic_adapters&left_color=%2363C7E6&right_color=%23CEE75F)

[**Overview**](#-overview) | [**Get Started**](#%EF%B8%8F-lets-get-started) | [**Results**](#-results) | [**Citation**](#-citation) | [**Q&A**](#-q--a)

</div>

---

## 🛎️ Updates

* **`March 2026`**: Paper accepted at **WACV 2026** (pages 7605–7615). 🎉
* **`March 2026`**: Repository initialized. Testing code and BKAI checkpoint released. Additional dataset checkpoints coming soon — stay tuned!

---

## 🔭 Overview

**Telescopic Adapters** is a Parameter-Efficient Fine-Tuning (PEFT) framework for adapting Vision Language Segmentation Models (VLSMs) to medical imaging domains. Instead of applying uniform adapter dimensions across all transformer layers, we introduce a **depth-aware telescopic scaling** strategy: adapter capacity grows progressively from shallow to deep layers, mirroring the coarse-to-fine representational hierarchy of transformers.

<div align="center">
  <img src="img/Image_Transformer.pdf" alt="Architecture Overview" width="90%">
  <p><em>Overview of the telescopic adaptation framework: (a) modified CLIPSeg architecture, (b) adapter placement within encoder blocks, (c) adapter formulation with learnable scalar α, (d) progressive dimension allocation across vision, text, and conditional branches.</em></p>
</div>

**Key properties:**
- 🏋️ Only **613k trainable parameters** — 244× fewer than end-to-end fine-tuning
- 🏆 **State-of-the-art** across 5 medical datasets (polyp, skin lesion, breast ultrasound)
- 🚀 No camera poses, optical flow, or additional annotations required
- 🏥 Deployable in resource-constrained clinical environments

<div align="center">
  <img src="img/plot-7.pdf" alt="Performance vs Parameter Efficiency" width="60%">
  <p><em>Performance vs. Parameter Efficiency. Telescopic Adapters sit on the Pareto front — best PEFT performance at the lowest parameter cost.</em></p>
</div>

---

## 🗝️ Let's Get Started!

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended) or CPU
- Conda

### 1. Clone the Repository

```bash
git clone https://github.com/Ujjwal238/Telescopic_adapters.git
cd Telescopic_adapters
```

### 2. Set Up Environment

```bash
conda create -n telescopic_adap python==3.10
conda activate telescopic_adap
pip install -r requirements.txt
```

### 3. Download Pretrained Weights

Download the checkpoint for the **BKAI dataset** from our HuggingFace space:

👉 [https://huggingface.co/spaces/Ujjwal101/Telescopic_Adapters/tree/main](https://huggingface.co/spaces/Ujjwal101/Telescopic_Adapters/tree/main)

> **Note:** Checkpoints for Kvasir-SEG, ClinicDB, ISIC-16, and BUSI will be released soon.

### 4. Configure and Run Inference

Open `infer_single.py` and edit the CONFIG block at the top:

```python
IMAGE_PATH  = "/path/to/your/image.png"   # Input image
PROMPT      = "pink polyp"                 # Text prompt describing the target region
CKPT_PATH   = "/path/to/best.ckpt"        # Path to downloaded checkpoint
OUTPUT_PATH = "output_mask.png"            # Where to save the binary mask
```

Then run:

```bash
python infer_single.py
```

The script will print progress and save a binary segmentation mask to `OUTPUT_PATH`.

### Inference Script Details

`infer_single.py` is a self-contained inference script — no dataset files, annotation JSONs, or project-specific imports needed. It:

1. Loads and preprocesses the input image to `[352, 352]`
2. Tokenizes the text prompt using the CLIPSeg tokenizer
3. Loads `CLIPSegDenseAdapter` weights from the Lightning checkpoint (strips `net.` prefix automatically)
4. Runs a forward pass (fp16 on CUDA, fp32 on CPU)
5. Applies sigmoid + 0.5 threshold and resizes the mask back to the original image dimensions
6. Saves the binary mask as a PNG

**Supported precision:** `fp16` (GPU) or `fp32` (CPU/GPU)

---



> *We'd appreciate it if you could give this repo a ⭐️ **star** ⭐️ and stay tuned for more checkpoint releases!*

---

## 📜 Citation

If this work contributes to your research, please consider citing our paper and giving this repo a ⭐️:

```bibtex
@InProceedings{Mishra_2026_WACV,
    author    = {Mishra, Ujjwal and Shukla, Vinita and Hambarde, Praful and Shukla, Amit},
    title     = {Improvise, Adapt, Overcome -- Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7605--7615}
}
```

---

## 🤝 Acknowledgments

Our model builds upon [CLIPSeg](https://github.com/timojl/clipseg). We also thank the authors of [VLSMAdapt](https://github.com/naamiinepal/vlsm-adapter) for their dataset splits and evaluation protocol which we follow for fair comparison. We thank them for releasing their code openly.

---

## 🙋 Q & A

***For any questions, please feel free to [contact us.](mailto:ujjwalmishra238@gmail.com)***

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Ujjwal238/Telescopic_adapters&type=Date)](https://www.star-history.com/#Ujjwal238/Telescopic_adapters&Date)