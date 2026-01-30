# Transparentize the Internal and External Knowledge Utilization in LLMs with Trustworthy Citation

[![ACL 2025 Paper](https://img.shields.io/badge/ACL%202025-Paper-blue)](https://aclanthology.org/2025.findings-acl.919/)
[![PDF](https://img.shields.io/badge/PDF-Download-red)](https://aclanthology.org/2025.findings-acl.919.pdf)
[![HF Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/SHENJJ1017/TransparentizeCitation)

This repository contains the paper, data, and research scripts for **“Transparentize the Internal and External Knowledge Utilization in LLMs with Trustworthy Citation”** (Findings of ACL 2025).

## Dataset summary (from the paper)
The dataset is constructed from three recent RAG datasets: **CRAG**, **FRAMES**, and **SituatedFaithfulnessEval (SFE)**. Each question is paired with five retrieved documents, annotated with whether a document is ground-truth (contains the answer). The dataset is split into **GT** and **nGT** settings depending on whether a ground-truth document is present. See Section 3 of the paper for details.

## LlamaFactory note
`trainer.py` is intended to **replace** the `trainer.py` in LlamaFactory (path may vary by version; commonly `src/llamafactory/train/trainer.py`).

## License
MIT (applies to code and data in this repository).

## Scripts
This directory contains the research scripts used in the paper. They are provided as-is and may require environment setup (CUDA, model checkpoints, or API keys).

### Key files
- `trainer.py`: custom `WeightedSeq2SeqTrainer` used for weighted loss based on sections (CoT, references, answer, confidence, citations).
  - **LlamaFactory integration:** replace LlamaFactory's `trainer.py` with this file (path varies by version; commonly `src/llamafactory/train/trainer.py`).
- `data_gen.py`: generates training data with citation-aware outputs using an LLM backend.
- `data_processing.py`: cleans and normalizes oracle outputs and citations.
- `eval.py`: evaluation utilities for model outputs and reference extraction.
- `reference_eval.py`: reference-level evaluation logic and CLI.
- `system.py`: LLM wrapper for local HF models or OpenAI-compatible APIs.
- `nli.py`, `utils.py`, `useful_functions.py`: helper utilities.
- `upload_hf.py`: upload the HF-ready split in `../data/hf/` to the Hugging Face Hub (fill placeholders in the script).

### Expected environment
These scripts use (at minimum): `python`, `torch`, `transformers`, `numpy`, `tqdm`, `nltk`, and optionally `openai`, `datasets`, `huggingface_hub`.

If you use OpenAI-compatible APIs, set:
- `OPENAI_API_KEY`
- `OPENAI_ORG_ID` (if required)
- `OPENAI_API_BASE` (for Azure / custom endpoints)

### Example usage
Generate data (OpenAI backend):

```bash
python data_gen.py --dataset path/to/input.json --output out.json --total 1500 --max_sample 1
```

Process oracle outputs:

```bash
python data_processing.py
```

Run reference evaluation:

```bash
python reference_eval.py --accuracy_file path/to/preds.json --output_file out.json
```


### Notes
- Paths in the scripts are research-oriented and may need adjustment to your local layout.
- The data generation pipeline expects input items with `question`, `docs`, and `golden answer`-style fields (see `data_gen.py`).
- LlamaFactory repo: https://github.com/hiyouga/LlamaFactory

## Citation
```bibtex
@inproceedings{shen-etal-2025-transparentize,
    title = "Transparentize the Internal and External Knowledge Utilization in {LLM}s with Trustworthy Citation",
    author = "Shen, Jiajun  and
      Zhou, Tong  and
      Chen, Yubo  and
      Qiu, Delai  and
      Liu, Shengping  and
      Liu, Kang  and
      Zhao, Jun",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.919/",
    doi = "10.18653/v1/2025.findings-acl.919",
    pages = "17858--17877",
    ISBN = "979-8-89176-256-5",
    abstract = "While hallucinations of large language models could be alleviated through retrieval-augmented generation and citation generation, how the model utilizes internal knowledge is still opaque, and the trustworthiness of its generated answers remains questionable. In this work, we introduce Context-Prior Augmented Citation Generation task, requiring models to generate citations considering both external and internal knowledge while providing trustworthy references, with 5 evaluation metrics focusing on 3 aspects: answer helpfulness, citation faithfulness, and trustworthiness. We introduce RAEL, the paradigm for our task, and also design INTRALIGN, an integrated method containing customary data generation and an alignment algorithm. Our experimental results show that our method achieves a better cross-scenario performance with regard to other baselines. Our extended experiments further reveal that retrieval quality, question types, and model knowledge have considerable influence on the trustworthiness in citation generation."
}
```
