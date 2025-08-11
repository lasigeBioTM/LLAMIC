# LLaMIC : LLaMA Model Applied to MIMIC  
**Entity Recognition and Linking for Cardiovascular and Cerebrovascular Conditions in Free-text EHRs**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-🤗-yellow)](https://huggingface.co/)  
[![ICD-10](https://img.shields.io/badge/standard-ICD--10-green)](https://icd.who.int/)  

---

## Overview

**LLaMIC** is a modular pipeline designed for clinical entity recognition and standardization, focusing on identifying **cardiovascular and cerebrovascular disease mentions** in unstructured electronic health records (EHR) free-text notes.

LLaMIC combines instruction-tuned large language models (LLMs) with rule-based lexicons to achieve:

- **Named Entity Recognition (NER):** Recognition of disease mentions within ICD-10 codes I20–I25 (ischemic heart diseases) and I60–I64 (cerebrovascular diseases).
- **Named Entity Linking (NEL):** Linking recognized entities to standardized ICD-10 concepts.

<img src="assets/llamic_ner.png" alt="LLaMIC Pipeline" width="800"/>

---


## Installation

Clone the repository and install required dependencies in your environment:

```bash
pip install -r requirements.txt
```

Set the model and lexicon paths in `call_llamic.sh`: Select the entity type (disease or drug) and the LLaMA model for NER, NEL, and review phases. Options include the LLaMA 3.1 8B model (llama3.1) or the path of your own fine-tuned checkpoints. Update the domain lexicon (lexicon.csv) with target ICD-10 codes and descriptions. Adjust the iteration parameter for NER if needed (n_iterations).

```bash
--entity_type disease/drug
--model_name_or_path_ner llama3.1 (default)
--model_name_or_path_nel llama3.1 (default)
--model_name_or_path_review llama3.1 (default)

--lexicon_path $datadir/lexicon.csv
--n_iterations 1 (default)
```


Run the LLaMIC on HPC with SLURM:
```bash
sbatch call_llamic.sh
```

## Results
The fine-tuned LAMIC model was evaluated on cardiovascular and cerebrovascular diseases and ICD-10 entity linking tasks. In MIMIC-IV notes dataset, it achieved precision scores of 0.83 (strict mode) and 0.89 (lenient mode) for disease identification. For ICD-10 linking, the model reached a precision of approximately 0.82.

## References

https://physionet.org/content/mimic-iv-note/2.2/

