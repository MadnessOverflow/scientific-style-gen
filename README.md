# Improving Few-Shot Capabilities of LLMs for Style-Conditioned Text Generation - Codebase

This is the code repository for the masterthesis [Improving Few-Shot Capabilities of LLMs for Style-Conditioned Text Generation](TODO_LINK_TO_MA). It contains the codebase for generating, steering, and evaluating synthetic abstracts. 

## Structure

The codebase is organized into several modules:

- `src/activation_steering`: Contains scripts for contrastive steering (either our approach or [Konen et. al.](https://arxiv.org/abs/2402.01618v1))
- `src/author_classifier`: Implements training and hyperparameter optimization of author classification models.
- `src/datasets`: Scripts that were used for building the datasets. (see huggingface: [scientific-style-gen-data](https://huggingface.co/datasets/MadnessOverflow/scientific-style-gen-data))
- `src/evaluation`: Scripts for evaluating generated abstracts using our proposed metrics.
- `src/style_control_network`: Implementation and training of the style control networks
- `src/inference`: Scripts for generating abstracts.
- `src/training`: Contains scripts for LoRA fine-tuning.
- `src/utils`: Contains shared utilities, prompts, helper functions, and paper extraction logic.

## Environment

The environment for this project was created using Conda and pip. You can find the exact dependencies and recreate the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate faiss_env
```

## Datasets

All datasets used in this study were generated from scratch using the scripts provided in the `src/datasets/` directory. However, the fully processed datasets are also made directly available and can be downloaded from HuggingFace:
**[scientific-style-gen-data](https://huggingface.co/datasets/MadnessOverflow/scientific-style-gen-data)**

Attribution for the papers used in the dataset is provided in `ATTRIBUTION.md`.

## Usage

The scripts are meant to be run directly as modules from the root directory.

For example:
```bash
python -m src.evaluation.evaluate_abstracts
```

## Prediction Workflows

**To generate abstracts with contrastive steering (our method):**
1. Run `src/activation_steering/generate_contrastive_steering_vectors.py`
2. Then run `src/activation_steering/prediction_activation_steering.py`
3. *(Optional)* Run `src/evaluation/evaluate_abstracts.py`

**To generate abstracts with the Style Control Network:**
1. Run `src/style_control_network/prediction_style_control_network.py`
2. *(Optional)* Run `src/evaluation/evaluate_abstracts.py`

**To generate abstracts with the baselines:**
1. Run `src\inference\prediction_baselines.py`
2. *(Optional)* Run `src/evaluation/evaluate_abstracts.py`

---

## Citation & Attribution

If you use these datasets or code in your research, please cite the original thesis and repository:

```bibtex
@mastersthesis{popp2026improving,
  author       = {Leonard Popp},
  title        = {Improving Few-Shot Capabilities of {LLMs} for Style-Conditioned Text Generation},
  school       = {Karlsruhe Institute of Technology (KIT)},
  year         = {2026},
  type         = {Master's Thesis},
  url          = {https://github.com/MadnessOverflow/scientific-style-gen}
}

```

---

> [!WARNING]
> **Disclaimer:** This repository was created during and for a Master's Thesis. It is not intended to be production-ready or perfectly optimized code. It serves primarily as a proof of concept and as a detailed reference for the exact implementation used in the study. It is possible that there are issues introduced through reorganization.
> 
> For more details, please refer to the Master's Thesis:
> **[Improving Few-Shot Capabilities of LLMs for Style-Conditioned Text Generation](TODO_LINK_TO_MA)**

