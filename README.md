# Polymer-solubility-VLM
Polymer solubility classification with object detection in Vision Language Model.

This repository contains the Vision–Language Model (VLM) code used for polymer solubility classification with object detection.

The workflow is based on PaliGemma 2 and includes model fine-tuning, evaluation, inference, and dataset splitting scripts.

Repository Contents
finetune_paligemma2.py – model fine-tuning
evaluate_paligemma2.py – model evaluation
evaluate_all_checkpoints_no_images.py – evaluation of trained checkpoints
inference_paligemma2.py – inference workflow
inference_list_paligemma2.py – inference on image lists
inference_single_paligemma2.py – inference on individual images
result_paligemma2.py – processing of model outputs
split.py – dataset splitting
dataset_split_summary.csv – summary of dataset splits
vlm_checkpoint_accuracy_summary.csv – checkpoint accuracy results
vlm_checkpoint_predictions.csv – model predictions

The dataset_ and dataset_resized directories contain JSON files used to define the datasets and image annotations used by the VLM workflow.

Model

The implementation uses the PaliGemma 2 vision–language model through the Hugging Face Transformers library.

Code Contribution

The VLM training, evaluation, and inference code was developed by Zhengxue Zhou and used and released with permission as part of this research project.

The repository was organised and released by Seda Uyanik as part of the polymer solubility classification work conducted during her PhD research.

