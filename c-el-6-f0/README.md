---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: c-el-6-f0
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# c-el-6-f0

This model is a fine-tuned version of [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5420
- Proba Mse: 0.0199
- Proba Pearson: 0.8429

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 9e-06
- train_batch_size: 32
- eval_batch_size: 128
- seed: 1
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-06
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Proba Mse | Proba Pearson |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:-------------:|
| 0.576         | 1.0   | 910  | 0.5589          | 0.0270    | 0.7825        |
| 0.5404        | 2.0   | 1820 | 0.5390          | 0.0206    | 0.8350        |
| 0.5244        | 3.0   | 2730 | 0.5429          | 0.0203    | 0.8425        |
| 0.5172        | 4.0   | 3640 | 0.5420          | 0.0199    | 0.8429        |


### Framework versions

- Transformers 4.20.0.dev0
- Pytorch 1.11.0+cu113
- Datasets 2.2.3.dev0
- Tokenizers 0.12.1
