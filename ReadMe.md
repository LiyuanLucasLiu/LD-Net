# LD-Net

[![Documentation Status](https://readthedocs.org/projects/ld-net/badge/?version=latest)](http://ld-net.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LD-Net provides sequence labeling models features:
- **Efficiency**, i.e., constructing *efficient contextualized representations* without retraining language models. 
- **Portability**, i.e., our implementations are well *organized*, easy to *modify* and well *[documented](http://lm-lstm-crf.readthedocs.io/en/latest/)*.

Remarkablely, our pre-trained NER model achieved 92.08 on the CoNLL03 test set, which can conduct decoding w. the speed of 160K words / sec, a **six times** speed-up comparing to its original model.

Details about LD-Net can be accessed at: https://arxiv.org/abs/1804.07827.

- [Model](#model-notes)
- [Benchmarks](#benchmarks)
- [Pretrained model](#pretrained-model)
- [Training](#model-training)
	- [Dependency](#dependency)
	- [Data](#data)
	- [Model](#model)
	- [Command](#command)

## Model Notes

## Benchmarks

## Pretrained Models

## Training

### Dependency

tensorboard wrapper

```
numpy==1.13.1
tqdm
torch-scope
pytorch==0.4.1
```

### Data

### Model

### Command 

