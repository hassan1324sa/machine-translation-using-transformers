# Machine Translation Using Transformers

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A PyTorch implementation of a Transformer-based neural machine translation (NMT) model for translating text between languages. This repository includes training, evaluation, and inference pipelines, along with tools for dataset preprocessing.

## Overview

This project implements the Transformer architecture introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) for sequence-to-sequence machine translation. It supports training custom models on parallel corpora (e.g., English-German) and includes utilities for tokenization, batching, and evaluation using metrics like BLEU score.

## Features

- **Transformer Architecture**: Full implementation of encoder-decoder layers with multi-head attention.
- **Custom Tokenizer**: Byte-Pair Encoding (BPE) or subword tokenization for efficient text processing.
- **Training Pipeline**: Flexible training script with configurable hyperparameters.
- **Inference API**: Translate sentences interactively or in batch mode.
- **Evaluation**: BLEU score calculation and attention visualization.
- **Supports Multiple Datasets**: Preprocessing scripts for popular datasets (e.g., OPUS, WMT).

## Installation

**Clone the repository**:
   ```bash
   git clone https://github.com/hassan1324sa/machine-translation-using-transformers.git
   cd machine-translation-using-transformers
   pip install torch==2.0.0 torchtext=0.15.1 numpy pandas tqdm sentencepiece huggingface-hub
   ```
