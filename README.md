# DNLP SS25 Final Project - BERT for Multitask Learning and BART for Paraphrasing Tasks
**Table of Contents**

- [DNLP SS24 Final Project – BERT for Multitask Learning and BART for Paraphrasing Tasks](#dnlp-ss25-final-project-bert-for-multitask-learning-and-bart-for-paraphrasing-tasks)

  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
    - [Paraphrase Identification on Quora Question Pairs (QQP)](#paraphrase-identification-on-quora-question-pairs-qqp)
    - [Sentiment Classification on Stanford Sentiment Treebank (SST)](#sentiment-classification-on-stanford-sentiment-treebank-sst)
    - [Semantic Textual Similarity on STS](#semantic-textual-similarity-on-sts)
  - [Methodology](#methodology)
    - [POS and NER Tag Embeddings](#pos-and-ner-tag-embeddings)
      - [Experimental Results](#experimental-results)
      - [Explanation of Results](#explanation-of-results)
    - [Optimizers](#optimizers)
      - [Sophia](#sophia)
        - [Implementation](#implementation)
        - [Experimental Results](#experimental-results)
        - [Explanation of Results](#explanation-of-results)
    - [Gradient Optimization (Projectors)](#gradient-optimization-projectors)
      - [Pcgrad](#pcgrad)
      - [Experimental Results](#experimental-results)
      - [Explanation of Results](#explanation-of-results)
      - [GradVac](#gradvac)
      - [Experimental Results](#experimental-results)
      - [Explanation of Results](#explanation-of-results)
    - [Schedulers](#schedulers)
      - [Linear Warmup](#linear-warmup)
        - [Experimental Results](#experimental-results)
      - [Plateau](#plateau)
      - [Cosine](#cosine)
      - [Round Robin](#round-robin)
        - [Experimental Results](#experimental-results)
      - [PAL](#pal)
        - [Experimental Results](#experimental-results)
      - [Random](#random)
    - [Regularization](#regularization)
      - [SMART](#smart)
        - [Experimental Results](#experimental-results)
    - [Custom Loss for BART generation](#custom-loss-for-bart-generation)
    - [Noise for training for BART](#noise-for-training-for-bart)
    - [Synonyms for training for BART](#synonyms-for-training-for-bart)
    - [Generally: BART generation](#generally-bart-generation)
    - [Custom Loss for BART detection](#custom-loss-for-bart-detection)
  - [Details](#details)
    - [Data Imbalance](#data-imbalance)
    - [Classifier Architecture](#classifier-architecture)
    - [Augmented Attention](#augmented-attention)
    - [BiLSTM](#bilstm)
    - [Feed more Sequences](#feed-more-sequences)
    - [Hierarchical BERT](#hierarchical-bert)
    - [CNN BERT](#cnn-bert)
    - [Combined Models](#combined-models)
    - [BERT-Large](#bert-large)
    - [PALs](#pals)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Computation Resources](#computation-resources)
  - [Contributors](#contributors)
  - [Licence](#licence)
  - [Usage Guidelines](#usage-guidelines)
    - [Pre-commit Hooks](#pre-commit-hooks)
    - [GWDG Cluster](#gwdg-cluster)
  - [AI-Usage Card](#ai-usage-card)
  - [Acknowledgement](#acknowledgement)
  - [Disclaimer](#disclaimer)

  

<div align="left">

- **Group name:** Seq2Squad
- **Group code:** G14
- **Group repository:** [Seq2Squad - DNLP](https://github.com/moujanmirjalili/dnlp)
- **Tutor responsible:** Frederik Hennecke
- **Group team leader:** Moujan Mirjalili
- **Group member 1:** Moujan Mirjalili,
- **Group member 2:** Farhan Kayhan
- **Group member 3:** Ali Hamza Bashir
- **Group member 4:** Skyler Anthony McDonnell
- **Group member 5:** Hanumanth Padmanabhan


</div>

## Introduction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://https://img.shields.io/badge/Status-Final-blue.svg)
[![Black Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/en/stable/)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./AI-Usage-Card.pdf/)

This repository contains our official implementation of the Multitask BERT project for the Deep Learning for Natural Language Processing course at the University of Göttingen.

A pretrained BERT ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)) model was used as the basis for our experiments. The model was fine-tuned on multiple tasks using a multitask learning approach. The model was trained on the tasks simultaneously, with a single shared BERT encoder and separate task-specific classifiers.

## Requirements

To install requirements and all dependencies and to create the environment with all the required packages, run:


```sh
./setup.sh
```

For setting up the environment on GWDG clusters, run:

 ```sh
./setup_gwdg.sh
```

## Data

The datasets used in this project are described in the table below:

<table>
  <tr>
    <th>Dataset</th>
    <th>Task</th>
    <th>Description</th>
    <th style="width: 300px;">Size / Splits</th>
    <th>Evaluation Metric</th>
  </tr>
  <tr>
    <td>Quora Dataset (QQP)</td>
    <td>Paraphrase Detection</td>
    <td>Two questions are given as input and a binary label (0/1) is the output indicating if they are paraphrases of one another.</td>
    <td>Total: ~200,000 <br /> Train: 121,620 <br /> Dev: 40,540 <br /> Test: 40,540</td>
    <td>Accuracy</td>
  </tr>
  <tr>
    <td>SemEval STS Benchmark Dataset</td>
    <td>Textual Similarity (Regression)</td>
    <td>Two sentences are given as input and their semantic similarity is scored on a continuous scale from 0 (unrelated) to 5 (equivalent meaning).</td>
    <td>Total: 8,579 <br /> Train: 5,149 <br /> Dev: 1,709 <br /> Test: 1,721</td>
    <td>Pearson correlation</td>
  </tr>
  <tr>
    <td>Stanford Sentiment Treebank (SST)</td>
    <td>Sentiment Analysis (Classification)</td>
    <td>Movie review sentences, each labeled as negative, somewhat negative, neutral, somewhat positive, or positive (5-way classification).</td>
    <td>Total: ~11,847 <br /> Train: 7,111 <br /> Dev: 2,365 <br /> Test: 2,371</td>
    <td>Accuracy</td>
  </tr>
  <tr>
    <td>Extended Typology Paraphrase Corpus (ETPC)</td>
    <td>Paraphrase Type detection and Generation</td>
    <td>3,900 sentence pairs annotated with atomic paraphrase types. Used for both classification (type detection) and generation tasks.</td>
    <td>Total: 3,900 <br /> Train: 2,020 <br /> Test (detection): 573 <br /> Test (generation): 701</td>
    <td>Depends on the task (classification / generation)</td>
  </tr>
</table>

## Training

After activating the environment:

- To train the multitask BERT, you should run:

```sh
python multitask_classifier.py --option [pretrain/finetune] --task [sst/sts/qqp/etpc] --use_gpu 
```
- To train the BART paraphrase type detection, you should run:

```sh
python bart_detection.py 
```
You can change the arguments of this task in the *def get_args()* in the file.

- To train the BART generation, you should run:

```sh
python bart_generation.py 
```

There are a lot of parameters that can be set. To see all of them, run `python multitask_classifier.py --help`. The most important general ones that are useful for 3 BERT tasks are:

| Parameter               | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `--batch_size`          | Batch size.                                                             |
| `--epochs`              | Number of epochs.                                                       |
| `--hidden_dropout_prob` | Dropout probability for hidden layers.                                  |
| `--lr`                  | Learning rate.                                                          |
| `--optimizer`           | Optimizer to use. Options are `AdamW` and `LAMB`.                       |
| `--option`              | Determines if BERT parameters are frozen (`pretrain`) or updated (`finetune`).|
| `--use_gpu`             | Whether to use GPU.                                                     |
| `--freeze_layers`       | Number of BERT layers to freeze from the bottom. (0 = No freezing, 9 = Freeze first 9 layers)|
| `--freeze_strategy`     | Strategy for freezing layers. Options are `none`, `bottom`, `top` and `embeddings_only`.|
| `--freeze_embeddings`   | Whether to freeze the embeddings layer.                                 |
| `--pooling`             | Pooling strategy to aggregate token embeddings into sentence representation. Options are `None`, `mean`, `max`, and `attention`.|


The parameters useful for SST task are in the table below:

| **Parameter**           | **Description**                                                         |
| ----------------------- | ----------------------------------------------------------------------- |
| `--label_smoothing`     | Factor for cross_entropy loss. (0.0 = No smoothing, 0.1 = 10% smoothing)|
| `--truncation_strategy` | Truncation strategy for long sequences. Options are `standard`, `head_tail`, `head_only` and `tail_only`.|
| `--max_length`          | Maximum sequence length.                                                |
| `--head_ratio`          | Ration of tokens to keeop on head in head_tail truncation.              |
| `--use_augmentation`    | Use data augmentation for training.                                     |
| `--augment_factor`      | Factor by which to augment the dataset (0.3 = 30% more data)            |
| `--augment_prob`        | Probability of augmenting each word (0.1 = 10% of words)                |
| `--use_lexicon`         | Use lexicon features for sentiment analysis                             |


The parameters useful for QQP are in the table below:

| **Parameter**           | **Description**                                                          |
| ----------------------- | ------------------------------------------------------------------------ |
| `--qqp_swap`            | Whether or not to swap order of sentence pairs for QQP task.             |
| `--sample_size`         | Whether to subset data (randomly in each epoch); if so, what sample size.|
| `--qqp_loss_fn`         | Loss function to use for QQP task. Options are  `binary_cross_entropy_with_logits` and `hinge`.|
| `--add_layers`          | Use deeper sentiment head.                                               |


The parameters useful for STS are in the table below: HANU

| **Parameter**             | **Description**                                                        |
| ------------------------- | ---------------------------------------------------------------------- |
| `--use_sbert_augmentation`| Use SBERT-style data augmentation for STS task                         |


The parameters for bart_detection.py are in the table below:

| **Parameter**    | **Description**                                                       |
| ---------------- | --------------------------------------------------------------------- |
| `--seed`         | Seed for random numbers                                               |
| `--use_gpu`      | Whether to use the GPU.                                               |
| `--epochs`       | Number of epochs.                                                     |
| `--lr`           | Learning rate.                                                        |
| `--batch_size`   | Batch size.                                                           |
| `--dropout`      | Dropout probability.                                                  |
| `--epsilon`      | Magnitude of perturbation for regularization.                         |
| `--lambda_reg`   | Weight for the regularization term.                                   |



The parameters for bart_generation.py are in the table below: BASHIR

| **Parameter**            | **Description**                                                       |
| ------------------------ | --------------------------------------------------------------------- |
| `--seed`                 | Seed for random numbers                                               |
| `--use_gpu`              | Whether to use the GPU.                                               |
| `--batch_size`           | Batch size.                                                           |
| `--epochs`               | Number of epochs.                                                     |
| `--lr`                   | Learning rate.                                                        |

## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in

the `logdir` directory. The best model is saved in the `models` directory.

  
## Results

### Baseline Phase:

- epochs: `10 for BERT` and `5 for BART` 
- learning rate: `1e-5`
- hidden dropout prob: `0.3`
- batch size: `64 for BERT` and `16 for BART` 
- Optimizer: `AdamW` 
  
### Improvement Phase:

Parameters for the multitask training:

- epochs: `1 for QQP` and `10 for SST` and `15 for STS`
- learning rate: `2e-5`
- hidden dropout prob: `0.3`
- batch size: `32 for QQP` and `64 for SST` and `32 for STS`
- optimizer: `AdamW` 
- weight decay: `0.01`

Parameters for BART type detection: 

- epochs: `10`
- learning rate: `2e-5`
- batch size: `16`
- dropout: `0.3`
- epsilon: `1e-5`
- lambda reg: `0.01`
- Optimizer: `Custom AdamW` 
- Loss: `BCEWithLogitsLoss with class weights`
- Evaluation metrics: `Accuracy (per label)` and `Matthews Correlation Coefficient (MCC)`

Parameters for BART generation: 

- epochs: `15`
- learning rate: `3e-5`
- batch size: `32`
- weight decay: `0.01`
- contrastive weight: `0.35`
- scheduler type: `cosine`
- warmup ratio: `0.06`


## Results
### Quora Question Pairs Paraphrase (BERT)

This task determines whether two given questions are paraphrases of each other (binary classification). Unlike semantic similarity, it is a strict yes/no decision about equivalence.

The Quora Question Pairs dataset was used, containing ~200,000 labeled pairs. For example:
- Question 1: “What is the step-by-step guide to investing in the share market in India?”
- Question 2: “What is the step-by-step guide to investing in share market?”
- Label: **No**
- Question 1: “I am a Capricorn Sun Cap moon and cap rising…what does that say about me?”
- Question 2: “I’m a triple Capricorn (Sun, Moon, and ascendant in Capricorn). What does this say about me?”
- Label: **Yes**

- 
#### Comparing Results
- Final Improved Model:
	- Accuracy: ANTHONY
- Baseline Test: 
  - Accuracy ANTHONY
 
  
| Model name                  | Description                                                  | Accuracy |
| --------------------------- |--------------------------------------------------------------|----------|
| Baseline                    | ANTHONY                                                         | 76 %  |
| Hyperparameter Optimization | lr=1e-5, hidden_dropout_prob=0.4                                | 79.2% |
| Hinge Loss Function         | for full results, see `full_data_loss_function_test.csv`        | 33.9% |
| Optimizers                  | ANTHONY                                                         |ANTHONY|
| Pair Swapping               | ANTHONY                                                         | 78.5% |
| Sampling                    | sample_size=10000, epochs=20                                    | 76.9% |
| Pair Swapping & Sampling    | lr=1e-5, hidden_dropout_prob=0.3, sample_size=10000, epochs=20  | 77.3% |
| Activation Function (ReLU)  | lr=1e-5, hidden_dropout_prob=0.3, epochs=10                     | 78.5% |
| Activation Function (SiLU)  | lr=1e-5, hidden_dropout_prob=0.3, epochs=10                     | 77.8% |


### Semantic Textual Similarity (BERT)

This task measures the degree of semantic similarity between two sentences on a continuous scale from 0 (no relation) to 5 (equivalent meaning). Unlike paraphrase detection, this allows for graded similarity.

The SemEval STS Benchmark dataset (~8.5k sentence pairs) was used. Examples with gold similarity scores:
- “The bird is bathing in the sink.”
- “Birdie is washing itself in the water basin.”
- Score: **5 (equivalent)**
- “The woman is playing the violin.”
- “The young lady enjoys listening to the guitar.”
- Score: **1 (same topic, not equivalent)**
- “John went horseback riding at dawn with a whole group of friends.”
- “Sunrise at dawn is a magnificent view to take in if you wake up early enough for it.”
- Score: **0 (different topics)**

### Stanford Sentiment Treebank (BERT)

A basic classification task in understanding a given text is classifying its polarity (i.e., whether the expressed opinion in a text is positive, negative, or neutral). Sentiment analysis can be used to determine individual feelings towards particular products, politicians, or news reports. 
The Standford Sentiment Treebank (SST-5) was used, where each phrase (IMDB movie reviews) has a label of negative, somewhat negative, neutral, somewhat positive, or positive. For example:

- Movie Review: Light, silly, photographed with color and depth, and rather a good time.
Sentiment: 4 (Positive)

- Movie Review: Opening with some contrived banter, cliches and some loose ends, the
screenplay only comes into its own in the second half.
Sentiment: 2 (Neutral)

- Movie Review: ... a sour little movie at its core; an exploration of the emptiness that
underlay the relentless gaiety of the 1920’s ... The film’s ending has a “What was it all for?"
Sentiment: 0 (Negative)

#### Comparing Results
- Final Improved Model:
	- Accuracy: 55.2%
- Baseline Test Result: 
    - Accuracy 51.8%
 
FARHAN


### Paraphrase Type Detection (BART)

While Quora Paraphrase Detection only decides whether two sentences are paraphrases, this task identifies which of the 26 paraphrase types apply to a given sentence pair.
The Extended Typology Paraphrase Corpus (ETPC) was used, where each example may belong to multiple paraphrase types simultaneously. For example:

- Sentence 1: Moore had no immediate comment Tuesday.
- Sentence 2: Moore did not have an immediate response Tuesday.
- Paraphrase types: [6, 11, 15, 29]

This makes the problem a multi-label classification task.

#### Label Conversion
- Types {12, 19, 20, 23, 27} were dropped, leaving 26 valid types.
- Labels were converted to binary vectors of length 26.

#### Comparing Results
- Final Improved Model:
	- Accuracy: 94.3%
	- MCC: 0.722
- Baseline Test: 
  - Accuracy: 90.8%
  - MCC: 0.06


| Model name                           | Description | Accuracy | MCC |
| ------------------------------------ |-------------|----------|---------------|
| BCEWithLogitsLoss without sigmoid    | replaced `BCELoss + Sigmoid` with `BCEWithLogitsLoss` for numerical stability| **94.3%**  | 0.722 |
| Hyperparameter-tuned setup           | Optimal configuration: `lr=2e-5`, `batch_size=16`, `epochs=10`, `dropout=0.3` | **%** | x |
| Regularization + gradient clipping   | Added Gaussian noise to embeddings (`epsilon=1e-5`), penalized logit divergence, combined with classification loss (`lambda_reg=0.01`); used `clip_grad_norm_(max_norm=1.0)`        | **%** | x |
| Class weighting                      | Inverse-frequency class weights (capped at 100) applied via `pos_weight` | **%** | x |
| Label smoothing (removed)            | Tested label smoothing (0→0.1, 1→0.9); removed due to degraded performance on rare classes | **%** | x |
| Learning rate scheduler (removed)    | `ReduceLROnPlateau` reduced LR too early; hurt MCC and accuracy | **%**| x |
| Baseline (no class weights, BCELoss) | Initial version | **90.8%** | 0.06 |



### Paraphrase Type Generation (BART)




---------------------

### Hyperparameter Optimization

We performed systematic hyperparameter optimization using grid search and validation set performance. The optimization process focused on finding the best combination of learning rate, dropout probability, and batch size to maximize performance while minimizing overfitting.


-----------------------

## AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./AI-Usage-Card.pdf/) at the top. The card is based on [https://ai-cards.org/](https://ai-cards.org/).

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John Hewitt, Amelie Byun, John Cho, and their team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html), created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
