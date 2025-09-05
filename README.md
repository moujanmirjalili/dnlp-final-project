# DNLP SS25 Project - BERT for Multitask Learning and BART for Paraphrasing Tasks

**Table of Contents**

- [DNLP SS25 Final Project - BERT for Multitask Learning and BART for Paraphrasing Tasks](#dnlp-ss25-project---bert-for-multitask-learning-and-bart-for-paraphrasing-tasks)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
    - [Quora Question Pairs Paraphrase (BERT)](#quora-question-pairs-paraphrase-bert)
    - [Semantic Textual Similarity (BERT)](#semantic-textual-similarity-bert)
    - [Stanford Sentiment Treebank (BERT)](#stanford-sentiment-treebank-bert)
    - [Paraphrase Type Detection (BART)](#paraphrase-type-detection-bart)
    - [Paraphrase Type Generation (BART)](#paraphrase-type-generation-bart)
  - [Improvements](#improvements)
    - [Quora Question Pairs Paraphrase (BERT)](#quora-question-pairs-paraphrase-bert-1)
    - [Semantic Textual Similarity (BERT)](#semantic-textual-similarity-bert-1)
    - [Stanford Sentiment Treebank (BERT)](#stanford-sentiment-treebank-bert-1)
    - [Paraphrase Type Detection (BART)](#paraphrase-type-detection-bart-1)
    - [Paraphrase Type Generation (BART)](#paraphrase-type-generation-bart-1)
  - [Dataset](#dataset)
    - [Quora Question Pairs Paraphrase (BERT)](#quora-question-pairs-paraphrase-bert-2)
    - [Semantic Textual Similarity (BERT)](#semantic-textual-similarity-bert-2)
    - [Stanford Sentiment Treebank (BERT)](#stanford-sentiment-treebank-bert-2)
    - [Paraphrase Type Detection (BART)](#paraphrase-type-detection-bart-2)
    - [Paraphrase Type Generation (BART)](#paraphrase-type-generation-bart-2)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
    - [Quora Question Pairs Paraphrase (BERT)](#quora-question-pairs-paraphrase-bert-3)
    - [Semantic Textual Similarity (BERT)](#semantic-textual-similarity-bert-3)
    - [Stanford Sentiment Treebank (BERT)](#stanford-sentiment-treebank-bert-3)
    - [Paraphrase Type Detection (BART)](#paraphrase-type-detection-bart-3)
    - [Paraphrase Type Generation (BART)](#paraphrase-type-generation-bart-3)
  - [Members' Contribution](#members-contribution)
  - [AI-Usage Card](#ai-usage-card)
  - [Acknowledgement](#acknowledgement)
  - [References](#references)

  

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



The parameters for bart_generation.py are in the table below: 

| Argument                    | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| `--seed`                    | Random seed for reproducibility                                  |
| `--use_gpu`                 | Use GPU if available (set flag to enable)                        |
| `--num_epochs`              | Number of training epochs                                        |
| `--lr`                      | Learning rate for optimizer                                      |
| `--grad_accum_steps`        | Gradient accumulation steps                                      |
| `--contrastive_weight`      | Contrastive loss weight                                          |
| `--weight_decay`            | Weight decay for optimizer                                       |
| `--scheduler_type`          | Scheduler type for learning rate (`cosine`, `linear`, or `none`) |
| `--warmup_ratio`            | Warmup ratio (fraction of total steps)                           |
| `--early_stopping_patience` | Patience for early stopping (in epochs)                          |
| `--save_path`               | Path to save the best model checkpoint                           |
 
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

Parameters for the multitask training BERT:

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

- epochs: `10`
- learning rate: `3e-5`
- batch size: `32`
- weight decay: `0.01`
- contrastive weight: `0.35`
- scheduler type: `cosine`
- warmup ratio: `0.06`
- early stopping patience: `3` 
- grad accum steps: `1`  
- contrastive weight: `0.35` 

## Results

In this section, each task is briefly explained. The **Comparing Results** subsection then presents the final improved model's performance on the development set, compared to the baseline results on the test set. Finally, a table summarizes each improvement along with the corresponding score achieved.

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
	- Accuracy: 76.6%
 
  
| Model Name                  | Description                                                  | Accuracy |
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

#### Comparing Results
- Final Improved Model:
	- DePearson Correlation: HANU
- Baseline Test: 
	- DePearson Correlation: 0.38

| Model Name 							| Description    | DePearson Correlation |
| ------------------------------------- | -------------- | --------------------- |
| Baseline 					  			| HANU			 | 0.357 |
| Cosine Similarity			   			| HANU			 | 0.428 |
| Similarity Head 			   			| HANU			 | 0.475 |
| LayerNorm in Similarity Head			| HANU			 | 0.601 |
| Mean Pooling 							| HANU			 | 0.825 |
| Contrastive + MSE 					| HANU			 | 0.827 |
| Simplify head + hyperparameter tuning | HANU			 | 0.852 |
| Bert-Large 							| HANU			 | 0.868 |

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
    - Accuracy: 51.8%
 
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

Unlike paraphrase detection or classification, which decide whether two sentences are paraphrases or which paraphrase types apply, this task is generative. Given one sentence and a list of target paraphrase types, the model must generate a new paraphrase that reflects those specific transformations.

The Extended Typology Paraphrase Corpus (ETPC) was used for this task. Each sentence pair is annotated with atomic paraphrase types, which guide the generation process. Example:

- Input sentence: “Moore had no immediate comment Tuesday.”
- Paraphrase types: [6, 11, 15, 29]
- Generated paraphrase: “Moore did not have an immediate response Tuesday.”

#### Comparing Results
- Final Improved Model:
	- Accuracy: BASHIR
	- BLEU Score: BASHIR
- Baseline Test: 
	- Accuracy: 44.9%
	- BLEU Score: 14.50

| Model  Name         | Description                                             | Penalized BLEU (%) |
| ------------------- | ------------------------------------------------------- | ------------------ |
| facebook/bart-large | Baseline fine-tuning on ETPC (80/20 split)              | 12.53              |
| facebook/bart-large | + WordNet synonym augmentation (15% replacement)        | 8.13 / 16.2 (some experiments with unstable score and loss) |
| facebook/bart-large | + Contrastive loss (weight ≈ 0.35)                      | 16.1               |
| facebook/bart-large | + Cosine LR scheduler with warm-up (6%)                 | 21.51              |
| facebook/bart-large | + Back-translation augmentation (≈20–25% data)          | [X.XX]             |
| facebook/bart-large | Two-stage training (MultiPIT pretrain → ETPC fine-tune) | 25.18              |


---------------------
## Improvements 

Detailed explanations of the improvements made for each task are provided in this section.

### Quora Question Pairs Paraphrase (BERT)

#### Loss Function
ANTHONY: Describe loss function experiment (hinge vs default) (for full data, see `full_data_loss_function_test.csv`)

#### Pair Swapping
ANTHONY: Polish this description

This experiment tested the affects of alternating the order in which sentences in each pair in the training data were fed into the model with each epoch. The hypothesis was that this swapping would help prevent overfitting, resulting in development accuracy closer to what it would be with a training dataset twice the size. This did not turn out to be the case, with the highest dev_acc reached (with batch size 16, 10 epochs, optimizer AdamW, hidden dropout probability 0.3, and learning rate 1e-5) being 0.788 without swapping and 0.785 with (for full data, see `full_data_swapped_pairs.csv`). When trained on half the data (the first 67571 entries in `quora-paraphrase-train.csv`) and the same hyperparameters, the highest dev_acc reached was 0.768 and 0.765, respectively (for full data, see `swap_pairs_half_data.csv`).

#### Sampling in Place

This improvement consisted of training off of a random sample of the training data, sampled freshly each epoch. The size of the sample is an integer given via command-line argument (`--sample_size`). Because there were significantly more training examples provided for the QQP task than other tasks (which therefore took a siginificant amount of time and energy to process), this improvement aimed to increase training efficiency.

The main benefit provided by sampling was decreased runtime. With a sample size of 10,000, 20 epochs could be completed in 39 minutes (97.5 core-hours, job 10629427), compared to 10 epochs in 137 minutes (1370 core-hours) without sampling (n ≈ 135,000, job 9810386; for full data, see `test_sampling.csv`). This increased efficiency came at the cost of a slight dropoff in performance (with the best development accuracy achieved in the first 10 epochs being 0.760 (learing rate 1e-5, hidden dropout probability 0.3, batch size 16), compared to 0.788 when run for 10 epochs with the same hyperparameters without sampling), but given the 28-fold decrease in core-hours consumed per epoch (137 to 4.875), __ (it wasn't significant).

#### Optimizers
ANTHONY: Describe optimizers tested

#### Activation Functions

This improvement involved testing the performance of three activation functions applied to the output of the feed forward layer in BertLayer (`self.interm_af`). The activation functions tested were GELU (Gaussian Error Linear Unit, the default option), ReLU (Rectified Linear Unit), and SiLU (Sigmoid Linear Unit). SiLU was chosen for analysis because it was mentioned in the slides from lecture 9 (slide 84) as providing improvements over the transformer architecture as introduced in Attention is All You Need.

The highest performance reached in terms of development accuracy for each activation function was as follows (lr=1e-5, hidden_dropout_prob=0.3, batch_size=16, epochs=10; for full data, see `activ_fn.csv`):

| Sampling \ Activation Function | GELU      | ReLU  | SiLU  |
| ------------------------------ | --------- | ----- | ----- |
| **Full Training Data** 		 | **0.788** | 0.785 | 0.778 |
| **With Sampling (n=10000)**    | 0.760     | 0.752 | 0.749 |

The best performance from training on both the full data and a sample thereof came from GELU, followed by ReLU and SiLU.

One potential disadvantage of SiLU that was anticipated was slower evaluation (as compared to ReLU) due to i ANTHONY

### Semantic Textual Similarity (BERT)

To improve the semantic textual similarity task, we progressively refined our similarity prediction approach through the following steps:

#### Cosine Similarity for Semantic Relevance:
We replaced the original similarity prediction with cosine similarity between sentence embeddings. We estimated that cosine similarity would capture the semantic closeness better and this was a good first step. This simple change increased Pearson correlation from 0.357 to 0.428.

#### Neural Network Similarity Head:
Instead of a fixed similarity function, we introduced a small neural network as the similarity head. We wanted to see if this would allow the modelto learn more complex interactions between the sentence embeddings. This increased correlation to 0.475

#### Layer Normalization in the Similarity Head:
The NN we used earlier did not have a normalization layer. We decided that incorporating a LayerNorm in the similairty head would stabilize the training process and hopefully improve generalization. This yielded a substantial increase to 0.601.

#### Mean Pooling of Token Embeddings:
The next logical step was to look at how the embeddings were represented. CLS is optimized for classification tasks and might not be capturing the slight nuanced semantic information with sentences. We replaced the CLS token embedding with mean pooling across all token embeddings. This generated the most significant jump in correlation with a value of 0.825.

#### Combined Contrastive and MSE Loss:
To further refine the embedding space, we combined contrastive loss, encouraging closer embeddings for similar sentences, with mean squared error loss to maintain regression accuracy. This synergy slightly improved correlation to 0.827.

#### Deeper and then simplify:
We tried to make changes to the head to try and check if a deeper head would help improve the correlation. It did improve it to 0.839. However, this was too deep and was overfitting the model. A simplified approach with better hyperparameter tuning would be better optimized to the STS task. This is exactly what happened with correlation improving to 0.852.

#### Bert-large:
`bert-large` is larger and has more attention heads. It models semantic relationships better which is important for STS. Dependening on the use case, it might not be justified to use the larger model due a combination of factor, however, for performance improvements in this specific task it does the job. Correlation improved from 0.852 to 0.868.

#### Explored but Approaches that were ineffective in moving the needle:
We experimented with several other methods that did not improve results.
  - Concatenating CLS and mean pooled embeddings
  - Bilinear similarity functions
  - Ranking loss in addition to MSE
  - Cross-attention embeddings

### Stanford Sentiment Treebank (BERT)

#### Data Augmentation:

- Motivation: The SST-5 dataset, while widely used, is relatively small and may lack sufficient linguistic diversity. We hypothesized that data augmentation could improve generalization by exposing the model to varied phrasings of similar sentiment, encouraging robustness to syntactic variation.
- Methods and Expectations: I tested two main augmentation strategies:
    - Easy Data Augmentation (EDA): Includes synonym replacement, random insertion, random swap, and random deletion. Applied with an *augment_factor* controlling the proportion of words modified. I expected moderate but reliable gains due to naturalistic variations.
- Back Translation: Sentences translated to German (via a pre-trained translation model) and back to English. I expected higher-quality paraphrases leading to stronger generalization, though at higher computational cost.
- I expected EDA to provide a solid improvement, while back translation might offer larger gains if semantic integrity was preserved.
- Outcome:
    - EDA was highly effective: Increasing the *augment_factor* from 0.4 to 0.6 consistently improved accuracy, confirming that controlled lexical and structural variation helps the model learn invariant sentiment features.
    - Back translation failed to improve performance and was removed from the pipeline. We suspect that the round-trip translation introduced subtle semantic distortions or unnatural phrasings not typical of movie reviews, thereby confusing the model rather than aiding generalization.
- Conclusion: Simple, rule-based augmentation (EDA) outperformed complex, model-based methods in this domain, likely due to better preservation of sentiment semantics.

#### Input Truncation Strategy:

- Motivation: BERT has a maximum input length of 512 tokens. Many SST-5 examples exceed this limit, requiring truncation. Since sentiment cues can appear at the beginning (e.g., opening sentiment statements) or end (e.g., concluding opinions), naive truncation risks discarding critical information.
- Methods and Expectations: I compared three truncation strategies:
    - Head-only: Keep first 512 tokens
    - Tail-only: Keep last 512 tokens
    - Head-Tail: Retain first *x%* and last *(1-x)%* of tokens, concatenated within the 512-token limit
- I expected head-tail to perform best by preserving sentiment signals from both ends of the text.
- Outcome:  
    - Head-tail strategy was superior, validating our hypothesis. It allowed the model to access both initial impressions and final judgments.
    - Further optimization showed that reducing `max_length` from 512 to **384** improved performance slightly. This suggests that focusing on salient portions (head and tail) while trimming less-relevant middle content reduces noise and enhances focus.
    - Optimal split: 70% head, 30% tail (empirically determined).

#### Layer Freezing Strategy:

- Motivation: Fine-tuning all BERT parameters on small datasets like SST-5 can lead to overfitting. Lower layers of BERT encode general linguistic features (e.g., syntax, morphology), which are already well-learned during pre-training. Freezing them may stabilize training and improve generalization.
- Methods and Expectations: We experimented with freezing:
    - Only the embedding layer
    - Bottom *k* layers (k = 4, 8, 12)
    - Top layers (least expected to help)
- I expected freezing bottom layers to be optimal, preserving foundational representations while allowing higher, more task-specific layers to adapt.
- Outcome: 
    - Freezing bottom 8 layers (including embeddings) yielded the best results.
    - This strategy significantly improved accuracy and stabilized training.
    - Freezing only the embedding layer or top layers led to inferior performance.
    - Conclusion: A hybrid approach preserving low-level linguistic knowledge while adapting high-level semantics, proved most effective.

#### Lexicon-Enhanced Features

- Motivation: BERT’s contextual embeddings may struggle with ambiguous words whose sentiment depends heavily on context (e.g., "bad" in "bad good"). We hypothesized that augmenting inputs with external sentiment knowledge from lexicons could guide the model in borderline cases.
- Methods and Expectations: We implemented a *—use_lexicon* flag to inject features from established sentiment lexicons:
    - VADER: Rule-based sentiment scores
    - SentiWordNet: Word-level positivity/negativity scores
- These features were either concatenated to the [CLS] vector or used as auxiliary inputs. I expected a small but consistent boost, especially for neutral or weakly polarized examples.
- Outcome:  
    - Lexicon features increased accuracy from **54.7% to 54.9%** in one experiment.
    - The improvement was modest but reproducible.
- Conclusion: External sentiment knowledge complements BERT’s internal representations, particularly in ambiguous cases, though the benefit is limited in magnitude.

#### Label Smoothing:

- Motivation: Sentiment labels are inherently subjective. Standard cross-entropy loss assumes hard labels (e.g., class 3 = 100% neutral), which can encourage overconfidence and poor calibration. Label smoothing softens targets to prevent overfitting.
- Methods and Expectations:I replaced standard cross-entropy with **label smoothing loss** (smoothing factor = 0.1):
    - True class: 0.9
    - Others: 0.025 each (for 5 classes)
- I expected this to improve generalization and model calibration, especially on ambiguous inputs.
- Outcome:
    - Label smoothing was one of the most effective techniques.
    - It consistently improved accuracy across experiments.
    - Prevented overconfidence and enhanced performance on borderline sentiment examples.
    - Became a **core component of the final model**, contributing directly to the final accuracy of **55.2%**.
- Conclusion: A simple yet powerful regularization method for subjective classification tasks.

#### BiLSTM on BERT Hidden States:
- Motivation: While BERT captures contextual information bidirectionally, adding a recurrent layer on top might better model sequential dynamics or combine information across multiple layers.
- Methods and Expectations: We added a **BiLSTM** on top of BERT outputs:
    - Input: Last hidden layer only
    - Input: Concatenated last 5 hidden layers(???)
- I expected richer representations from combining deep layers would lead to significant gains.
- Outcome: 
    - Last layer only: No improvement
    - Last 5 layers concatenated: Accuracy increased from **52.8% to 54.0%**
    - The improvement suggests that deeper contextual fusion is beneficial
    - Full potential realized only when combined with other optimizations (e.g., EDA, truncation)
    - Conclusion: Hierarchical modeling of BERT’s internal states adds value, but only when sufficient context is provided.

#### Ensemble Models:
- Motivation: Ensemble methods reduce variance and overfitting by averaging predictions from diverse models. We expected that averaging outputs from multiple BERT instances trained with different random seeds would yield a more robust classifier.
- Methods and Expectations:
    - Trained 3–5 BERT models with different seeds
    - Combined predictions via **probability averaging**
    - Expected modest but consistent improvement
- Outcome  
    - No performance gain observed
    - Individual models showed high agreement; errors were **correlated**
    - Likely due to identical data, architecture, and hyperparameters limiting diversity
- Conclusion: Ensembles require sufficient model diversity (e.g., different architectures, data sampling, or initialization) to be effective. In this setup, ensembling added cost without benefit.

### Paraphrase Type Detection (BART)

#### Class Imbalance:
- Class imbalance is a critical challenge in multi-label classification tasks such as paraphrase type detection, where some paraphrase types occur far more frequently than others. In the ETPC dataset, certain types such as lexical substitution appear commonly, while others such as syntactic reordering or generalization are rare. This leads to a long-tailed label distribution. If unaddressed, the model becomes biased toward frequent labels, achieving high overall accuracy but performing poorly on rare yet meaningful paraphrase types. To counter this, inverse-frequency class weighting was implemented, assigning higher loss weights to underrepresented classes based on how infrequently they appear in the training data. 
- To be specific, the frequency of each label is computed as the proportion of samples where it is present, and the weight for each class is set to the inverse of this frequency, clipped to a maximum value of 100 to prevent excessive dominance. These weights are passed to *BCEWithLogitsLoss* via the *pos_weight* parameter, ensuring that errors on rare labels contribute more to the gradient than errors on common ones. By prioritizing rare classes during training, the model learns to recognize subtle paraphrase patterns that might otherwise be ignored.  
- This leads to more balanced predictions across all 26 types and directly improves metrics like Matthews Correlation Coefficient (MCC), which are sensitive to performance on minority classes. Unlike data augmentation or resampling techniques, class weighting is computationally efficient and easy to implement without modifying the data or training loop structure. As a result, it serves as an ideal first step improvement that is simple, safe, and highly effective at enhancing model robustness and generalization in imbalanced multi-label scenarios.
- Why is data augmentation not ideal here? Data augmentation is not ideal for paraphrase type detection task because it risks introducing semantic inaccuracies or label noise that can mislead the model during training. This task requires precise alignment between sentence pairs and their corresponding paraphrase types such as lexical, syntactic, or structural transformations. Augmenting data through automated methods like back-translation or random insertion could disrupt these subtle linguistic patterns, creating examples that appear similar but do not reflect the true paraphrase type, leading to incorrect supervision.Given that the goal is fine-grained classification of paraphrase *types* rather than binary paraphrase identification, preserving label fidelity is more important than increasing data volume, which is why I chose not to use data augmentation.
#### Smoothing:
- I thought label smoothing would help because it’s a common technique to prevent overfitting by making the model less confident in its predictions, which can improve generalization. Given the complexity of paraphrase type detection, I hoped it would make the model more robust to noise and uncertainty in multi-label annotations. 
- However, in my experiments, it didn’t work and actually hurt performance. I believe this is because the task relies heavily on precise label distinctions especially for rare paraphrase types and smoothing the binary labels (turning 1s into 0.9 and 0s into 0.1) weakens the already sparse supervision signal. This made it harder for the model to learn clear decision boundaries, particularly for infrequent classes, ultimately reducing MCC and overall accuracy. As a result, I removed it in favor of more effective strategies.
#### Regularization:
- This approach is a form of regularization, specifically known as adversarial regularization or noise injection regularization. I implemented this regularization approach because standard fine-tuning can lead to models that are overly sensitive to small input changes, resulting in poor generalization. The goal of this method is to encourage the model to produce stable and smooth predictions by making its output robust to minor perturbations in the input embeddings. To achieve this, I compute the model’s logits on the original input and then add small random noise to the token embeddings simulating slight variations in the input space. The model’s output is then recomputed on this perturbed input, and a smoothness loss is calculated as the L2 norm between the original and perturbed logits. This penalty discourages large changes in predictions due to tiny input fluctuations, effectively regularizing the model’s decision boundaries.
- The total loss is a combination of the original classification loss and this smoothness term, controlled by a hyperparameter *lambda_reg*. By including this during training, the model learns to rely less on fragile, high-variance features and instead focuses on more robust linguistic patterns. Unlike label smoothing which hurt performance by weakening the already sparse supervision signal, this method operates on the model’s internal representations without altering the ground-truth labels, preserving the integrity of the multi-label structure. It also avoids the risks of data augmentation, which could introduce semantically invalid paraphrase pairs. In practice, this led to more stable training dynamics and improved performance on the development set, confirming that the model had learned more generalized and reliable features. As a result, this regularization technique proved effective and was retained in the final training pipeline.
#### Drop out:
- I included dropout in the model architecture because it helps with preventing overfitting. By randomly setting a fraction of the hidden unit activations to zero during training controlled by the *args.dropout* parameter (0.3) the model is forced to avoid relying too heavily on any single neuron or feature, promoting more distributed and robust representations.
- In my implementation, dropout is applied both after the BART model’s [CLS] token representation and reused during the adversarial regularization step. Specifically, the same self.dropout layer is used on the clean and perturbed hidden states, ensuring consistency in regularization. This dual use strengthens the model’s robustness by preventing it from memorizing fragile feature combinations that only exist in the unperturbed input. 
- Unlike label smoothing which I removed because it weakened the already sparse supervision signal for rare paraphrase types, dropout works directly on the model’s internal activations without altering the label space or introducing noise into the ground truth. In practice, I observed that training was more stable and dev performance improved with dropout enabled, confirming its positive impact. Therefore, I kept it in the model’s design.
#### Learning Rate Scheduler:
- I initially included a learning rate scheduler *ReduceLROnPlateau*. The idea was to reduce the learning rate when the dev accuracy plateaued, allowing the model to make finer updates and potentially converge to a better solution. This can be especially helpful in later training stages when large parameter updates might overshoot optimal weights.
- However, I removed it after observing that it did not improve performance and in some cases led to premature convergence. In my experiments with the BART-based paraphrase detection model, the learning rate was being reduced too early often within the first few epochs due to small fluctuations in dev accuracy that the scheduler interpreted as stagnation.
- Plus, BART models are typically fine-tuned with a small, constant learning rate, and aggressive scheduling can disrupt the delicate adaptation of pre-trained weights to the downstream task. Unlike methods like dropout or class weighting which provided consistent benefits, removing the scheduler led to more stable and reproducible results. I concluded that for this task, a fixed learning rate with AdamW optimization was sufficient and more reliable, making the scheduler unnecessary.
#### Adversarial Regularization and Gradient Control:
- I incorporated adversarial embedding perturbation and gradient clipping to improve the stability and generalization of the model. The adversarial regularization works by adding small Gaussian noise controlled by *args.epsilon* to the input token embeddings during training and then penalizing large changes in the model’s output logits. This encourages the model to produce consistent predictions even under minor input variations, leading to smoother decision boundaries and reduced overfitting. Unlike methods that alter the labels or data distribution, this approach operates directly on the model’s internal representations, preserving label fidelity while enhancing robustness. 
- Additionally, I applied gradient clipping (torch.nn.utils.clip_grad_norm_ with max_norm=1.0) during optimization to prevent exploding gradients. This ensures that parameter updates remain stable, especially when using higher learning rates or deep architectures like BART-large. Together, these techniques make the training process more resilient and help the model generalize better to unseen paraphrase patterns. Both improvements are complementary to other regularization strategies like dropout and class weighting. In practice, they contributed to more consistent convergence and improved performance on the development set.
#### Loss:
- I initially used *BCELoss* combined with *nn.Sigmoid()* in the model’s forward pass, which is a common pattern for multi-label classification. However, I later replaced this with *BCEWithLogitsLoss* and removed the *nn.Sigmoid()* layer from the model. This change was motivated by numerical stability and training reliability.
- The key issue with using *BCELoss* after a sigmoid is that it operates on probabilities that have already been squashed into the (0, 1) range. When these values are very close to 0 or 1 they can lead to infinite or undefined loss values. In contrast, *BCEWithLogitsLoss* combines the sigmoid and the binary cross-entropy loss into a single, numerically stable operation that works directly on raw logits. 
- Additionally, by removing *nn.Sigmoid()* from the forward pass, I ensured that the model outputs raw logits during training and only applies the sigmoid function at inference time when computing probabilities for prediction. During training, the loss function benefits from stable gradient computation, and during inference, I can apply the sigmoid only when needed to interpret the outputs.
- This change led to smoother and more consistent performance across epochs and led to huge a improvement.
#### Hyperparameter Tuning:
- I performed hyperparameter tuning to find the optimal configuration for my model. This involved experimenting with key training parameters: learning rate, batch size, and number of epochs.
- I tested learning rate of 1e-5, 1e-4, and 1e-3, but found that **2e-5** provided a good balance between convergence speed and stability, allowing the optimizer to make steady progress without overshooting optimal weights.
- For batch size, I compared values of 16, 32, and 64. While larger batches can offer more stable gradient estimates, I observed that a batch size of **16** worked best in practice.
- Regarding the number of epochs, I evaluated 5, 10, and 15. I observed that **10** epochs provided the best trade-off, allowing the model to fully learn from the training data without memorizing it.
- This tuning process ensured that the final model was trained under conditions that maximized performance and stability, forming a solid foundation for the application of other improvements like class weighting and adversarial regularization.

### Paraphrase Type Generation (BART)

#### Baseline Fine-Tuning 

**Setup:** We fine-tuned **BART** on the **ETPC dataset** (80/20 split) without augmentation or extra loss terms. Inputs were sentence + type-ids, trained to generate the target paraphrase. Training ran up to **10 epochs**.

**Results:** The baseline reached a **Penalized BLEU of 12.53%** on validation. After the **2nd epoch**, validation loss began to rise, showing overfitting, while Penalized BLEU fluctuated across epochs.

**Takeaway:** The baseline shows that BART can generate paraphrases out-of-the-box but tends to overfit quickly. This establishes a starting point and highlights the need for methods that improve diversity and adherence to type labels.

#### Lexical Augmentation with Synonyms

**Setup:** We augmented ~15% of training examples with **WordNet-based synonym replacements** (12% of words per sentence), keeping type labels unchanged.

**Motivation:** Encourage the model to generalize beyond exact word forms, reduce overfitting, and improve **Penalized BLEU** by producing outputs less tied to the input wording.

**Outcome:** Performance **slightly degraded** Penalized BLEU dropped below baseline to **8%** but in some experiments it went to like **16%** but with massive increase of validation loss it went to 2.06 from 1.2. Likely causes:

- Augmentation introduced **label noise**, since substitutions often broke the alignment with type labels.
- BART’s pretraining already covers lexical variation, so added data only confused training.
- Validation still plateaued early, showing no overfitting improvement.

**Decision:** Tihs could have been a good way to use but we dropped synonym augmentation in later experiments, shifting focus toward **objective-level methods** for encouraging diversity.

**Note:** Code for the synonms is still there it just that we are not using it anymore.

#### Contrastive Learning for Diversity 

**Setup:** We added a **contrastive loss term** to discourage outputs that are too similar to inputs. The total loss was:
$$
L_\text{total} = L_\text{MLE} + \alpha \cdot L_\text{contrastive}
$$
using cosine similarity between encoder embeddings of the input and its generated output. We tested α values and found **0.35** worked best.

**Decoding Change:** Alongside this, we updated the decoding strategy from the earlier **beam size = 5 (no grouping/diversity)** to a more diverse setup:

- **Beam size = 6**
- **2 beam groups**
- **Diversity penalty = 1.0**

This decoding configuration, introduced here for the first time, further encouraged output variation.

**Motivation:** Encourage paraphrases that preserve meaning but differ in expression, directly aligning with the **Penalized BLEU** objective (rewarding divergence from the input while preserving fidelity).

**Outcome:** With contrastive loss and improved decoding, **Penalized BLEU** increased from *12.53%* (baseline) to *16.1%* (a relative gain of ~3.6%). Outputs showed more lexical and structural variety. Higher α sometimes reduced fidelity, but 0.35 gave the best trade-off. Training stabilized but still plateaued after the initial improvements. We also **monitored validation loss** alongside Penalized BLEU to track overfitting and it was stable this time but still were showing some fluctuations in loss and penalized bleu score meaning **fixed learning** rate was creating the problem.

**Decision:** Contrastive learning combined with diverse decoding proved effective and was adopted as the new **baseline configuration** for subsequent experiments.

#### Learning Rate Scheduling

**Setup:** Building on the contrastive framework, we introduced a **learning rate scheduler with warm-up and decay**. Both cosine and linear schedules were implemented, but **cosine with a 6% warm-up** proved most effective. The initial learning rate remained `3e-5`, training ran for up to **15 epochs** increased from 10 epochs of before as model were able to learn with stability even after 10 epochs more than 15 could have worked as well but computational contstraint didn't allow this, with early stopping (patience = 3) was also introuduced.

**Additional Introductions:**

- **Linear scheduling** (tested, but minimal gains).
- **Cosine scheduling with warm-up** (chosen as default).
- **Weight decay = 0.01** for regularization.
- **Early stopping monitored with Penalized BLEU** (patience = 3), while **best model saving was based on validation loss**.

**Decoding:** Same as Part 3  **beam size = 6**, **2 beam groups**, **diversity penalty = 1.0**, max length 50.

**Outcome:** Training became **more stable**: no early divergence, smoother loss curves, and steady improvements in validation Penalized BLEU across epochs. The model could train effectively up to ~15 epochs, yielding a modest but consistent performance gain (*16.1% → 21.51%*, +5.5%) compared to fixed LR or linear scheduling and this time penalized bleu score and most importantly validation loss was much more stable.

**Decision:** **Cosine scheduling with warm-up** was adopted as the standard, forming a more reliable backbone for subsequent experiments.

#### Back-Translation Augmentation

**Setup:** With a stable training pipeline in place, we revisited data augmentation using **back-translation**. A portion of English sentences was translated to French and back, creating paraphrased variants. For example:

- Original: `("A man is biking in the park", "A person rides a bicycle through a park")`
- Augmented: `("A man is riding a bike in the park", "A person rides a bicycle through a park")`

These augmented pairs were added to the training data, nearly doubling its size.

**Motivation:** Back-translation provides **natural alternative phrasings**, giving the model more diverse training signals and reducing its tendency to copy the input.

**Outcome:** Validation **Penalized BLEU improved further** (from *Y%* to *Z%*), showing gains over the contrastive + cosine scheduler setup. The model produced more varied paraphrases and was less prone to trivial copying. However, the larger, noisier dataset caused **instability**: BLEU scores fluctuated across epochs instead of rising steadily.

**Decision:** Despite the fluctuations, back-translation proved beneficial overall. We relied on **early stopping** to pick the best epoch, while noting that further refinements to training dynamics would be needed to stabilize progress.

#### Two-Stage Training 

**Change Introduced:** The final improvement was adopting a **two-stage training process**:  

1. **Stage 1 Pre-training:** We first performed a pre-training on an external paraphrase dataset without type labels. For this, we used the **MultiPIT corpus (Multi-Party Paraphrase in Twitter)** [arxiv.org](https://arxiv.org/html/2310.14863v2) a collection of paraphrastic sentence pairs from Twitter across multiple topics. MultiPIT is much larger than ETPC but does not have fine-grained type annotations. We fine-tuned **BART** on MultiPIT paraphrase pairs (for 3 epochs) using a standard seq2seq objective (input → output paraphrase). We also incorporated the contrastive loss in this stage (to encourage dissimilar outputs). We picked 5000 (5k) samples from the data as total were 130k.

2. **Stage 2 Fine-tuning:** We then took this pre-trained model and fine-tuned it on ETPC data with type-controlled inputs, as before. Stage 2 training included all our previous best practices (back-translation augmentation, contrastive loss, scheduler, etc.).

**Motivation:** ETPC, while very rich in annotation, is relatively **limited in size** for fine-tuning a large generative model. By leveraging an **auxiliary paraphrase corpus (MultiPIT)**, we exposed the model to more paraphrase patterns and vocabulary. MultiPIT was chosen because it consists of diverse, high-quality paraphrases mined from Twitter [arxiv.org](https://arxiv.org/html/2310.14863v2).  

This helped the model learn broad paraphrasing skills and avoid overfitting to ETPC’s peculiarities. Essentially, stage 1 acts as a **paraphrase-centric pre-training**, similar to how one might pre-train on a large general corpus before fine-tuning on a niche task. The two-stage approach improves **generalization**: the model enters stage 2 with a strong paraphrasing foundation, needing only to map paraphrase types to actual transformations.

**Outcome:** The two-stage training showed the **highest performance overall**. After final fine-tuning on ETPC, the model achieved a **Penalized BLEU of around [Placeholder]**, representing a substantial improvement over the single-stage baseline.  

- Models initialized from MultiPIT pretraining converged **faster** and reached **better outcomes** on ETPC.  
- Qualitatively, paraphrases generated by the final model were more **fluent** and adhered better to the requested type changes.  
- For example, when asked for a lexical substitution, the model was more likely to choose a suitable synonym instead of trivially reordering words likely due to the richness of MultiPIT’s paraphrase variety.

**Two-Stage Details:**  

- Stage 1: We sampled ~5,000 sentence pairs from MultiPIT and trained for a couple of epochs with a moderate learning rate. Even limited pre-training provided a noticeable boost.  
- Stage 2: Fine-tuning on ETPC was done with a lower LR (`3e-5` or `1e-5`) to avoid forgetting paraphrasing capability gained in stage 1.  
- We also tested contrastive loss usage in stage 1. Applying it consistently from the start yielded slightly better final results, suggesting that enforcing **diversity early** is beneficial.

**Final Model:** Importantly, all prior improvements worked **in unison** in the final setup:  

- Contrastive loss prevented copying.  
- Back-translation increased paraphrase diversity.  
- MultiPIT pretraining gave the model a broad repertoire.  
- Scheduler + early stopping stabilized the process.  

Together, these yielded the **best Penalized BLEU** (~XYZ% vs. baseline’s ~XYZ% *[placeholders]*).

---------------------
## Dataset
### Quora Question Pairs Paraphrase (BERT)
### Semantic Textual Similarity (BERT)
### Stanford Sentiment Treebank (BERT)
### Paraphrase Type Detection (BART)
### Paraphrase Type Generation (BART)

In this section **Data Augmentation** on this task is explained. We explored two strategies to increase training diversity:

- **Synonym Replacement:** About 15% of training samples were augmented using **WordNet** by randomly replacing ~15% of eligible words (nouns, verbs, etc.) with synonyms or hypernyms. This introduced basic lexical variation.
- **Back-Translation:** Using **English → French → English** translation, we generated paraphrased versions of up to **100% of the training set**, effectively doubling the dataset. These paraphrases were added alongside the original pairs.

All augmented examples inherited the original **paraphrase type labels**, assuming the transformation type remained valid (though this may introduce some noise).

**Note:** **Back-translation proved significantly more effective** than synonym replacement and became the primary augmentation used in later stages.


During validation and inference for **decoding**, we generate paraphrases using **beam search** with:

- Beam size of **5**
- **Maximum length of 50 tokens**
- **Early stopping** once all beams finish

This configuration encourages diverse yet accurate outputs.




---------------------
## Hyperparameter Optimization

We performed systematic hyperparameter optimization using grid search and validation set performance. The optimization process focused on finding the best combination of learning rate, dropout probability, and batch size to maximize performance while minimizing overfitting.

### Quora Question Pairs Paraphrase (BERT)

Tests were run with:
- **Hidden Dropout Probabilties** 0.1, 0.2, 0.3, and 0.4
- **Learning Rate** 5e-3, 1e-3, 1e-4, 5e-5, 1e-5, and 5e-6

Highest dev_acc reached (batch size 16, 10 epochs, optimizer AdamW) for each combination tested:
| Hidden Dropout Probability \ Learning Rate | 5e-3 | 1e-3 | 1e-4 | 5e-5 | 1e-5 | 5e-6 |
| --- | --- | --- | --- | --- | --- | --- |
| **0.1** | X | X | X | 0.774 | 0.787 | 0.783 |
| **0.2** | 0.628 | 0.628 | X | 0.787 | 0.790 | 0.779 |
| **0.3** | 0.628 | 0.628 | 0.628 | 0.779 | 0.788 | 0.783 |
| **0.4** | 0.628 | 0.628 | X | 0.782 | **0.792** | 0.786 |


(see `grete_test_full_data.csv` for full results) 

ANTHONY: Is the lr=1e-5, hidden dropout probability 0.4, actually statistically better than lr=1e-5, hdp=0.3? (chi squared test)
Chi squared value of 4.822e-3 with one degree of freedom -> accept null hypothesis (there is no difference) (according to this table: https://passel2.unl.edu/view/lesson/9beaa382bf7e/8) (but that is calculating from percent, not number of observations)


### Semantic Textual Similarity (BERT)



### Stanford Sentiment Treebank (BERT)



### Paraphrase Type Detection (BART)



### Paraphrase Type Generation (BART)

| Hyperparameter | Baseline (ETPC) | +Synonyms | +Contrastive Loss| +Scheduler | +Back-Translation |Two-Stage (MultiPIT → ETPC)|
| ---------------------- | ---- | ---- | ---- | ---- | ---- | ------------------------------------- |
| **Learning Rate (LR)** | 3e-5 | 3e-5 | 3e-5 | 3e-5 | 3e-5 (with scheduler) | Stage 1: 3e-5, Stage 2: 3e-5 |
| **Batch Size**         | 16   | 16   | 16   | 16   | 16   | Stage 1: 32, Stage 2: 16 |
| **Weight Decay**       | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **Contrastive Weight** | –    | –    | 0.35 (stable setting) | 0.35 | 0.35 | Stage 1: 0.2, Stage 2: 0.35 |
| **Augmentation**       | None | Synonym replacement (15%) | None (focus on loss change) | None | Back-translation (~20–25%) | Stage 1: MultiPIT pretrain (5k pairs), Stage 2: ETPC + back-translation |
| **Scheduler**          | None (fixed LR) | None | None  | Cosine decay + 6% warm-up | Cosine decay + 6% warm-up | Cosine decay + 6% warm-up |
| **Epochs (max)**       | 10 (picked the best model) | 10 (picked the best model) | 10 (picked the best model) | Up to 15 (more stable) | Up to 15 (more stable) | Stage 1: 3, Stage 2: up to 15 (early stopped) |
| **Early Stopping**     | -    | -    | Yes (patience=3, Penalized BLEU) | Yes (patience=3, Penalized BLEU) | Yes (patience=3, Penalized BLEU) | Yes (patience=3, Penalized BLEU) |



-----------------------

## Members' Contribution

- **Moujan Mirjalili:** 
1. Implemented the baseline for the Paraphrase Type Detection task.
2. Worked on improvements for the Paraphrase Type Detection task.
3. Assisted with QQP task improvements.
4. Implemented a baseline for the minBERT bonus task.
5. Wrote the README content for the BART Paraphrase Detection and STS tasks.
6. Compiled and completed the overall project README using material provided by team members.

- **Farhan Kayhan:**
1. Implemented the SST baseline.
2. Voluntarily developed the entire multitask BERT classifier as a fallback in case of emergencies, which was ultimately used as the official baseline for the BERT tasks.
3. Worked on improvements for the STS task.
4. Wrote the README content for the SST and STS tasks.
5. Implemented STS improvements again.
6. Unified all BERT tasks BERT.

- **Ali Hamza Bashir:** 
1. Developed the baseline for the Paraphrase Type Generation task.
2. Implemented improvements for the paraphrase generation task. 
3. Wrote Paraphrase Type Generation task README.
4. Continued work on the minBERT bonus implementation.
5. Ran experiments on his personal cluster when the main cluster was unavailable.

- **Skyler Anthony McDonnell:** 
1. Implemented the QQP baseline
2. Implemented improvements for the QQP task.
3. Wrote the QQP README.

- **Hanumanth Padmanabhan:**  
1. Implemented the baseline for the STS task.
2. Wrote a mini baseline README.


-----------------------

## AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./AI-Usage-Card.pdf/) at the top. The card is based on [https://ai-cards.org/](https://ai-cards.org/).

-----------------------

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John Hewitt, Amelie Byun, John Cho, and their team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html), created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

-----------------------

## References

- Vaswani et al., "Attention is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
- minBERT assignment, CMU CS11-711 Advanced NLP (http://phontron.com/class/anlp2021/index.html)
- Facebook BART-large model (https://github.com/facebookresearch/ParlAI/tree/main/projects/bart)
- Datasets: QQP, SST, STS, ETPC (see data/ directory)
