# Clustering Optimization as Blackbox (COB)

This repository provides the implementation of the Clustering Optimization technique as Blackbox (COB) based on [Learn The Big Picture: Representation Learning for Clustering](https://www.cs.unh.edu/~sk1105/papers/repl4nlp_paper.pdf) by [Sumanta Kashyapi](https://www.cs.unh.edu/~sk1105/) and [Laura Dietz](https://www.cs.unh.edu/~dietz/).

## Abstract
Existing supervised models for text clustering find it difficult to directly optimize for clustering results. This is because clustering is a discrete process and it is difficult to estimate meaningful gradient of any discrete function that can drive gradient based optimization algorithms. So, existing supervised clustering algorithms indirectly optimize for some continuous function that approximates the clustering process. We propose a scalable training strategy that directly optimizes for a discrete clustering metric. We train a BERT-based embedding model using our method and evaluate it on two publicly available datasets. We show that our method outperforms another BERT-based embedding model employing Triplet loss and other unsupervised baselines. This suggests that optimizing directly for the clustering outcome indeed yields better representations suitable for clustering.

## Getting started with training COB
1. To train the COB model with default parameters on NG20 dataset, simply run ```python experiments/ng20_run.py```
2. To train using the large train dataset from TRECCAR, run ```python experiments/treccar_run.py```

## Important parameters
- -in: Path to input directory, only used for TRECCAR experiments
- -tin: Prefix of the training files 
- -tp: A tab separated file with raw text for each passage in the following format
  passage_ID1 passage_text1
  passage_ID2 passage_text2
  ...
- -out: Path where the trained model will be saved
- -mn: Name of / Path to the sentence-bert embedding model
- -ls: (Experimental) Choice between spectral clustering loss ('bbspec') or COB loss ('bb')
- -lm: Hyperparameter lambda
- -b: Hyperparameter beta
- -li: Rate of increment of lambda if 'bbinc' is supplied for the -ex option
- -rg: Hyperparameter regularizer
- -np/ -ntp: Max no. of pages to be clustered in each training sample
- -tf: Used as the denominator for the fraction of total no of triplets to be included for training, only used for the triplet loss baseline
- -vs: No of samples used for the validation set
- -bt: Batch size
- -ep: No of training epochs
- -ws: Warmup steps
- -es: No of training steps between two consecutive evaluation on validation set
- --gpu_eval: Whether to use GPU for evaluation
- --balanced: Whether to balance the dataset, used only for a baseline
- -ex: Types of experiments, available choices are bbfix (fixed lambda), bbinc (incremental lambda), trip (triplet loss baseline)
