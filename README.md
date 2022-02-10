# Clustering Optimization as Blackbox (COB)

This repository provides the implementation of the Clustering Optimization technique as Blackbox (COB) based on [Learn The Big Picture: Representation Learning for Clustering](https://www.cs.unh.edu/~sk1105/papers/repl4nlp_paper.pdf) by [Sumanta Kashyapi](https://www.cs.unh.edu/~sk1105/) and [Laura Dietz](https://www.cs.unh.edu/~dietz/).

## Abstract
Existing supervised models for text clustering find it difficult to directly optimize for clustering results. This is because clustering is a discrete process and it is difficult to estimate meaningful gradient of any discrete function that can drive gradient based optimization algorithms. So, existing supervised clustering algorithms indirectly optimize for some continuous function that approximates the clustering process. We propose a scalable training strategy that directly optimizes for a discrete clustering metric. We train a BERT-based embedding model using our method and evaluate it on two publicly available datasets. We show that our method outperforms another BERT-based embedding model employing Triplet loss and other unsupervised baselines. This suggests that optimizing directly for the clustering outcome indeed yields better representations suitable for clustering.

## Getting started
1. To train the COB model with default parameters on NG20 dataset, simply run ```python experiments/ng20_run.py```
