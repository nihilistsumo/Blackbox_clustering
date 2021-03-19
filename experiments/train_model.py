from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator, SentenceEvaluator
from tqdm.autonotebook import trange
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Type, Iterable, Union
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_20newsgroups
import transformers
import math
from pprint import pprint
import random
from collections import Counter
from util.Data import InputTRECCARExample
from util.Evaluator import ClusterEvaluator
from model.BBCluster import BBClusterLossModel, BBSpectralClusterLossModel, CustomSentenceTransformer
import argparse
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

from clearml import Task

def run_fixed_lambda_bbcluster(train_cluster_data, val_cluster_data, test_cluster_data, output_path, train_batch_size, eval_steps,
                               num_epochs, warmup_frac, lambda_val, reg, beta, loss_name, model_name='distilbert-base-uncased', out_features=256):
    task = Task.init(project_name='BB Clustering', task_name='bbclustering_fixed_lambda')
    config_dict = {'lambda_val': lambda_val, 'reg': reg}
    config_dict = task.connect(config_dict)
    if torch.cuda.is_available():
        print('CUDA is available')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    doc_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features,
                                   activation_function=nn.Tanh())

    model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if loss_name == 'bbspec':
        loss_model = BBSpectralClusterLossModel(model=model, device=device,
                                                lambda_val=config_dict.get('lambda_val', lambda_val),
                                                reg_const=config_dict.get('reg', reg), beta=beta)
    else:
        loss_model = BBClusterLossModel(model=model, device=device,
                                        lambda_val=config_dict.get('lambda_val', lambda_val),
                                        reg_const=config_dict.get('reg', reg))
    # reg_loss_model = ClusterDistLossModel(model=model)

    train_dataloader = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    # train_dataloader2 = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    evaluator = ClusterEvaluator.from_input_examples(val_cluster_data)
    test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)

    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_frac)  # 10% of train data

    print("Raw BERT embedding performance")
    model.to(device)
    evaluator(model, output_path)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, loss_model)],
              evaluator=evaluator,
              test_evaluator=test_evaluator,
              epochs=num_epochs,
              evaluation_steps=eval_steps,
              warmup_steps=warmup_steps,
              output_path=output_path)

def run_incremental_lambda_bbcluster(train_cluster_data, val_cluster_data, test_cluster_data, output_path, train_batch_size, eval_steps,
                               num_epochs, warmup_frac, lambda_val, lambda_increment, reg, model_name='distilbert-base-uncased', out_features=256):
    task = Task.init(project_name='BB Clustering', task_name='bbclustering_inc_lambda')
    config_dict = {'lambda_val': lambda_val, 'reg': reg}
    config_dict = task.connect(config_dict)
    if torch.cuda.is_available():
        print('CUDA is available')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    doc_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features,
                                   activation_function=nn.Tanh())

    model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    #loss_model = BBClusterLossModel(model=model, device=device, lambda_val=lambda_val, reg_const=reg)
    # reg_loss_model = ClusterDistLossModel(model=model)

    train_dataloader = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    # train_dataloader2 = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    evaluator = ClusterEvaluator.from_input_examples(val_cluster_data)
    test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)

    per_lambda_num_epochs = 1
    warmup_steps = int(len(train_dataloader) * per_lambda_num_epochs * warmup_frac)  # 10% of train data

    print("Raw BERT embedding performance")
    model.to(device)
    evaluator(model, output_path)

    # Train the model
    for e in range(num_epochs):
        lambda_val_curr = lambda_val + lambda_increment * e
        loss_model = BBClusterLossModel(model=model, device=device, lambda_val=lambda_val_curr, reg_const=reg)
        model.fit(train_objectives=[(train_dataloader, loss_model)],
                  evaluator=evaluator,
                  test_evaluator=test_evaluator,
                  epochs=per_lambda_num_epochs,
                  evaluation_steps=eval_steps,
                  warmup_steps=warmup_steps,
                  output_path=output_path)
        print('Epoch: %3d, lambda: %.2f' % (e, lambda_val_curr))

def run_triplets_model(train_triplets, val_cluster_data, test_cluster_data, output_path, train_batch_size, eval_steps, num_epochs, warmup_frac,
                       model_name='distilbert-base-uncased', out_features=256):
    task = Task.init(project_name='BB Clustering', task_name='bbclustering_triplets')
    if torch.cuda.is_available():
        print('CUDA is available')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    doc_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features,
                                   activation_function=nn.Tanh())

    model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])

    train_dataloader = DataLoader(train_triplets, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = ClusterEvaluator.from_input_examples(val_cluster_data)
    test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)

    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_frac)  # 10% of train data

    print("Raw BERT embedding performance")
    model.to(device)
    evaluator(model, output_path)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              test_evaluator=test_evaluator,
              epochs=num_epochs,
              evaluation_steps=eval_steps,
              warmup_steps=warmup_steps,
              output_path=output_path)