from model.BBCluster import CustomSentenceTransformer, OptimCluster, euclid_dist
from experiments.treccar_run import prepare_cluster_data_train_only, prepare_cluster_data2, get_trec_dat, \
    get_paratext_dict
from util.Data import InputTRECCARExample
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Iterable, List
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from tqdm.autonotebook import trange
from clearml import Task
import pickle
import argparse

import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class ContextualScalerClusterModel(nn.Module):

    def __init__(self, path:str=None, query_transformer:CustomSentenceTransformer=None,
                 psg_transformer:CustomSentenceTransformer=None, device:torch.device=None):
        super(ContextualScalerClusterModel, self).__init__()
        self.scaler =
        self.optim = OptimCluster
        self.device = device

    def save(self, path):
        self.query_model.save(path+'/query_model')
        self.psg_model.save(path+'/psg_model')

    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Taken from SentenceTransformers
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                   num_warmup_steps=warmup_steps,
                                                                                   num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def query_batch_collate_fn(self, batch):
        num_texts = len(batch[0].texts)
        queries = []
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            queries.append(example.q_context)
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            labels.append(example.label)

        labels = torch.tensor(labels).to(self.device)

        q_tokenized = self.query_model.tokenize(queries)
        batch_to_device(q_tokenized, self.device)

        psg_features = []
        for idx in range(num_texts):
            p_tokenized = self.psg_model.tokenize(texts[idx])
            batch_to_device(p_tokenized, self.device)
            psg_features.append(p_tokenized)

        return q_tokenized, psg_features, labels

    def forward(self, query_feature: Dict[str, Tensor], passage_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        n = labels.shape[1]

        query_embedding = self.query_model(query_feature)['sentence_embedding']
        # its the scaling vector, so each element in vector should be [0, 1]
        psg_embeddings = torch.stack([self.psg_model(passages)['sentence_embedding']
                                      for passages in passage_features], dim=1)
        scaled_psg_embeddings = torch.tile(query_embedding.unsqueeze(1), (1, n, 1)) * psg_embeddings

        return scaled_psg_embeddings