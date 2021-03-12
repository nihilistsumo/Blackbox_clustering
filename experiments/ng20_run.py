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
from model.BBCluster import BBClusterLossModel, BBSpectralClusterLossModel
from experiments.train_model import run_triplets_model, run_fixed_lambda_bbcluster, run_incremental_lambda_bbcluster
from experiments.eval_model import evaluate_ng20
import argparse
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def prepare_cluster_data(pages_to_cluster, val_samples):
    ng_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    ng_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    print(ng_train.target_names)

    ng_train.keys()
    train_cluster_data = []
    test_cluster_data = []
    for i in range(len(ng_train['filenames']) // pages_to_cluster):
        train_cluster_data.append(InputExample(texts=ng_train['data'][i * pages_to_cluster: (i + 1) * pages_to_cluster],
                                               label=ng_train['target'][
                                                     i * pages_to_cluster: (i + 1) * pages_to_cluster]))
    val_cluster_data = train_cluster_data[-val_samples:]
    train_cluster_data = train_cluster_data[:-val_samples]
    for i in range(len(ng_test['filenames']) // pages_to_cluster):
        test_cluster_data.append(InputExample(texts=ng_test['data'][i * pages_to_cluster: (i + 1) * pages_to_cluster],
                                              label=ng_test['target'][
                                                    i * pages_to_cluster: (i + 1) * pages_to_cluster]))
    print("Train instances: %5d" % len(train_cluster_data))
    print("Val instances: %5d" % len(val_cluster_data))
    print("Test instances: %5d" % len(test_cluster_data))

    return train_cluster_data, val_cluster_data, test_cluster_data

def get_frac_triples(cluster_data, num_triples_frac):
    frac_triples = []
    for c in trange(len(cluster_data)):
        text = cluster_data[c].texts
        t = list(cluster_data[c].label)
        triples = []
        for i in range(len(t) - 2):
            for j in range(i + 1, len(t) - 1):
                for k in range(i + 2, len(t)):
                    if len(set([t[i], t[j], t[k]])) == 2:
                        if t[i] == t[j]:
                            triples.append(InputExample(texts=[text[i], text[j], text[k]], label=0))
                        elif t[j] == t[k]:
                            triples.append(InputExample(texts=[text[j], text[k], text[i]], label=0))
                        else:
                            triples.append(InputExample(texts=[text[i], text[k], text[j]], label=0))
        frac_triples += random.sample(triples, len(triples) // num_triples_frac)
    print('No of train triples: %2d' % len(frac_triples))

    return frac_triples

def main():
    parser = argparse.ArgumentParser(description='Run 20 news groups experiments')
    parser.add_argument('-out', '--output_model_path', default='/home/sk1105/sumanta/bb_cluster_models/temp_model')
    parser.add_argument('-ls', '--loss', default='bb')
    parser.add_argument('-lm', '--lambda_val', type=float, default=200.0)
    parser.add_argument('-b', '--beta', type=float, default=10.0)
    parser.add_argument('-li', '--lambda_inc', type=float, default=10.0)
    parser.add_argument('-rg', '--reg_const', type=float, default=2.5)
    parser.add_argument('-np', '--num_pages', type=int, default=50)
    parser.add_argument('-tf', '--triple_fraction', type=int, default=25)
    parser.add_argument('-vs', '--val_samples', type=int, default=25)
    parser.add_argument('-bt', '--batch_size', type=int, default=1)
    parser.add_argument('-ep', '--num_epoch', type=int, default=1)
    parser.add_argument('-ws', '--warmup', type=float, default=0.1)
    parser.add_argument('-es', '--eval_steps', type=int, default=100)
    parser.add_argument('-ex', '--exp_type', default='bbfix')
    args = parser.parse_args()
    output_path = args.output_model_path
    loss_name = args.loss
    lambda_val = args.lambda_val
    beta = args.beta
    lambda_increment = args.lambda_inc
    reg = args.reg_const
    num_pages = args.num_pages
    triple_frac = args.triple_fraction
    val_samples = args.val_samples
    batch_size = args.batch_size
    epochs = args.num_epoch
    warmup_fraction = args.warmup
    eval_steps = args.eval_steps
    experiment_type = args.exp_type

    print('Preparing cluster data')

    train_cluster_data, val_cluster_data, test_cluster_data = prepare_cluster_data(num_pages, val_samples)

    if experiment_type == 'bbfix':
        run_fixed_lambda_bbcluster(train_cluster_data, val_cluster_data, output_path, batch_size, eval_steps, epochs, warmup_fraction,
                               lambda_val, reg, beta, loss_name)
    elif experiment_type == 'bbinc':
        run_incremental_lambda_bbcluster(train_cluster_data, val_cluster_data, output_path, batch_size, eval_steps, epochs, warmup_fraction,
                               lambda_val, lambda_increment, reg)
    elif experiment_type == 'trip':
        train_triples = get_frac_triples(train_cluster_data, triple_frac)
        run_triplets_model(train_triples, val_cluster_data, output_path, batch_size, eval_steps, epochs, warmup_fraction)
    evaluate_ng20(output_path, test_cluster_data)

if __name__ == '__main__':
    main()