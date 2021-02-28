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
from model.BBCluster import BBClusterLossModel
import argparse

def get_trec_dat(art_qrels, top_qrels, hier_qrels):
    page_paras = {}
    art_qrels_lines = []
    with open(art_qrels, 'r') as f:
        art_qrels_lines = f.readlines()
    random.shuffle(art_qrels_lines)
    for l in art_qrels_lines:
        page = l.split(' ')[0]
        para = l.split(' ')[2]
        if page not in page_paras.keys():
            page_paras[page] = [para]
        else:
            page_paras[page].append(para)

    rev_para_top = {}
    with open(top_qrels, 'r') as f:
        for l in f:
            topic = l.split(' ')[0]
            para = l.split(' ')[2]
            rev_para_top[para] = topic

    rev_para_hier = {}
    with open(hier_qrels, 'r') as f:
        for l in f:
            topic = l.split(' ')[0]
            para = l.split(' ')[2]
            rev_para_hier[para] = topic

    return page_paras, rev_para_top, rev_para_hier

def get_paratext_dict(paratext_file):
    paratext_dict = {}
    with open(paratext_file, 'r') as pt:
        for l in pt:
            paratext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()
    return paratext_dict

def prepare_cluster_data(train_art_qrels, train_top_qrels, train_hier_qrels, train_paratext,
                         test_art_qrels, test_top_qrels, test_hier_qrels, test_paratext, max_num_doc=75, val_samples=50):
    train_page_paras, train_rev_para_top, train_rev_para_hier = get_trec_dat(train_art_qrels, train_top_qrels,
                                                                             train_hier_qrels)
    test_page_paras, test_rev_para_top, test_rev_para_hier = get_trec_dat(test_art_qrels, test_top_qrels,
                                                                          test_hier_qrels)
    train_len_paras = np.array([len(train_page_paras[page]) for page in train_page_paras.keys()])
    test_len_paras = np.array([len(test_page_paras[page]) for page in test_page_paras.keys()])
    print('train mean paras: %.2f, std: %.2f, max paras: %.2f, test mean paras: %.2f, std: %.2f, max paras: %.2f' %
          (np.mean(train_len_paras), np.std(train_len_paras), np.max(train_len_paras), np.mean(test_len_paras),
           np.std(test_len_paras), np.max(test_len_paras)))
    train_ptext_dict = get_paratext_dict(train_paratext)
    test_ptext_dict = get_paratext_dict(test_paratext)
    train_top_cluster_data = []
    train_hier_cluster_data = []
    train_pages = list(train_page_paras.keys())
    skipped_pages = 0
    for i in trange(len(train_pages)):
        page = train_pages[i]
        paras = train_page_paras[page]
        paratexts = [train_ptext_dict[p] for p in paras]
        top_sections = list(set([train_rev_para_top[p] for p in paras]))
        top_labels = [top_sections.index(train_rev_para_top[p]) for p in paras]
        hier_sections = list(set([train_rev_para_hier[p] for p in paras]))
        hier_labels = [hier_sections.index(train_rev_para_hier[p]) for p in paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        n = len(paras)
        if n < 20:  ## at least 20 passages should be contained in the page
            skipped_pages += 1
            continue
        paras = paras[:max_num_doc] if n >= max_num_doc else paras + ['dummy'] * (max_num_doc - n)
        paratexts = paratexts[:max_num_doc] if n >= max_num_doc else paratexts + [''] * (max_num_doc - n)
        top_labels = top_labels[:max_num_doc] if n >= max_num_doc else top_labels + [-1] * (max_num_doc - n)
        hier_labels = hier_labels[:max_num_doc] if n >= max_num_doc else hier_labels + [-1] * (max_num_doc - n)
        if len(set(top_labels)) < 2:  ## the page should have at least 2 top level sections
            skipped_pages += 1
            continue
        train_top_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                          label=np.array(top_labels)))
        train_hier_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                           label=np.array(hier_labels)))
    val_top_cluster_data = train_top_cluster_data[-val_samples:]
    train_top_cluster_data = train_top_cluster_data[:-val_samples]
    val_hier_cluster_data = train_hier_cluster_data[-val_samples:]
    train_hier_cluster_data = train_hier_cluster_data[:-val_samples]
    test_top_cluster_data = []
    test_hier_cluster_data = []
    max_num_doc_test = max([len(test_page_paras[p]) for p in test_page_paras.keys()])
    test_pages = list(test_page_paras.keys())
    for i in trange(len(test_pages)):
        page = test_pages[i]
        paras = test_page_paras[page]
        paratexts = [test_ptext_dict[p] for p in paras]
        top_sections = list(set([test_rev_para_top[p] for p in paras]))
        top_labels = [top_sections.index(test_rev_para_top[p]) for p in paras]
        hier_sections = list(set([test_rev_para_hier[p] for p in paras]))
        hier_labels = [hier_sections.index(test_rev_para_hier[p]) for p in paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        n = len(paras)
        paras = paras + ['dummy'] * (max_num_doc_test - n)
        paratexts = paratexts + [''] * (max_num_doc_test - n)
        top_labels = top_labels + [-1] * (max_num_doc_test - n)
        hier_labels = hier_labels + [-1] * (max_num_doc_test - n)
        test_top_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                         label=np.array(top_labels)))
        test_hier_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                          label=np.array(hier_labels)))
    print("Top-level datasets")
    print("Train instances: %5d" % len(train_top_cluster_data))
    print("Val instances: %5d" % len(val_top_cluster_data))
    print("Test instances: %5d" % len(test_top_cluster_data))
    print("Skipped pages: %5d" % skipped_pages)

    return train_top_cluster_data, train_hier_cluster_data, val_top_cluster_data, val_hier_cluster_data, \
           test_top_cluster_data, test_hier_cluster_data

def get_triples(cluster_data, max_triples_per_page=25):
    all25_triples = []
    for c in trange(len(cluster_data)):
        text = cluster_data[c].texts
        t = list(cluster_data[c].label)
        triples = []
        page_done = False
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
                        if len(triples) >= max_triples_per_page:
                            page_done = True
                            break
                if page_done:
                    break
            if page_done:
                break
        all25_triples += triples
    return all25_triples

def prepare_triples_data(train_cluster_data, val_cluster_data):

    # 25 triples from each page
    train_all25_triples = get_triples(train_cluster_data)
    val_all25_triples = get_triples(val_cluster_data)
    pprint('No of train triples: %2d' % len(train_all25_triples))
    pprint('No of val triples: %2d' % len(val_all25_triples))

    return train_all25_triples, val_all25_triples

def run_fixed_lambda_bbcluster(train_cluster_data, val_cluster_data, output_path, train_batch_size, lambda_val=200.0,
                               model_name='distilbert-base-uncased', num_epochs=1, out_features=256,
                               eval_steps=20):
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

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    loss_model = BBClusterLossModel(model=model, device=device, lambda_val=lambda_val)
    # reg_loss_model = ClusterDistLossModel(model=model)

    train_dataloader = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    # train_dataloader2 = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    evaluator = ClusterEvaluator.from_input_examples(val_cluster_data)

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

    print("Raw BERT embedding performance")
    evaluator(model, output_path)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, loss_model)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=eval_steps,
              warmup_steps=warmup_steps,
              output_path=output_path)

def run_triplets_model(train_triplets, val_triplets, output_path, train_batch_size, model_name='distilbert-base-uncased',
                       num_epochs=1, out_features=256, eval_steps=20):
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

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])

    train_dataloader = DataLoader(train_triplets, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = ClusterEvaluator.from_input_examples(val_triplets)

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

    print("Raw BERT embedding performance")
    evaluator(model, output_path)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=eval_steps,
              warmup_steps=warmup_steps,
              output_path=output_path)

def main():
    parser = argparse.ArgumentParser(description='Run treccar experiments')
    parser.add_argument('-in', '--input_dir', default='/home/sk1105/sumanta/trec_dataset')
    parser.add_argument('-tin', '--train_input', default='train/base.train.cbor')
    parser.add_argument('-tp', '--train_paratext', default='train/train_paratext/train_paratext.tsv')
    parser.add_argument('-out', '--output_model_path', default='/home/sk1105/sumanta/bb_cluster_models')
    parser.add_argument('-md', '--max_doc', type=int, default=50)
    parser.add_argument('-vs', '--val_samples', type=int, default=25)
    parser.add_argument('-bt', '--batch_size', type=int, default=1)
    args = parser.parse_args()
    input_dir = args.input_dir
    train_in = args.train_input
    train_pt = args.train_paratext
    output_path = args.output_model_path
    max_num_doc = args.max_doc
    val_samples = args.val_samples
    batch_size = args.batch_size
    train_art_qrels = input_dir + '/' + train_in + '-article.qrels'
    train_top_qrels = input_dir + '/' + train_in + '-toplevel.qrels'
    train_hier_qrels = input_dir + '/' + train_in + '-hierarchical.qrels'
    train_paratext = input_dir + '/' + train_pt
    test_art_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels'
    test_top_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels'
    test_hier_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels'
    test_paratext = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv'

    train_top_cluster_data, train_hier_cluster_data, val_top_cluster_data, val_hier_cluster_data, \
    test_top_cluster_data, test_hier_cluster_data = prepare_cluster_data(train_art_qrels, train_top_qrels, train_hier_qrels,
                                                                         train_paratext, test_art_qrels, test_top_qrels,
                                                                         test_hier_qrels, test_paratext, max_num_doc,
                                                                         val_samples)
    run_fixed_lambda_bbcluster(train_top_cluster_data, val_top_cluster_data, output_path, batch_size)

if __name__ == '__main__':
    main()