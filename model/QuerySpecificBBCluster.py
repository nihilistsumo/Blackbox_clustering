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

def prepare_cluster_data_train(pages_file, art_qrels, top_qrels, paratext):
    page_paras, rev_para_top, _ = get_trec_dat(art_qrels, top_qrels, None)
    ptext_dict = get_paratext_dict(paratext)
    top_cluster_data = []
    pages = []
    with open(pages_file, 'r') as f:
        for l in f:
            pages.append(l.rstrip('\n'))
    for i in trange(len(pages)):
        page = pages[i]
        paras = page_paras[page]
        paratexts = [ptext_dict[p] for p in paras]
        top_sections = list(set([rev_para_top[p] for p in paras]))
        if len(top_sections) < 2:
            continue
        top_labels = [top_sections.index(rev_para_top[p]) for p in paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        top_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                    label=np.array(top_labels)))
    print('Total data instances: %5d' % len(top_cluster_data))
    return top_cluster_data

def prepare_cluster_data_for_eval(art_qrels, top_qrels, paratext, do_filter, val_samples):
    page_paras, rev_para_top, _ = get_trec_dat(art_qrels, top_qrels, None)
    len_paras = np.array([len(page_paras[page]) for page in page_paras.keys()])
    print('mean paras: %.2f, std: %.2f, max paras: %.2f' % (np.mean(len_paras), np.std(len_paras), np.max(len_paras)))
    ptext_dict = get_paratext_dict(paratext)
    top_cluster_data = []
    pages = list(page_paras.keys())
    skipped_pages = 0
    max_num_doc = max([len(page_paras[p]) for p in page_paras.keys()])
    for i in trange(len(pages)):
        page = pages[i]
        paras = page_paras[page]
        paratexts = [ptext_dict[p] for p in paras]
        top_sections = list(set([rev_para_top[p] for p in paras]))
        top_labels = [top_sections.index(rev_para_top[p]) for p in paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        n = len(paras)
        if do_filter:
            if n < 20 or n > 200:
                skipped_pages += 1
                continue
        paras = paras[:max_num_doc] if n >= max_num_doc else paras + ['dummy'] * (max_num_doc - n)
        paratexts = paratexts[:max_num_doc] if n >= max_num_doc else paratexts + [''] * (max_num_doc - n)
        top_labels = top_labels[:max_num_doc] if n >= max_num_doc else top_labels + [-1] * (max_num_doc - n)
        if do_filter:
            if len(set(top_labels)) < 2 or n / len(set(top_labels)) < 2.5:
                ## the page should have at least 2 top level sections and n/k should be at least 2.5
                skipped_pages += 1
                continue
        top_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                          label=np.array(top_labels)))
    if val_samples > 0:
        top_cluster_data = top_cluster_data[:val_samples]
    print('Total data instances: %5d' % len(top_cluster_data))
    return top_cluster_data

class QuerySpecificClusterModel(nn.Module):

    def __init__(self, path:str=None, query_transformer:CustomSentenceTransformer=None,
                 psg_transformer:CustomSentenceTransformer=None, device:torch.device=None):
        super(QuerySpecificClusterModel, self).__init__()
        if path is not None:
            self.query_model = CustomSentenceTransformer(path+'/query_model')
            self.psg_model = CustomSentenceTransformer(path+'/psg_model')
        else:
            self.query_model = query_transformer
            self.psg_model = psg_transformer
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

class BBClusterLossModel(nn.Module):

    def __init__(self, model: QuerySpecificClusterModel, device, lambda_val: float, reg_const: float):
        super(BBClusterLossModel, self).__init__()
        self.model = model
        self.lambda_val = lambda_val
        self.reg = reg_const
        self.optim = OptimCluster()
        self.device = device

    def true_adj_mat(self, label):
        n = label.numel()
        adj_mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j or label[i] == label[j]:
                    adj_mat[i][j] = 1.0
        return adj_mat

    def forward(self, query_feature: Dict[str, Tensor], passage_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        batch_size = labels.shape[0]
        n = labels.shape[1]
        ks = [torch.unique(labels[i]).numel() for i in range(batch_size)]
        true_adjacency_mats = torch.stack([self.true_adj_mat(labels[i]) for i in range(batch_size)]).to(self.device)

        query_embedding = self.model.query_model(query_feature)['sentence_embedding']
        # its the scaling vector, so each element in vector should be [0, 1]
        psg_embeddings = torch.stack([self.model.psg_model(passages)['sentence_embedding']
                                      for passages in passage_features], dim=1)
        scaled_psg_embeddings = torch.tile(query_embedding.unsqueeze(1), (1, n, 1)) * psg_embeddings

        embeddings_dist_mats = torch.stack([euclid_dist(scaled_psg_embeddings[i]) for i in range(batch_size)])
        mean_similar_dist = (embeddings_dist_mats * true_adjacency_mats).sum() / true_adjacency_mats.sum()
        mean_dissimilar_dist = (embeddings_dist_mats * (1.0 - true_adjacency_mats)).sum() / (
                1 - true_adjacency_mats).sum()
        adjacency_mats = self.optim.apply(embeddings_dist_mats, self.lambda_val, ks).to(self.device)

        err_mats = adjacency_mats * (1.0 - true_adjacency_mats) + (1.0 - adjacency_mats) * true_adjacency_mats
        err_mean = err_mats.mean(dim=0).sum()
        loss = err_mean + self.reg * (mean_similar_dist - mean_dissimilar_dist)

        return loss

class QueryClusterEvaluator(SentenceEvaluator):

    def __init__(self, queries: List[str], passages: List[List[str]], labels: List[Tensor], use_model_device=True):
        self.queries = queries
        self.passages = passages
        self.labels = labels
        self.use_model_device = use_model_device

    @classmethod
    def from_input_examples(cls, examples: List[InputTRECCARExample], use_model_device, **kwargs):
        queries = []
        passages = []
        labels = []
        for example in examples:
            queries.append(example.q_context)
            passages.append(example.texts)
            labels.append(torch.from_numpy(example.label))
        return cls(queries=queries, passages=passages, labels=labels, use_model_device=use_model_device, **kwargs)

    def euclid_dist(self, x):
        dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
        return dist_mat

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        rand_scores, nmi_scores, ami_scores = [], [], []
        model_device = model.device
        if not self.use_model_device:
            model.cpu()
        for i in trange(len(self.queries), desc="Evaluating on val", smoothing=0.05):
            query = self.queries[i]
            passages_to_cluster = [self.passages[i][p] for p in range(len(self.passages[i]))
                                   if len(self.passages[i][p])>0]
            true_label = self.labels[i][:len(passages_to_cluster)]
            query_feature = model.query_model.tokenize(query)
            doc_features = model.psg_model.tokenize(passages_to_cluster)
            if self.use_model_device:
                batch_to_device(doc_features, model_device)
            query_embedding = model.query_model(query_feature)['sentence_embedding']
            psg_embeddings = model.psg_model(doc_features)['sentence_embedding']
            scaled_psg_embeddings = query_embedding * psg_embeddings
            embeddings_dist_mat = self.euclid_dist(scaled_psg_embeddings)
            cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed',
                                         linkage='average')
            cluster_label = cl.fit_predict(embeddings_dist_mat.detach().cpu().numpy())
            rand_scores.append(adjusted_rand_score(true_label.numpy(), cluster_label))
            nmi_scores.append(normalized_mutual_info_score(true_label.numpy(), cluster_label))
            ami_scores.append(adjusted_mutual_info_score(true_label.numpy(), cluster_label))
        mean_rand = np.mean(np.array(rand_scores))
        mean_nmi = np.mean(np.array(nmi_scores))
        mean_ami = np.mean(np.array(ami_scores))
        print("\nRAND: %.5f, NMI: %.5f, AMI: %.5f\n" % (mean_rand, mean_nmi, mean_ami), flush=True)
        if not self.use_model_device:
            model.to(model_device)
        return mean_rand

def train(train_cluster_data, val_cluster_data, test_cluster_data, output_path, eval_steps,
          num_epochs, warmup_frac, lambda_val, reg, beta, loss_name, use_model_device, max_train_size=-1,
          model_name='distilbert-base-uncased', out_features=256, steps_per_epoch=None, weight_decay=0.01,
          optimizer_class=transformers.AdamW, scheduler='WarmupLinear', optimizer_params={'lr':2e-5},
          show_progress_bar=True, max_grad_norm=1, save_best_model=True):
    tensorboard_writer = SummaryWriter('./tensorboard_logs')
    task = Task.init(project_name='Query Specific BB Clustering', task_name='query_bbc_fixed_lambda')
    config_dict = {'lambda_val': lambda_val, 'reg': reg}
    config_dict = task.connect(config_dict)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    query_word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    query_pooling_model = models.Pooling(query_word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    query_dense_model = models.Dense(in_features=query_pooling_model.get_sentence_embedding_dimension(),
                                     out_features=out_features,
                                     activation_function=nn.Sigmoid())
    psg_word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    psg_pooling_model = models.Pooling(psg_word_embedding_model.get_word_embedding_dimension(),
                                         pooling_mode_mean_tokens=True,
                                         pooling_mode_cls_token=False,
                                         pooling_mode_max_tokens=False)

    psg_dense_model = models.Dense(in_features=psg_pooling_model.get_sentence_embedding_dimension(),
                                     out_features=out_features,
                                     activation_function=nn.Tanh())

    query_model = CustomSentenceTransformer(modules=[query_word_embedding_model, query_pooling_model,
                                                     query_dense_model])
    psg_model = CustomSentenceTransformer(modules=[psg_word_embedding_model, psg_pooling_model, psg_dense_model])
    model = QuerySpecificClusterModel(query_transformer=query_model, psg_transformer=psg_model, device=device)

    train_dataloader = DataLoader(train_cluster_data, shuffle=True, batch_size=1)
    evaluator = QueryClusterEvaluator.from_input_examples(val_cluster_data, use_model_device)
    test_evaluator = QueryClusterEvaluator.from_input_examples(test_cluster_data, use_model_device)

    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_frac)  # 10% of train data

    print("Untrained performance")
    model.to(device)
    evaluator(model)

    train_dataloader.collate_fn = model.query_batch_collate_fn

    # Train the model
    best_score = -9999999
    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch = len(train_dataloader)
    num_train_steps = int(steps_per_epoch * num_epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    data_iter = iter(train_dataloader)
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler_obj = model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                        t_total=num_train_steps)
    config = {'epochs': num_epochs, 'steps_per_epoch': steps_per_epoch}
    global_step = 0
    loss_model = BBClusterLossModel(model, device, lambda_val, reg)
    for epoch in trange(config.get('epochs'), desc="Epoch", disable=not show_progress_bar):
        training_steps = 0
        running_loss_0 = 0.0
        model.zero_grad()
        model.train()
        for _ in trange(config.get('steps_per_epoch'), desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                data = next(data_iter)
            query_feature, psg_features, labels = data
            if max_train_size > 0 and labels.shape[1] > max_train_size:
                print('skipping instance with '+str(labels.shape[1])+' passages')
                continue
            loss_val = loss_model(query_feature, psg_features, labels)
            running_loss_0 += loss_val.item()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler_obj.step()
            training_steps += 1
            global_step += 1

            if eval_steps > 0 and training_steps % eval_steps == 0:
                tensorboard_writer.add_scalar('training_loss', running_loss_0 / eval_steps, global_step)
                # logger.report_scalar('Loss', 'training_loss', iteration=global_step, v
                # alue=running_loss_0/evaluation_steps)
                running_loss_0 = 0.0
                # self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)
                if evaluator is not None:
                    score = evaluator(model, output_path=output_path, epoch=epoch, steps=training_steps)
                    tensorboard_writer.add_scalar('val_ARI', score, global_step)
                    # logger.report_scalar('Training progress', 'val_ARI', iteration=global_step, value=score)
                    if score > best_score:
                        best_score = score
                        if save_best_model:
                            print('Saving model at: ' + output_path)
                            model.save(output_path)
                model.zero_grad()
                model.train()
        if evaluator is not None:
            score = evaluator(model, output_path=output_path, epoch=epoch, steps=training_steps)
            tensorboard_writer.add_scalar('val_ARI', score, global_step)
            # logger.report_scalar('Training progress', 'val_ARI', iteration=global_step, value=score)
            if score > best_score:
                best_score = score
                if save_best_model:
                    model.save(output_path)
        if test_evaluator is not None:
            best_model = QuerySpecificClusterModel(output_path)
            if torch.cuda.is_available():
                model.to(torch.device('cpu'))
                best_model.to(device)
                test_ari = test_evaluator(best_model)
                best_model.to(torch.device('cpu'))
                model.to(device)
            else:
                test_ari = test_evaluator(best_model)
            tensorboard_writer.add_scalar('test_ARI', test_ari, global_step)
            # logger.report_scalar('Training progress', 'test_ARI', iteration=global_step, value=test_ari)
    if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
        model.save(output_path)

def save_sqst_dataset(train_pages_file, art_qrels, top_qrels, paratext, val_samples, outdir):
    page_paras, rev_para_top, _ = get_trec_dat(art_qrels, top_qrels, None)
    ptext_dict = get_paratext_dict(paratext)
    train_cluster_data = []
    test_cluster_data = []
    pages = []
    with open(train_pages_file, 'r') as f:
        for l in f:
            pages.append(l.rstrip('\n'))
    for i in trange(len(pages)):
        page = pages[i]
        paras = page_paras[page]
        page_sec_para_dict = {}
        for p in paras:
            sec = rev_para_top[p]
            if sec not in page_sec_para_dict.keys():
                page_sec_para_dict[sec] = [p]
            else:
                page_sec_para_dict[sec].append(p)
        sections = list(set([rev_para_top[p] for p in paras]))
        train_paras = []
        test_paras = []
        for s in page_sec_para_dict.keys():
            test_paras += page_sec_para_dict[s][:len(page_sec_para_dict[s])//2]
            train_paras += page_sec_para_dict[s][len(page_sec_para_dict[s])//2:]
        test_labels = [sections.index(rev_para_top[p]) for p in test_paras]
        train_labels = [sections.index(rev_para_top[p]) for p in train_paras]
        test_paratexts = [ptext_dict[p] for p in test_paras]
        train_paratexts = [ptext_dict[p] for p in train_paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        test_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=test_paras,
                                                     texts=test_paratexts, label=np.array(test_labels)))
        train_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=train_paras,
                                                     texts=train_paratexts, label=np.array(train_labels)))
    random.shuffle(test_cluster_data)
    val_cluster_data = test_cluster_data[:val_samples]
    test_cluster_data = test_cluster_data[val_samples:]
    with open(outdir + '/sqst_treccar_train.pkl', 'wb') as f:
        pickle.dump(train_cluster_data, f)
    with open(outdir + '/sqst_treccar_val.pkl', 'wb') as f:
        pickle.dump(val_cluster_data, f)
    with open(outdir + '/sqst_treccar_test.pkl', 'wb') as f:
        pickle.dump(test_cluster_data, f)
    print('No. of data instances - Train: %5d, Val: %5d, Test: %5d' % (len(train_cluster_data), len(val_cluster_data),
                                                                       len(test_cluster_data)))

def save_squt_dataset(train_pages_file, art_qrels, top_qrels, paratext, val_samples, outdir):
    page_paras, rev_para_top, _ = get_trec_dat(art_qrels, top_qrels, None)
    ptext_dict = get_paratext_dict(paratext)
    train_cluster_data = []
    test_cluster_data = []
    pages = []
    with open(train_pages_file, 'r') as f:
        for l in f:
            pages.append(l.rstrip('\n'))
    for i in trange(len(pages)):
        page = pages[i]
        paras = page_paras[page]
        page_sec_para_dict = {}
        for p in paras:
            sec = rev_para_top[p]
            if sec not in page_sec_para_dict.keys():
                page_sec_para_dict[sec] = [p]
            else:
                page_sec_para_dict[sec].append(p)
        sections = list(set([rev_para_top[p] for p in paras]))
        random.shuffle(sections)
        test_sections, train_sections = sections[:len(sections)//2], sections[len(sections)//2:]
        train_paras = []
        test_paras = []
        for s in test_sections:
            test_paras += page_sec_para_dict[s]
        for s in train_sections:
            train_paras += page_sec_para_dict[s]
        test_labels = [sections.index(rev_para_top[p]) for p in test_paras]
        train_labels = [sections.index(rev_para_top[p]) for p in train_paras]
        test_paratexts = [ptext_dict[p] for p in test_paras]
        train_paratexts = [ptext_dict[p] for p in train_paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        test_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=test_paras,
                                                     texts=test_paratexts, label=np.array(test_labels)))
        train_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=train_paras,
                                                      texts=train_paratexts, label=np.array(train_labels)))
    random.shuffle(test_cluster_data)
    val_cluster_data = test_cluster_data[:val_samples]
    test_cluster_data = test_cluster_data[val_samples:]
    with open(outdir + '/squt_treccar_train.pkl', 'wb') as f:
        pickle.dump(train_cluster_data, f)
    with open(outdir + '/squt_treccar_val.pkl', 'wb') as f:
        pickle.dump(val_cluster_data, f)
    with open(outdir + '/squt_treccar_test.pkl', 'wb') as f:
        pickle.dump(test_cluster_data, f)
    print(
        'No. of data instances - Train: %5d, Val: %5d, Test: %5d' % (len(train_cluster_data), len(val_cluster_data),
                                                                     len(test_cluster_data)))


def main():
    parser = argparse.ArgumentParser(description='Run treccar experiments')
    parser.add_argument('-in', '--input_dir', default='/home/sk1105/sumanta/trec_dataset/train')
    parser.add_argument('-out', '--output_model_path', default='/home/sk1105/sumanta/bb_cluster_models/temp_model')
    parser.add_argument('-mn', '--model_name', default='distilbert-base-uncased')
    parser.add_argument('-ls', '--loss', default='bb')
    parser.add_argument('-lm', '--lambda_val', type=float, default=200.0)
    parser.add_argument('-b', '--beta', type=float, default=10.0)
    parser.add_argument('-rg', '--reg_const', type=float, default=2.5)
    parser.add_argument('-ep', '--num_epoch', type=int, default=3)
    parser.add_argument('-ws', '--warmup', type=float, default=0.1)
    parser.add_argument('-es', '--eval_steps', type=int, default=100)
    parser.add_argument('-md', '--max_sample_size', type=int, default=-1)
    parser.add_argument('-ext', '--exp_type', default='sqst')
    parser.add_argument('--gpu_eval', default=False, action='store_true')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_path = args.output_model_path
    model_name = args.model_name
    loss_name = args.loss
    lambda_val = args.lambda_val
    beta = args.beta
    reg = args.reg_const
    epochs = args.num_epoch
    warmup_fraction = args.warmup
    eval_steps = args.eval_steps
    max_sample_size = args.max_sample_size
    exp_type = args.exp_type
    gpu_eval = args.gpu_eval

    if exp_type == 'sqst':
        with open(input_dir + '/sqst/sqst_treccar_train.pkl', 'rb') as f:
            train_cluster_data = pickle.load(f)
        with open(input_dir + '/sqst/sqst_treccar_val.pkl', 'rb') as f:
            val_cluster_data = pickle.load(f)
        with open(input_dir + '/sqst/sqst_treccar_test.pkl', 'rb') as f:
            test_cluster_data = pickle.load(f)
    elif exp_type == 'squt':
        with open(input_dir + '/squt/squt_treccar_train.pkl', 'rb') as f:
            train_cluster_data = pickle.load(f)
        with open(input_dir + '/squt/squt_treccar_val.pkl', 'rb') as f:
            val_cluster_data = pickle.load(f)
        with open(input_dir + '/squt/squt_treccar_test.pkl', 'rb') as f:
            test_cluster_data = pickle.load(f)

    print('Data loaded, starting to train')

    train(train_cluster_data, val_cluster_data, test_cluster_data, output_path, eval_steps, epochs, warmup_fraction,
          lambda_val, reg, beta, loss_name, gpu_eval, max_train_size=max_sample_size, model_name=model_name)

if __name__ == '__main__':
    main()