from model.BBCluster import CustomSentenceTransformer, OptimCluster, euclid_dist
from util.Data import InputTRECCARExample
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Iterable, List
from sentence_transformers import models
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from tqdm.autonotebook import trange
from clearml import Task
import os

class QuerySpecificClusterModel(nn.Module):

    def __init__(self, query_transformer: CustomSentenceTransformer, psg_transformer: CustomSentenceTransformer,
                 lambda_val: float, reg: float, device):
        super(QuerySpecificClusterModel, self).__init__()
        self.query_model = query_transformer
        self.psg_model = psg_transformer
        self.optim = OptimCluster
        self.lambda_val = lambda_val
        self.reg = reg
        self.device = device

    def true_adj_mat(self, label):
        n = label.numel()
        adj_mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j or label[i] == label[j]:
                    adj_mat[i][j] = 1.0
        return adj_mat

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

        labels = torch.tensor(labels).to(self._target_device)

        query_features, psg_features = [], []
        q_tokenized = self.query_model.tokenize(queries)
        batch_to_device(q_tokenized, self._target_device)
        for idx in range(num_texts):
            p_tokenized = self.psg_model.tokenize(texts[idx])
            batch_to_device(p_tokenized, self._target_device)
            psg_features.append(p_tokenized)

        return query_features, psg_features, labels

    def forward(self, query_feature: Dict[str, Tensor], passage_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        batch_size = labels.shape[0]
        n = labels.shape[1]
        ks = [torch.unique(labels[i]).numel() for i in range(batch_size)]
        true_adjacency_mats = torch.stack([self.true_adj_mat(labels[i]) for i in range(batch_size)]).to(self.device)

        query_embedding = self.query_model(query_feature)['sentence_embedding'] # its the scaling vector, so each element in vector should be [0, 1]
        psg_embeddings = torch.stack([self.psg_model(passages)['sentence_embedding'] for passages in passage_features], dim=1)
        scaled_psg_embeddings = query_embedding * psg_embeddings

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
            passages_to_cluster = [self.passages[i][p] for p in range(len(self.passages[i])) if len(self.passages[i][p])>0]
            true_label = self.labels[i][:len(passages_to_cluster)]
            query_feature = model.query_model.tokenize(query)
            doc_features = model.psg_model.tokenize(passages_to_cluster)
            if self.use_model_device:
                batch_to_device(doc_features, model_device)
            query_embedding = model.query_model(query_feature)['sentence_embedding']
            psg_embeddings = model.psg_model(doc_features)['sentence_embedding']
            scaled_psg_embeddings = query_embedding * psg_embeddings
            embeddings_dist_mat = self.euclid_dist(scaled_psg_embeddings)
            cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed', linkage='average')
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

def train(train_cluster_data, val_cluster_data, test_cluster_data, output_path, train_batch_size, eval_steps,
                               num_epochs, warmup_frac, lambda_val, reg, beta, loss_name, use_model_device, model_name='distilbert-base-uncased', out_features=256):
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
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    query_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features,
                                     activation_function=nn.Sigmoid())
    psg_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features,
                                   activation_function=nn.Tanh())

    query_model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, query_dense_model])
    psg_model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, psg_dense_model])
    model = QuerySpecificClusterModel(query_transformer=query_model, psg_transformer=psg_model,
                                                         lambda_val=lambda_val, reg=reg, device=device)

    train_dataloader = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    evaluator = QueryClusterEvaluator.from_input_examples(val_cluster_data, use_model_device)
    test_evaluator = QueryClusterEvaluator.from_input_examples(test_cluster_data, use_model_device)

    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_frac)  # 10% of train data

    print("Untrained performance")
    model.to(device)
    evaluator(model)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    train_dataloader.collate_fn = model.query_batch_collate_fn

    # Train the model
