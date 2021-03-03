from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Dict
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

def euclid_dist(x):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    return dist_mat

def euclid_affinity(x, beta):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    affinity_mat = torch.exp(-beta*dist_mat/torch.std(dist_mat, unbiased=False))
    return affinity_mat

def clustering(batch_pairscore_matrix, num_clusters):
    batch_adjacency_matrix = np.zeros(batch_pairscore_matrix.shape)
    num_batch = batch_pairscore_matrix.shape[0]
    clustering_labels = []
    for i in range(num_batch):
        cl = AgglomerativeClustering(n_clusters=num_clusters[i], affinity='precomputed', linkage='average')
        cluster_label = cl.fit_predict(batch_pairscore_matrix[i])
        #clustering_labels.append(torch.from_numpy(cluster_label))
        clustering_labels.append(cluster_label)
        for m in range(cluster_label.shape[0]):
            for n in range(cluster_label.shape[0]):
                if cluster_label[m] == cluster_label[n]:
                    batch_adjacency_matrix[i][m][n] = 1.0
    return batch_adjacency_matrix, clustering_labels

class OptimCluster(torch.autograd.Function):

    @staticmethod
    def forward(ctx, batch_pairscore_matrix, lambda_val, num_clusters):
        ctx.lambda_val = lambda_val
        ctx.num_clusters = num_clusters
        ctx.batch_pairscore_matrix = batch_pairscore_matrix.detach().cpu().numpy()
        ctx.batch_adj_matrix, _ = clustering(ctx.batch_pairscore_matrix, ctx.num_clusters)
        return torch.from_numpy(ctx.batch_adj_matrix).float().to(batch_pairscore_matrix.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_numpy = grad_output.detach().cpu().numpy()
        batch_pairscore_matrix_prime = np.maximum(ctx.batch_pairscore_matrix + ctx.lambda_val * grad_output_numpy, 0.0)
        better_batch_adj_matrix, _ = clustering(batch_pairscore_matrix_prime, ctx.num_clusters)
        gradient = -(ctx.batch_adj_matrix - better_batch_adj_matrix) / ctx.lambda_val
        return torch.from_numpy(gradient.astype(np.float32)).to(grad_output.device), None, None

def spectral_clustering(batch_pairscore_matrix, num_clusters):
    batch_adjacency_matrix = np.zeros(batch_pairscore_matrix.shape)
    num_batch = batch_pairscore_matrix.shape[0]
    clustering_labels = []
    for i in range(num_batch):
        print('pairscore matrix shape: ' + str(batch_pairscore_matrix[i].shape))
    for i in range(num_batch):
        cl = SpectralClustering(n_clusters=num_clusters[i], affinity='precomputed')
        cluster_label = cl.fit_predict(batch_pairscore_matrix[i])
        #clustering_labels.append(torch.from_numpy(cluster_label))
        clustering_labels.append(cluster_label)
        for m in range(cluster_label.shape[0]):
            for n in range(cluster_label.shape[0]):
                if cluster_label[m] == cluster_label[n]:
                    batch_adjacency_matrix[i][m][n] = 1.0
    return batch_adjacency_matrix, clustering_labels

class OptimSpectralCluster(torch.autograd.Function):

    @staticmethod
    def forward(ctx, batch_pairscore_matrix, lambda_val, num_clusters):
        ctx.lambda_val = lambda_val
        ctx.num_clusters = num_clusters
        ctx.batch_pairscore_matrix = batch_pairscore_matrix.detach().cpu().numpy()
        ctx.batch_adj_matrix, _ = spectral_clustering(ctx.batch_pairscore_matrix, ctx.num_clusters)
        return torch.from_numpy(ctx.batch_adj_matrix).float().to(batch_pairscore_matrix.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_numpy = grad_output.detach().cpu().numpy()
        batch_pairscore_matrix_prime = np.maximum(ctx.batch_pairscore_matrix + ctx.lambda_val * grad_output_numpy, 0.0)
        better_batch_adj_matrix, _ = spectral_clustering(batch_pairscore_matrix_prime, ctx.num_clusters)
        gradient = -(ctx.batch_adj_matrix - better_batch_adj_matrix) / ctx.lambda_val
        return torch.from_numpy(gradient.astype(np.float32)).to(grad_output.device), None, None

class BBClusterLossModel(nn.Module):

    def __init__(self, model: SentenceTransformer, device, lambda_val: float, reg_const: float):

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

    def forward(self, passage_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        # passage_features will be a list of feature dicts -> [all 1st passages in batch, all 2nd passages in batch, ...]
        # labels shape: batch X maxpsg
        batch_size = labels.shape[0]
        n = labels.shape[1]
        ks = [torch.unique(labels[i]).numel() for i in range(batch_size)]
        true_adjacency_mats = torch.stack([self.true_adj_mat(labels[i]) for i in range(batch_size)]).to(self.device)

        # embeddings shape: batch X maxpsg X emb
        embeddings = torch.stack([self.model(passages)['sentence_embedding'] for passages in passage_features], dim=1)
        embeddings_dist_mats = torch.stack([euclid_dist(embeddings[i]) for i in range(batch_size)])
        #print('embedding dist mats device: '+str(embeddings_dist_mats.device))
        #print('true adj device: '+str(true_adjacency_mats.device))
        mean_similar_dist = (embeddings_dist_mats * true_adjacency_mats).sum() / true_adjacency_mats.sum()
        mean_dissimilar_dist = (embeddings_dist_mats * (1.0 - true_adjacency_mats)).sum() / (
                    1 - true_adjacency_mats).sum()
        adjacency_mats = self.optim.apply(embeddings_dist_mats, self.lambda_val, ks).to(self.device)

        p = torch.sum(true_adjacency_mats, dim=(1,2)) - n
        adjacency_wt_mats = torch.stack([(1.0 - true_adjacency_mats[i])*p[i]/(n*(n-1)) +
                                         true_adjacency_mats[i]*(1.0-p[i]/(n*(n-1))) for i in range(batch_size)])

        weighted_err_mats = adjacency_wt_mats * (adjacency_mats * (1.0 - true_adjacency_mats) + (1.0 - adjacency_mats) * true_adjacency_mats)
        weighted_err_mean = weighted_err_mats.mean(dim=0).sum()

        #pprint('Weighted err mat mean: %.5f, mean similar dist: %.5f, mean dissimilar dist: %.5f, reg value: %.5f' %
        #       (weighted_err_mean, mean_similar_dist, mean_dissimilar_dist, 20*(mean_similar_dist/mean_dissimilar_dist)))

        loss = weighted_err_mean + self.reg*(mean_similar_dist - mean_dissimilar_dist)
        #loss = 20*(mean_similar_dist/mean_dissimilar_dist)
        #pprint('Loss: %.5f' % loss.item())
        #print('Loss: '+str(loss.device))
        return loss

class BBSpectralClusterLossModel(nn.Module):

    def __init__(self, model: SentenceTransformer, device, lambda_val: float, reg_const: float, beta: float):

        super(BBSpectralClusterLossModel, self).__init__()
        self.model = model
        self.lambda_val = lambda_val
        self.reg = reg_const
        self.beta = beta
        self.optim = OptimSpectralCluster()
        self.device = device

    def true_adj_mat(self, label):
        n = label.numel()
        adj_mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j or label[i] == label[j]:
                    adj_mat[i][j] = 1.0
        return adj_mat

    def forward(self, passage_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        # passage_features will be a list of feature dicts -> [all 1st passages in batch, all 2nd passages in batch, ...]
        # labels shape: batch X maxpsg
        batch_size = labels.shape[0]
        n = labels.shape[1]
        ks = [torch.unique(labels[i]).numel() for i in range(batch_size)]
        true_adjacency_mats = torch.stack([self.true_adj_mat(labels[i]) for i in range(batch_size)]).to(self.device)

        # embeddings shape: batch X maxpsg X emb
        embeddings = torch.stack([self.model(passages)['sentence_embedding'] for passages in passage_features], dim=1)
        embeddings_affinity_mats = torch.stack([euclid_affinity(embeddings[i], self.beta) for i in range(batch_size)])

        mean_similar_affinity = (embeddings_affinity_mats * true_adjacency_mats).sum() / true_adjacency_mats.sum()
        mean_dissimilar_affinity = (embeddings_affinity_mats * (1.0 - true_adjacency_mats)).sum() / (
                    1 - true_adjacency_mats).sum()
        adjacency_mats = self.optim.apply(embeddings_affinity_mats, self.lambda_val, ks).to(self.device)

        p = torch.sum(true_adjacency_mats, dim=(1,2)) - n
        adjacency_wt_mats = torch.stack([(1.0 - true_adjacency_mats[i])*p[i]/(n*(n-1)) +
                                         true_adjacency_mats[i]*(1.0-p[i]/(n*(n-1))) for i in range(batch_size)])

        weighted_err_mats = adjacency_wt_mats * (adjacency_mats * (1.0 - true_adjacency_mats) + (1.0 - adjacency_mats) * true_adjacency_mats)
        weighted_err_mean = weighted_err_mats.mean(dim=0).sum()

        #pprint('Weighted err mat mean: %.5f, mean similar dist: %.5f, mean dissimilar dist: %.5f, reg value: %.5f' %
        #       (weighted_err_mean, mean_similar_dist, mean_dissimilar_dist, 20*(mean_similar_dist/mean_dissimilar_dist)))

        loss = weighted_err_mean + self.reg*(mean_dissimilar_affinity - mean_similar_affinity)
        #loss = 20*(mean_similar_dist/mean_dissimilar_dist)
        #pprint('Loss: %.5f' % loss.item())
        #print('Loss: '+str(loss.device))
        return loss