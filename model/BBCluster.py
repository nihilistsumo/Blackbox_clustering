import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch import Tensor
from typing import Iterable, Dict, Tuple, Type, Callable
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import os
from tqdm.autonotebook import trange

class CustomSentenceTransformer(SentenceTransformer):

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            test_evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        tensorboard_writer = SummaryWriter('./tensorboard_logs')
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False

        config = {'epochs': epochs, 'steps_per_epoch': steps_per_epoch}
        for epoch in trange(config.get('epochs'), desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            running_loss_0 = 0.0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(config.get('steps_per_epoch'), desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        if train_idx == 0:
                            running_loss_0 += loss_value.item()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        if train_idx == 0:
                            running_loss_0 += loss_value.item()
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    tensorboard_writer.add_scalar('training_loss', running_loss_0/evaluation_steps, global_step)
                    running_loss_0 = 0.0
                    #self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)
                    if evaluator is not None:
                        score = evaluator(self, output_path=output_path, epoch=epoch, steps=training_steps)
                        tensorboard_writer.add_scalar('val_ARI', score, global_step)
                        if callback is not None:
                            callback(score, epoch, training_steps)
                        if score > self.best_score:
                            self.best_score = score
                            if save_best_model:
                                self.save(output_path)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
            if test_evaluator is not None:
                test_ari = self.evaluate(test_evaluator)
                tensorboard_writer.add_scalar('test_ARI', test_ari, epoch)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(output_path)

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

        #p = torch.sum(true_adjacency_mats, dim=(1,2)) - n
        #adjacency_wt_mats = torch.stack([(1.0 - true_adjacency_mats[i])*p[i]/(n*(n-1)) + true_adjacency_mats[i]*(1.0-p[i]/(n*(n-1))) for i in range(batch_size)])

        #weighted_err_mats = adjacency_wt_mats * (adjacency_mats * (1.0 - true_adjacency_mats) + (1.0 - adjacency_mats) * true_adjacency_mats)
        weighted_err_mats = adjacency_mats * (1.0 - true_adjacency_mats) + (1.0 - adjacency_mats) * true_adjacency_mats
        weighted_err_mean = weighted_err_mats.mean(dim=0).sum()

        #pprint('Weighted err mat mean: %.5f, mean similar dist: %.5f, mean dissimilar dist: %.5f, reg value: %.5f' %
        #       (weighted_err_mean, mean_similar_dist, mean_dissimilar_dist, 20*(mean_similar_dist/mean_dissimilar_dist)))

        loss = weighted_err_mean + self.reg*(mean_similar_dist - mean_dissimilar_dist)
        #loss = weighted_err_mean
        #pprint('Loss: %.5f' % loss.item())
        #print('Loss: '+str(loss.device))
        return loss

####################################################
# More experiments required for spectral clustering
# with proper similarity metrics and affinity matrix
####################################################

def spectral_clustering(batch_pairscore_matrix, num_clusters):
    batch_adjacency_matrix = np.zeros(batch_pairscore_matrix.shape)
    num_batch = batch_pairscore_matrix.shape[0]
    clustering_labels = []
    '''
    for i in range(num_batch):
        print('pairscore matrix shape: ' + str(batch_pairscore_matrix[i].shape))
    '''
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