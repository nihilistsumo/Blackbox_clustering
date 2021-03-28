from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.util import batch_to_device
import torch
from torch import Tensor
import numpy as np
from typing import List
from tqdm.autonotebook import trange
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

class ClusterEvaluator(SentenceEvaluator):

    def __init__(self, passages: List[List[str]], labels: List[Tensor]):
        self.passages = passages
        self.labels = labels

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        passages = []
        labels = []
        for example in examples:
            passages.append(example.texts)
            labels.append(torch.from_numpy(example.label))
        return cls(passages=passages, labels=labels, **kwargs)

    def euclid_dist(self, x):
        dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
        return dist_mat

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        rand_scores, nmi_scores = [], []
        model_device = model.device
        #if next(model.parameters()).is_cuda:
        #    model.cpu()
        for i in trange(len(self.passages), desc="Evaluating on val", smoothing=0.05):
            passages_to_cluster = [self.passages[i][p] for p in range(len(self.passages[i])) if len(self.passages[i][p])>0]
            true_label = self.labels[i][:len(passages_to_cluster)]
            doc_features = model.tokenize(passages_to_cluster)
            batch_to_device(doc_features, model_device)
            doc_embeddings = model(doc_features)['sentence_embedding']
            embeddings_dist_mat = self.euclid_dist(doc_embeddings)
            cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed', linkage='average')
            cluster_label = cl.fit_predict(embeddings_dist_mat.detach().cpu().numpy())
            rand_scores.append(adjusted_rand_score(true_label.numpy(), cluster_label))
            nmi_scores.append(adjusted_mutual_info_score(true_label.numpy(), cluster_label))
        mean_rand = np.mean(np.array(rand_scores))
        mean_nmi = np.mean(np.array(nmi_scores))
        print("\nRAND: %.5f, NMI: %.5f\n" % (mean_rand, mean_nmi), flush=True)
        #if torch.cuda.is_available():
        #    model.to(model_device)
        return mean_rand