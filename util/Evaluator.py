from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.readers import InputExample
import torch
from torch import Tensor
import numpy as np
from typing import List
from tqdm.autonotebook import trange
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

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
        rand_scores = []
        for i in trange(len(self.passages), desc="Evaluating on test", smoothing=0.05):
            print('Model device: '+str(model.device))
            print('Labels device: '+str(self.labels.device))
            passages_to_cluster = [p for p in self.passages[i] if len(p)>0]
            true_label = self.labels[i][:len(passages_to_cluster)]
            doc_features = model.tokenize(passages_to_cluster)
            doc_embeddings = model(doc_features)['sentence_embedding']
            embeddings_dist_mat = self.euclid_dist(doc_embeddings)
            cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed', linkage='average')
            cluster_label = cl.fit_predict(embeddings_dist_mat.detach().numpy())
            rand_scores.append(adjusted_rand_score(true_label.numpy(), cluster_label))
        mean_rand = np.mean(np.array(rand_scores))
        print("RAND: %.5f" % mean_rand, flush=True)
        return mean_rand