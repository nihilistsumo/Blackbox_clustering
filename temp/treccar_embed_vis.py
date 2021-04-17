import argparse
import pickle
import torch
from model.BBCluster import CustomSentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.treccar_run import prepare_cluster_data2
from tqdm.autonotebook import trange
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
from scipy.stats import ttest_rel

def euclid_dist(x):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    return dist_mat

parser = argparse.ArgumentParser(description='Eval treccar experiments')
parser.add_argument('-ip', '--input_dir', default='/home/sk1105/sumanta/trec_dataset')
parser.add_argument('-lv', '--level', default='top')
parser.add_argument('-pg', '--page_title')
parser.add_argument('-mp', '--model_path')
args = parser.parse_args()
input_dir = args.input_dir
level = args.level
page = args.page_title
model_path = args.model_path
model = CustomSentenceTransformer(model_path)

test_art_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels'
test_top_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels'
test_hier_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels'
test_paratext = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv'
test_top_cluster_data, test_hier_cluster_data = prepare_cluster_data2(test_art_qrels, test_top_qrels, test_hier_qrels,
                                                                        test_paratext, False, -1, 0)
if level == 'top':
    test_cluster_data = test_top_cluster_data
else:
    test_cluster_data = test_hier_cluster_data
for sample in test_cluster_data:
    if sample.qid == page:
        passages_to_cluster = [sample.texts[p] for p in range(len(sample.texts)) if len(sample.texts[p]) > 0]
        true_label = torch.from_numpy(sample.label[:len(passages_to_cluster)])
        doc_features = model.tokenize(passages_to_cluster)
        doc_embeddings = model(doc_features)['sentence_embedding']
        embeddings_dist_mat = euclid_dist(doc_embeddings)
        cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed',
                                     linkage='average')
        cluster_label = cl.fit_predict(embeddings_dist_mat.detach().cpu().numpy())
        rand_score = adjusted_rand_score(true_label.numpy(), cluster_label)