import argparse
import pickle
import torch
from model.BBCluster import CustomSentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.treccar_run import prepare_cluster_data2
from tqdm.autonotebook import trange
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
import json

def euclid_dist(x):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    return dist_mat

parser = argparse.ArgumentParser(description='Eval treccar experiments')
parser.add_argument('-ip', '--input_dir', default='~/trec_dataset')
parser.add_argument('-lv', '--level', default='top')
parser.add_argument('-pg', '--page_title')
parser.add_argument('-mp', '--model_path')
parser.add_argument('-out', '--outdict')
args = parser.parse_args()
input_dir = args.input_dir
level = args.level
page = args.page_title
model_path = args.model_path
outpath = args.outdict
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
emb_dict = {}
for sample in test_cluster_data:
    print(sample.qid)
    passages_to_cluster = [sample.texts[p] for p in range(len(sample.texts)) if len(sample.texts[p]) > 0]
    true_label = torch.from_numpy(sample.label[:len(passages_to_cluster)])
    doc_features = model.tokenize(passages_to_cluster)
    doc_embeddings = model(doc_features)['sentence_embedding']
    doc_embeddings_np = doc_embeddings.detach().numpy()
    embeddings_dist_mat = euclid_dist(doc_embeddings)
    cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed',
                                 linkage='average')
    cluster_label = cl.fit_predict(embeddings_dist_mat.detach().cpu().numpy())
    rand_score = adjusted_rand_score(true_label.numpy(), cluster_label)
    if sample.qid == page:
        ordered_true_topic_indices = {}
        for i in range(len(true_label)):
            topic = true_label[i].item()
            if topic in ordered_true_topic_indices.keys():
                ordered_true_topic_indices[topic].append(i)
            else:
                ordered_true_topic_indices[topic] = [i]
        ordered_emb = []
        for topic in ordered_true_topic_indices.keys():
            for p in ordered_true_topic_indices[topic]:
                ordered_emb.append(doc_embeddings_np[p])
            ordered_emb.append(np.zeros(doc_embeddings_np[0].shape))
        ordered_emb = np.array(ordered_emb)
        ax = sns.heatmap(ordered_emb, robust=True)
        plt.show()
    '''
    print(doc_embeddings_np.shape)
    pca = PCA(n_components=2)
    doc_embeddings_np_scaled = StandardScaler().fit_transform(doc_embeddings_np)
    pca_embeddings = pca.fit_transform(doc_embeddings_np_scaled)
    ax2 = sns.scatterplot(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], hue=true_label, palette='deep')
    plt.show()
    '''
    emb_dict[sample.qid] = {'vec': doc_embeddings_np, 'label': true_label}
if outpath is not None:
    np.save(outpath, emb_dict)