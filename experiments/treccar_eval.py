import argparse
import pickle
import torch
from model.BBCluster import CustomSentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.treccar_run import prepare_cluster_data2
from tqdm.autonotebook import trange
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster._supervised import check_clusterings
import numpy as np
from scipy.stats import ttest_rel
from sentence_transformers import models
import torch.nn as nn


def euclid_dist(x):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    return dist_mat

# from sklearn 0.24.2 source
def pair_confusion_matrix(labels_true, labels_pred):

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = contingency_matrix(
        labels_true, labels_pred, sparse=True, dtype=np.int64
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
    return C

# from sklearn 0.24.2 source
def rand_score(labels_true, labels_pred):

    contingency = pair_confusion_matrix(labels_true, labels_pred)
    numerator = contingency.diagonal().sum()
    denominator = contingency.sum()

    if numerator == denominator or denominator == 0:
        return 1.0

    return numerator / denominator

def get_eval_scores(model, cluster_data, anchor_rand=None, anchor_nmi=None, anchor_ami=None, anchor_urand=None):
    rand_scores, nmi_scores, ami_scores, urand_scores = {}, {}, {}, {}
    pages, passages, labels = [], [], []
    for sample in cluster_data:
        pages.append(sample.qid)
        passages.append(sample.texts)
        labels.append(torch.from_numpy(sample.label))
    for i in trange(len(passages), desc="Evaluating", smoothing=0.05):
        passages_to_cluster = [passages[i][p] for p in range(len(passages[i])) if len(passages[i][p]) > 0]
        true_label = labels[i][:len(passages_to_cluster)]
        doc_features = model.tokenize(passages_to_cluster)
        doc_embeddings = model(doc_features)['sentence_embedding']
        embeddings_dist_mat = euclid_dist(doc_embeddings)
        cl = AgglomerativeClustering(n_clusters=torch.unique(true_label).numel(), affinity='precomputed',
                                     linkage='average')
        cluster_label = cl.fit_predict(embeddings_dist_mat.detach().cpu().numpy())
        rand_scores[pages[i]] = adjusted_rand_score(true_label.numpy(), cluster_label)
        nmi_scores[pages[i]] = normalized_mutual_info_score(true_label.numpy(), cluster_label)
        ami_scores[pages[i]] = adjusted_mutual_info_score(true_label.numpy(), cluster_label)
        urand_scores[pages[i]] = rand_score(true_label.numpy(), cluster_label)
    print('Page\t\tAdj RAND\t\tNMI\t\tAMI\t\tUnadj RAND')
    for p in rand_scores.keys():
        print(p+'\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' % (rand_scores[p], nmi_scores[p], ami_scores[p], urand_scores[p]))
    rand_arr, nmi_arr, ami_arr, urand_arr = (np.array(list(rand_scores.values())), np.array(list(nmi_scores.values())),
                                    np.array(list(ami_scores.values())), np.array(list(urand_scores.values())))
    mean_rand, mean_nmi, mean_ami, mean_urand = (np.mean(rand_arr), np.mean(nmi_arr), np.mean(ami_arr), np.mean(urand_arr))
    if anchor_rand is not None:
        rand_ttest, nmi_ttest, ami_ttest, urand_ttest = (ttest_rel(anchor_rand, rand_arr), ttest_rel(anchor_nmi, nmi_arr),
                                            ttest_rel(anchor_ami, ami_arr), ttest_rel(anchor_urand, urand_arr))
        print('mean ARI: %.4f (%.4f), mean NMI: %.4f (%.4f), mean AMI: %.4f (%.4f), mean UARI: %.4f (%.4f)' % (mean_rand, rand_ttest[1], mean_nmi,
                                                                                       nmi_ttest[1], mean_ami, ami_ttest[1], mean_urand, urand_ttest[1]))
    else:
        print('mean ARI: %.4f, mean NMI: %.4f, mean AMI: %.4f, mean UARI: %.4f' % (mean_rand, mean_nmi, mean_ami, mean_urand))
    return rand_arr, nmi_arr, ami_arr, urand_arr

parser = argparse.ArgumentParser(description='Eval treccar experiments')
parser.add_argument('-ip', '--input_dir', default='~/trec_dataset')
parser.add_argument('-rm', '--raw_model_name', default='distilbert-base-uncased')
parser.add_argument('-lv', '--level', default='top')
parser.add_argument('-mp', '--model_paths', nargs='+')
args = parser.parse_args()
input_dir = args.input_dir
model_name = args.raw_model_name
level = args.level
model_paths = args.model_paths

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
tfidf = TfidfVectorizer()
rand_scores_tf, nmi_scores_tf, ami_scores_tf, urand_scores_tf = [], [], [], []
for input_exmp in test_cluster_data:
    n = len(input_exmp.pids) - input_exmp.pids.count('dummy')
    labels = input_exmp.label[:n]
    corpus = input_exmp.texts[:n]
    vecs = tfidf.fit_transform(corpus).toarray()
    cl = AgglomerativeClustering(n_clusters=len(set(labels)), linkage='average')
    cl_labels = cl.fit_predict(vecs)
    rand_scores_tf.append(adjusted_rand_score(labels, cl_labels))
    nmi_scores_tf.append(normalized_mutual_info_score(labels, cl_labels))
    ami_scores_tf.append(adjusted_mutual_info_score(labels, cl_labels))
    urand_scores_tf.append(rand_score(labels, cl_labels))
mean_rand_tf = np.mean(np.array(rand_scores_tf))
mean_nmi_tf = np.mean(np.array(nmi_scores_tf))
mean_ami_tf = np.mean(np.array(ami_scores_tf))
mean_urand_tf = np.mean(np.array(urand_scores_tf))
print('TFIDF')
print("\nRAND: %.5f, NMI: %.5f, AMI: %.5f, URAND: %.5f\n" % (mean_rand_tf, mean_nmi_tf, mean_ami_tf, mean_urand_tf), flush=True)

word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
doc_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                                   activation_function=nn.Tanh())

raw_model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])

anchor_rand, anchor_nmi, anchor_ami, anchor_urand = [], [], [], []
for i in range(len(model_paths)):
    mp = model_paths[i]
    m = CustomSentenceTransformer(mp)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    if i == 0:
        print('This is the anchor model for paired ttest')
        anchor_rand, anchor_nmi, anchor_ami, anchor_urand = get_eval_scores(m, test_cluster_data)
    else:
        mean_rand, mean_nmi, mean_ami, mean_urand = get_eval_scores(m, test_cluster_data, anchor_rand, anchor_nmi, anchor_ami, anchor_urand)

mean_rand, mean_nmi, mean_ami, mean_urand = get_eval_scores(raw_model, test_cluster_data, anchor_rand, anchor_nmi, anchor_ami, anchor_urand)

rand_ttest_tf, nmi_ttest_tf, ami_ttest_tf, urand_ttest_tf = (ttest_rel(anchor_rand, rand_scores_tf), ttest_rel(anchor_nmi, nmi_scores_tf),
                                            ttest_rel(anchor_ami, ami_scores_tf), ttest_rel(anchor_urand, urand_scores_tf))
print('\nTFIDF ttest pval ARI: %.5f, NMI: %.5f, AMI: %.5f, UARI: %.5f' % (rand_ttest_tf[1], nmi_ttest_tf[1], ami_ttest_tf[1], urand_ttest_tf[1]))