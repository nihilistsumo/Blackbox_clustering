import argparse
import pickle
import torch
from model.BBCluster import CustomSentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from tqdm.autonotebook import trange
import numpy as np
from scipy.stats import ttest_rel

def euclid_dist(x):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    return dist_mat

def get_eval_scores(model, cluster_data, anchor_rand=None, anchor_nmi=None, anchor_ami=None):
    rand_scores, nmi_scores, ami_scores = {}, {}, {}
    pages = list(np.arange(len(cluster_data)))
    passages, labels = [], []
    for sample in cluster_data:
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
        rand_scores['Page'+str(pages[i])] = adjusted_rand_score(true_label.numpy(), cluster_label)
        nmi_scores['Page'+str(pages[i])] = normalized_mutual_info_score(true_label.numpy(), cluster_label)
        ami_scores['Page'+str(pages[i])] = adjusted_mutual_info_score(true_label.numpy(), cluster_label)
    print('Page\t\tAdj RAND\t\tNMI\t\tAMI')
    for p in rand_scores.keys():
        print(p+'\t\t%.4f\t\t%.4f\t\t%.4f' % (rand_scores[p], nmi_scores[p], ami_scores[p]))
    rand_arr, nmi_arr, ami_arr = (np.array(list(rand_scores.values())), np.array(list(nmi_scores.values())),
                                    np.array(list(ami_scores.values())))
    mean_rand, mean_nmi, mean_ami = (np.mean(rand_arr), np.mean(nmi_arr), np.mean(ami_arr))
    if anchor_rand is not None:
        rand_ttest, nmi_ttest, ami_ttest = (ttest_rel(anchor_rand, rand_arr), ttest_rel(anchor_nmi, nmi_arr),
                                            ttest_rel(anchor_ami, ami_arr))
        print('mean ARI: %.4f (%.4f), mean NMI: %.4f (%.4f), mean AMI: %.4f (%.4f)' % (mean_rand, rand_ttest, mean_nmi,
                                                                                       nmi_ttest, mean_ami, ami_ttest))
    else:
        print('mean ARI: %.4f, mean NMI: %.4f, mean AMI: %.4f' % (mean_rand, mean_nmi, mean_ami))
    return rand_arr, nmi_arr, ami_arr

parser = argparse.ArgumentParser(description='Eval 20 news groups experiments')
parser.add_argument('-tp', '--test_data')
parser.add_argument('-mp', '--model_paths', nargs='+')
args = parser.parse_args()
test_data_path = args.test_data
model_paths = args.model_paths

with open(test_data_path, 'rb') as f:
    test_cluster_data = pickle.load(f)

tfidf = TfidfVectorizer()
rand_scores_tf, nmi_scores_tf, ami_scores_tf = [], [], []
for input_exmp in test_cluster_data:
    labels = input_exmp.label
    corpus = input_exmp.texts
    vecs = tfidf.fit_transform(corpus).toarray()
    cl = AgglomerativeClustering(n_clusters=len(set(labels)), linkage='average')
    cl_labels = cl.fit_predict(vecs)
    rand_scores_tf.append(adjusted_rand_score(labels, cl_labels))
    nmi_scores_tf.append(normalized_mutual_info_score(labels, cl_labels))
    ami_scores_tf.append(adjusted_mutual_info_score(labels, cl_labels))
mean_rand_tf = np.mean(np.array(rand_scores_tf))
mean_nmi_tf = np.mean(np.array(nmi_scores_tf))
mean_ami_tf = np.mean(np.array(ami_scores_tf))
print("\nRAND: %.5f, NMI: %.5f, AMI: %.5f\n" % (mean_rand_tf, mean_nmi_tf, mean_ami_tf), flush=True)

anchor_rand, anchor_nmi, anchor_ami = [], [], []
for i in range(len(model_paths)):
    mp = model_paths[i]
    m = CustomSentenceTransformer(mp)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    if i == 0:
        print('This is the anchor model for paired ttest')
        anchor_rand, anchor_nmi, anchor_ami = get_eval_scores(m, test_cluster_data)
    else:
        rand_scores, nmi_scores, ami_scores = get_eval_scores(m, test_cluster_data, anchor_rand, anchor_nmi, anchor_ami)