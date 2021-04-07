import argparse
import pickle
import torch
from model.BBCluster import CustomSentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from tqdm.autonotebook import trange
import numpy as np

def euclid_dist(x):
    dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
    return dist_mat

def get_eval_scores(model, cluster_data):
    rand_scores, nmi_scores, ami_scores = {}, {}, {}
    pages = list(np.arange(len(cluster_data)))
    passages, labels = [], [], []
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
    print('mean ARI: %.4f, mean NMI: %.4f, mean ARI: %.4f' % (np.mean(np.array(rand_scores.values())),
                                                              np.mean(np.array(nmi_scores.values())),
                                                              np.mean(np.array(ami_scores.values()))))
    return rand_scores, nmi_scores, ami_scores

parser = argparse.ArgumentParser(description='Eval 20 news groups experiments')
parser.add_argument('-tp', '--test_data')
parser.add_argument('-mp', '--model_paths', nargs='+')
args = parser.parse_args()
test_data_path = args.test_data
model_paths = args.model_paths

with open(test_data_path, 'rb') as f:
    test_cluster_data = pickle.load(f)

tfidf = TfidfVectorizer()
rand_scores, nmi_scores, ami_scores = [], [], []
for input_exmp in test_cluster_data:
    labels = input_exmp.label
    corpus = input_exmp.texts
    vecs = tfidf.fit_transform(corpus).toarray()
    cl = AgglomerativeClustering(n_clusters=len(set(labels)), linkage='average')
    cl_labels = cl.fit_predict(vecs)
    rand_scores.append(adjusted_rand_score(labels, cl_labels))
    nmi_scores.append(normalized_mutual_info_score(labels, cl_labels))
    ami_scores.append(adjusted_mutual_info_score(labels, cl_labels))
mean_rand = np.mean(np.array(rand_scores))
mean_nmi = np.mean(np.array(nmi_scores))
mean_ami = np.mean(np.array(ami_scores))
print("\nRAND: %.5f, NMI: %.5f, AMI: %.5f\n" % (mean_rand, mean_nmi, mean_ami), flush=True)

for mp in model_paths:
    m = CustomSentenceTransformer(mp)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    _ = get_eval_scores(m, test_cluster_data)