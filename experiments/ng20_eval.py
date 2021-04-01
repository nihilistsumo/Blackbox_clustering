import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
from util.Evaluator import ClusterEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
import numpy as np

parser = argparse.ArgumentParser(description='Eval 20 news groups experiments')
parser.add_argument('-tp', '--test_data')
parser.add_argument('-mp', '--model_paths', nargs='+')
args = parser.parse_args()
test_data_path = args.test_data
model_paths = args.model_paths

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: ' + str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: ' + str(device))

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

test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)
for mp in model_paths:
    m = SentenceTransformer(mp)
    m.to(device)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    m.evaluate(test_evaluator)