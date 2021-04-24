from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def cov(X):
    return np.dot(X.T, X) / X.shape[0]

def pca(data, pc_count = None):
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    C = cov(data)
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V

def viz_cluster(page, emb_dict_path):
    emb_dict = np.load(emb_dict_path, allow_pickle=True)
    vecs = emb_dict[()][page]['vec']
    labels = emb_dict[()][page]['label']
    print(vecs.shape)
    pca_embeddings = pca(vecs, 2)[0]
    ax = sns.scatterplot(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], hue=labels, palette='deep')
    plt.show()