from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def viz_cluster(page, emb_dict_path):
    emb_dict = np.load(emb_dict_path, allow_pickle=True)
    vecs = emb_dict[()][page]['vec']
    labels = emb_dict[()][page]['label']
    print(vecs.shape)
    pca = PCA(n_components=2)
    doc_embeddings_np_scaled = StandardScaler().fit_transform(vecs)
    pca_embeddings = pca.fit_transform(doc_embeddings_np_scaled)
    ax = sns.scatterplot(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], hue=labels, palette='deep')
    plt.show()