import pandas as pd
from mistralai.client import MistralClient
import os
from dotenv import load_dotenv
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import pairwise_distances
import numpy as np

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    distances = pairwise_distances(X)
    delta = []
    big_delta = []
    
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            mask_i = labels == unique_clusters[i]
            mask_j = labels == unique_clusters[j]
            inter_dist = distances[np.ix_(mask_i, mask_j)]
            delta.append(np.min(inter_dist))
    
    for k in unique_clusters:
        mask_k = labels == k
        intra_dist = distances[np.ix_(mask_k, mask_k)]
        if intra_dist.size > 1:
            big_delta.append(np.max(intra_dist))
    
    return np.min(delta) / np.max(big_delta)

df = pd.read_csv('reports.csv')

client = MistralClient(api_key=MISTRAL_API_KEY)

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings(model="mistral-embed", input=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

df["embeddings"] = get_embeddings_by_chunks(df["Report"].tolist(), 10)

embeddings = np.array(df['embeddings'].to_list())
predicted_classes = np.array(df['Predicted Class'].to_list())

tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(df['embeddings'].to_list()))
ax = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=np.array(df['Predicted Class'].to_list()))
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))

silhouette_avg = silhouette_score(embeddings, predicted_classes)
davies_bouldin = davies_bouldin_score(embeddings, predicted_classes)
calinski_harabasz = calinski_harabasz_score(embeddings, predicted_classes)
dunn = dunn_index(embeddings, predicted_classes)