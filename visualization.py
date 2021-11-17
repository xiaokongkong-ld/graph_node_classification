import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy

def visualize(out, color):
    z = TSNE(n_components=2).fit_transform(out.detach().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

