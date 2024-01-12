from sklearn.manifold import TSNE
import numpy as np
import matplotlib
from cycler import cycler
from sklearn.datasets import fetch_openml
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_mnist():
    print("Downloading MNIST dataset...")
    X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    return X, y

def plot(Y, classes, name):
    digits = set(classes)
    fig = plt.figure()
    colormap = plt.cm.spectral
    plt.gca().set_prop_cycle(
        cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 10)]))
    ax = fig.add_subplot(111)
    labels = []
    for d in digits:
        idx = classes == d
        ax.plot(Y[idx, 0], Y[idx, 1], 'o')
        labels.append(d)
    ax.legend(labels, numpoints=1, fancybox=True)
    fig.savefig(name)

################################################################


mnist, classes = get_mnist()

tsne = TSNE(n_components=2, random_state=0, n_iter = 1000, min_grad_norm=0, verbose=1000)

time1 = time.time()
mnist_tsne = tsne.fit_transform(mnist)
time2 = time.time()
print 'function took %0.3f ms' % ((time2-time1)*1000.0)

plot(mnist_tsne, classes, 'tsne_sk_core.png')
