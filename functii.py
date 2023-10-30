import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import linkage, dendrogram
from seaborn import scatterplot,kdeplot
from sklearn.cluster._agglomerative import AgglomerativeClustering


def calcul_ierarhie(x, metoda="complete"):
    distanta = linkage(x, method=metoda)  # calculez distanta cu fct linkage
    nr_jonctiuni = distanta.shape[0]
    # print("nr jonct partitie " , nr_jonctiuni)
    k_max = np.argmax(distanta[1:, 1] - distanta[:(nr_jonctiuni - 1), 1])  # k_max e un indice
    k = nr_jonctiuni - k_max
    return distanta, k


def plot_ierarhie(distanta, etichete, threshold, titlu="Plot ierarhie"):
    # desenam dendograma
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    dendrogram(distanta, labels=etichete, color_threshold=threshold, ax=ax)


def calcul_partitie(x, k, metoda="complete"):
    hclust = AgglomerativeClustering(k, linkage=metoda, compute_distances=True)
    hclust.fit(x)  # fit-> implementeaza modelul
    distante = hclust.distances_
    nr_jonctiuni = len(distante)
    # j-jonctiune care da partitia curenta de claster
    j = nr_jonctiuni - k

    threshold = (distante[j] + distante[j + 1]) / 2  # threshold se calculeaza la jumatatea distantei
    coduri = hclust.labels_  # avem cod pt fiecare cluster
    # pt cluster vom intoarce etichete, si nu coduri
    return np.array(["c" + str(cod + 1) for cod in coduri]), threshold


def histograme(z, p, variabila):
    fig = plt.figure(figsize=(9, 6))
    # clase  are etichete cluster
    fig.suptitle("Histograme pentru variabila " + variabila)
    clase = np.unique(p)
    # q este nr de clase
    q = len(clase)
    # sharey=True - partajez axa Y
    axe = fig.subplots(1, q, sharey=True)
    for i in range(q):
        axe[i].set_xlabel(clase[i])
        # din z iau doar instantele unde partitia
        axe[i].hist(x=z[p == clase[i]], range=(min(z), max(z)), rwidth=0.9)


def plot_partitie(z, partitie,title, etichete=None):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot partitie"+title)
    scatterplot(x=z[:, 0], y=z[:, 1], hue=partitie, hue_order=np.unique(partitie), ax=ax)
    if etichete is not None:
        for i in range(len(etichete)):
            ax.text(z[i, 0], z[i, 1], etichete[i])


def show():
    plt.show()
