"""

Script for cleaning & clustering data based on generated statistics in Assembly

"""
from time import time

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import metrics
from sklearn import cluster
from sklearn import decomposition
from sklearn import manifold

import matplotlib.pyplot as plt
import mplcursors


from Assembly import Assembly

DIRECTORY = '/Users/indiratatikola/Documents/McGrath/cluster_validation/'
FILE = 'CTRL_group7_054000-055799DLC_dlcrnetms5_demasoni_singlenucMay23shuffle1_50000_el_filtered.h5'
PATH = DIRECTORY + FILE
'''
Removing all assemblies with less than 9 key points, only keeping fully annotated fish
'''
def main(original_hdf):
    df = pd.read_hdf(original_hdf)
    ind1 = df.iloc[:, 0:27].dropna().apply(assemblify, axis=1)
    ind2 = df.iloc[:, 27:].dropna().apply(assemblify, axis=1)
    ind1_stats = pd.DataFrame.from_records([a.prop_dict() for a in ind1])
    ind2_stats = pd.DataFrame.from_records([a.prop_dict() for a in ind2])
    # print("init\t\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tFMI")
    # print(82 * "_")
    stats = pd.concat([ind1_stats, ind2_stats], axis=0)
    assemblies = pd.concat([ind1, ind2], axis=0)
    labels = [0] * ind1.index.size + [1] * ind2.index.size
    # visualize(assemblies, action='cluster', data=stats, labels=labels, cluster_type=optics)
    for i in range(0, 18, 2):
        histogram(ind1_stats.iloc[:, i], ind2_stats.iloc[:, i], stats.columns[i])

    '''
    kmeans_centroids = kmeans(proportions, labels)
    dbscan_centroids = dbscan(proportions, labels)
    hdbscan_centroids = hdbscan(proportions, labels)
    optics_hierarchy = optics(proportions, labels)
    '''

    # hdbscan(stats, labels)
    # visualize(stats)


def assemblify(row):
    assembly = Assembly(PATH, row)
    assembly.populate_matrix()
    assembly.generate_proportions()
    return assembly


def kmeans(data, labels):
    '''
    Kmeans clustering with initialization of centroids using 'k-means++'
    Prints various metrics specific to having ground-truth labels
    :param: data - the data to be clustered
    :param: true_labels - the ground truth labels
    :return: centroids
    '''

    estimator = sk.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=5000, tol=1E-4).fit(data)
    bench_cluster('kmeans++', labels, estimator)
    return estimator

# def kmeans(data, centroids):


def dbscan(data, labels):
    '''
    DBScan clustering
    :param: data - data to be clustered
    :param: true_labels - ground-truth labels
    :return: centroids
    '''

    estimator = sk.cluster.DBSCAN(eps=0.25, min_samples=10, metric='euclidean', algorithm='auto').fit(data)
    bench_cluster('dbscan', labels, estimator)
    return estimator

def hdbscan(data, labels):
    '''
    HDBScan clustering
    :param: data - data to be clustered
    :param: true_labels - ground-truth labels
    :return: centroids
    '''

    estimator = sk.cluster.HDBSCAN(min_cluster_size=500, min_samples=10, cluster_selection_epsilon=0.25,
                                   store_centers='centroid', allow_single_cluster=True).fit(data)
    bench_cluster('hdbscan', labels, estimator)
    # visualize(data, estimator.labels_)
    return estimator

def optics(data, labels):
    '''
    OPTICS clustering
    :param: data - data to be clustered
    :param: true_labels - ground-truth labels
    :return: centroids
    '''
    estimator = sk.cluster.OPTICS(min_samples=10, max_eps=0.75).fit(data)
    bench_cluster('optics', labels, estimator)
    return estimator

def bench_cluster(name, labels, estimator):
    results = [name]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
        metrics.fowlkes_mallows_score
    ]
    results += [m(labels, estimator.labels_) for m in clustering_metrics]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


def visualize(metadata, action='partition', data=None, labels=None, cluster_type=None):
    '''
    using PCA decomposition to reduce to 50 dimensions, as recommended
    using t-SNE to reduce to 2 dimensions
    visualize with matplotlib to see if there are possible cluster groups
    color points based on clustering
    '''

    result = []
    fig = plt.figure()
    plot = fig.add_subplot()
    title = ''
    info_labels = [] # for cursor hover
    after_pca = sk.decomposition.PCA(n_components=10).fit_transform(data)
    t_sne = sk.manifold.TSNE(n_components=2, perplexity=50, n_iter_without_progress=500, n_iter=5000, method='exact')
    result = t_sne.fit_transform(after_pca)
    if action == 'cluster':
        estimator = cluster_type(result, labels)
        plot.scatter(result[:, 0], result[:, 1], marker='x', c=estimator.labels_, cmap='viridis')
        title = FILE[0:25] + '\n Colored By ' + str(cluster_type)
    else:
        plot.scatter(result[:, 0], result[:, 1], marker='x', c=labels, cmap='viridis')
        title = FILE[0:25] + '\n Colored By True Individuals'
    if labels is not None:
        i = 0
        for row in enumerate(metadata):
            # metadata returned as tuple of (index, Assembly object)
            assembly = row[1]
            info = 'frame: ' + str(assembly.frame) + '\nindividual: ' + str(assembly.individual)
            info_labels += [info]
            i += 1

    plot.set_title(title)
    plot.set_xlabel('Dimension 1')
    plot.set_ylabel('Dimension 2')
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(info_labels[sel.target.index]))
    plt.savefig(DIRECTORY + 'plots/' + title + '.png')
    plt.show()

def histogram(column, proportion):
    (vals, bins, irrelevant) = plt.hist(reject_outliers(column, 3))
    title = proportion + '\n' + FILE[0:25]
    plt.title(title)
    plt.xlabel('Proportion')
    plt.ylabel('Frequency')
    x_loc = bins[0] + 0.75 * (bins[-1] - bins[0])
    y_loc = 400
    variance = "σ² = {:.4f}".format(np.var(column))
    plt.text(x_loc, y_loc, variance)
    plt.savefig(DIRECTORY + 'plots/' + title + '.png')
    plt.show()

def histogram(column_one, column_two, proportion):
    column_one = reject_outliers(column_one, 3)
    column_two = reject_outliers(column_two, 3)

    counts, bins, patches = plt.hist(column_one, bins=100, color='purple', alpha=0.5, label="Individual 1")
    plt.hist(column_two, bins=100, color='yellow', alpha=0.5, label="Individual 2")
    title = proportion + '\nBy Individual\n' + FILE[0:25]
    plt.title(title)
    plt.xlabel('Proportion')
    plt.ylabel('Frequency')
    plt.legend()
    x_loc = bins[0] + 0.75 * (bins[-1] - bins[0])
    y_loc = 400
    # variance = "σ² = {:.4f}" + str(np.var(column_one))
    # plt.text(x_loc, y_loc, variance)
    plt.savefig(DIRECTORY + 'plots/' + title + '.png')
    plt.show()


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


main(PATH)