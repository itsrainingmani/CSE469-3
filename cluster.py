import csv
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def kmeans():

    yeast_t = 7
    yeast_k = 6

    yeastData = np.empty([614, 7], dtype = float)
    with open('YeastGene.csv', 'rb') as yeastdata:
        yeastreader = csv.reader(yeastdata, delimiter=',')
        i = 0
        for row in yeastreader:
            yeastData[i] = row
            i += 1
    #print yeastData

    yeastCentroid = np.empty([yeast_k, 7], dtype = float)
    with open('YeastGene_Initial_Centroids.csv', 'rb') as yeastdata:
        yeastreader = csv.reader(yeastdata, delimiter=',')
        i = 0
        for row in yeastreader:
            yeastCentroid[i] = row
            i += 1
    #print yeastCentroid

    for t in range(0, yeast_t):
        yeast_c = [[] for i in range(0,yeast_k)]
        minCentroid = []
        for arr in yeastData:
            for cen in yeastCentroid:
                minCentroid.append(np.linalg.norm(arr - cen))
            yeast_c[minCentroid.index(min(minCentroid))].append(arr)
            minCentroid = []

        for k in range(0,yeast_k):
            yeastCentroid[k] = [float(sum(l))/len(l) for l in zip(*yeast_c[k])]
    #print "The new yeast Centroid values\n"
    #print yeastCentroid

    #print "The cluster sizes are - "
    print len(yeast_c[0]), len(yeast_c[1]), len(yeast_c[2]), len(yeast_c[3]), len(yeast_c[4]), len(yeast_c[5])
    clusters = np.zeros([614, 7], dtype=float)
    prev_len = 0
    for i in range(0,6):
        for j in range(0,len(yeast_c[i])):
            clusters[prev_len] = yeast_c[i][j]
            prev_len += 1

    sklearn_pca = sklearnPCA(n_components = 2)
    transf = sklearn_pca.fit_transform(clusters)
    plt.plot(transf[0:140, 0], transf[0:140, 1],'*', markersize = 7, color='blue', alpha=0.5, label='cluster 1')
    plt.plot(transf[140:191, 0], transf[140:191, 1],'*', markersize = 7, color='red', alpha=0.5, label='cluster 2')
    plt.plot(transf[191:355, 0], transf[191:355, 1],'*', markersize = 7, color='green', alpha=0.5, label='cluster 3')
    plt.plot(transf[355:376, 0], transf[355:376, 1],'*', markersize = 7, color='indigo', alpha=0.5, label='cluster 4')
    plt.plot(transf[376:538, 0], transf[376:538, 1],'*', markersize = 7, color='yellow', alpha=0.5, label='cluster 5')
    plt.plot(transf[538:614, 0], transf[538:614, 1],'*', markersize = 7, color='black', alpha=0.5, label='cluster 6')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.title("Kmeans")
    plt.show()

def hcluster():
    exData = np.empty([22, 8], dtype = float)
    with open('Utilities.csv', 'rb') as exdata:
        exreader = csv.reader(exdata, delimiter=',')
        i = 0
        for row in exreader:
            exData[i] = row
            i += 1

    dMatrix = np.zeros([22, 22], dtype = float)
    for i in range(0,22):
        for j in range(i,22):
            dMatrix[i][j] = np.linalg.norm(exData[i] - exData[j])
    #print fastcluster.single(dMatrix, preserve_input = True)
    #print fastcluster.linkage_vector(dMatrix, method = 'single', metric = 'euclidean')
    Z = linkage(dMatrix, method='single', metric='euclidean')
    for row in Z:
        row[0] += 1
        row[1] += 1
    print "The order for the hierarchial clustering is"
    print Z
    """
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z, truncate_mode='level', p=100, leaf_rotation=90., leaf_font_size=14., show_contracted=True, show_leaf_counts=True)
    plt.show()
    """
kmeans()
#hcluster()
