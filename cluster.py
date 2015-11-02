import csv
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def kmeans():

    iris_t = 12
    iris_k = 3

    irisData = np.empty([150, 4], dtype = float)
    with open('Iris.csv', 'rb') as irisdata:
        irisreader = csv.reader(irisdata, delimiter=',')
        i = 0
        for row in irisreader:
            irisData[i] = row
            i += 1
    #print irisData

    irisCentroid = np.empty([iris_k, 4], dtype = float)
    with open('Iris_Initial_Centroids.csv', 'rb') as irisdata:
        irisreader = csv.reader(irisdata, delimiter=',')
        i = 0
        for row in irisreader:
            irisCentroid[i] = row
            i += 1
    #print irisCentroid
    temp_iris = [[] for i in range(0,iris_k)]
    for t in range(0, iris_t):
        iris_c = [[] for i in range(0,iris_k)]
        minCentroid = []
        for arr in irisData:
            for cen in irisCentroid:
                minCentroid.append(np.linalg.norm(arr - cen))
            iris_c[minCentroid.index(min(minCentroid))].append(arr)
            minCentroid = []
        temp_iris = iris_c

        for k in range(0,iris_k):
            irisCentroid[k] = [float(sum(l))/len(l) for l in zip(*iris_c[k])]
    print "The new Iris Centroid values\n"
    print irisCentroid

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
    print "The new yeast Centroid values\n"
    print yeastCentroid

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
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z, truncate_mode='level', p=100, leaf_rotation=90., leaf_font_size=14., show_contracted=True, show_leaf_counts=True)
    plt.show()

kmeans()
hcluster()
