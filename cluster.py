import csv, math, numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from scipy.cluster.hierarchy import dendrogram, linkage

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
    exDict = defaultdict(list)
    cList = []
    with open('Utilities.csv', 'rb') as exdata:
        #exreader = csv.reader(exdata, delimiter=',')
        i = 1
        for row in exdata:
            exDict[i] = [float(x) for x in row.split(",")]
            cList.append(i)
            i += 1

    dMatrix = np.zeros([22, 22])
    eMatrix = np.zeros([22, 22])
    newIndex = len(exDict)
    while (len(exDict) > 1):
        #print exDict
        xPos = 0
        yPos = 0
        cList.sort()
        #print cList
        minDist = 100.0
        numCluster = len(exDict)
        #print (numCluster)
        #dMatrix = np.zeros([numCluster, numCluster])
        for i in range(0, len(cList)):
            for j in range(i, len(cList)):
                dMatrix[i][j] = math.sqrt(sum((exDict[cList[i]][k] - exDict[cList[j]][k])**2 for k in range(0, 8)))
                #print "Current distance - " + str(np.linalg.norm(cList[i] - cList[j]))
                if (dMatrix[i][j] != 0 and dMatrix[i][j] < minDist):
                    minDist = dMatrix[i][j]
                    #print "min distance - " + str(minDist)
                    xPos = i
                    yPos = j
        newList = []
        newIndex += 1
        newList.append(exDict[cList[xPos]])
        newList.append(exDict[cList[yPos]])
        x = cList[xPos]
        y = cList[yPos]
        print "Merged " + str(x) + " and " + str(y) + " into " + str(newIndex)
        newCluster = []
        del exDict[cList[xPos]]
        del exDict[cList[yPos]]
        cList.remove(x)
        cList.remove(y)
        for i in range(0, 8):
            newCluster.append((newList[0][i] + newList[1][i])/2)
        exDict[newIndex] = newCluster
        cList.append(newIndex)
        #print len(cList)
        dMatrix = eMatrix

kmeans()
hcluster()
