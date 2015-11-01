import csv, numpy
from sklearn.decomposition import PCA as sklearnPCA

def kmeans():

    iris_t = 12
    iris_k = 3

    irisData = numpy.empty([150, 4], dtype = float)
    with open('Iris.csv', 'rb') as irisdata:
        irisreader = csv.reader(irisdata, delimiter=',')
        i = 0
        for row in irisreader:
            irisData[i] = row
            i += 1
    #print irisData

    irisCentroid = numpy.empty([iris_k, 4], dtype = float)
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
                minCentroid.append(numpy.linalg.norm(arr - cen))
            iris_c[minCentroid.index(min(minCentroid))].append(arr)
            minCentroid = []
        temp_iris = iris_c

        for k in range(0,iris_k):
            irisCentroid[k] = [float(sum(l))/len(l) for l in zip(*iris_c[k])]
    print "The new Iris Centroid values\n"
    print irisCentroid

    yeast_t = 7
    yeast_k = 6

    yeastData = numpy.empty([614, 7], dtype = float)
    with open('YeastGene.csv', 'rb') as yeastdata:
        yeastreader = csv.reader(yeastdata, delimiter=',')
        i = 0
        for row in yeastreader:
            yeastData[i] = row
            i += 1
    #print yeastData

    yeastCentroid = numpy.empty([yeast_k, 7], dtype = float)
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
                minCentroid.append(numpy.linalg.norm(arr - cen))
            yeast_c[minCentroid.index(min(minCentroid))].append(arr)
            minCentroid = []

        for k in range(0,yeast_k):
            yeastCentroid[k] = [float(sum(l))/len(l) for l in zip(*yeast_c[k])]
    print "The new yeast Centroid values\n"
    print yeastCentroid

def hcluster():
    exData = numpy.empty([6, 5], dtype = float)
    with open('Example.csv', 'rb') as exdata:
        exreader = csv.reader(exdata, delimiter=',')
        i = 0
        for row in exreader:
            exData[i] = row
            i += 1

    dMatrix = numpy.zeros([6, 6], dtype = float)



#kmeans()
hcluster()
