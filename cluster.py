import csv, numpy

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

irisCentroid = numpy.empty([3, 4], dtype = float)
with open('Iris_Initial_Centroids.csv', 'rb') as irisdata:
    irisreader = csv.reader(irisdata, delimiter=',')
    i = 0
    for row in irisreader:
        irisCentroid[i] = row
        i += 1
#print irisCentroid

for t in range(0, iris_t):
    iris_c = [[] for i in range(0,3)]
    minCentroid = []
    for arr in irisData:
        for cen in irisCentroid:
            minCentroid.append(numpy.linalg.norm(arr - cen))
        iris_c[minCentroid.index(min(minCentroid))].append(arr)
        minCentroid = []

    for k in range(0,3):
        irisCentroid[k] = [float(sum(l))/len(l) for l in zip(*iris_c[k])]

print irisCentroid
