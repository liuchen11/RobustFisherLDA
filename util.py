import random
import numpy as np

def divide(dataX, dataY, alpha):
	'''
	divide a dataset into two parts, usually training and testing set
	'''
	instances = len(dataX)
	len1 = int(instances * alpha)
	len2 = instances - len1
	order = np.random.permutation(instances)

	dataX1 = []
	dataY1 = []
	dataX2 = []
	dataY2 = []

	for i in xrange(len1):
		dataX1.append(dataX[order[i]])
		dataY1.append(dataY[order[i]])

	for i in xrange(len2):
		dataX2.append(dataX[order[i + len1]])
		dataY2.append(dataY[order[i + len1]])

	return [dataX1, dataY1, dataX2, dataY2]

def resample(dataX, dataY):
	'''
	sample an equivalent size dataset uniformly from the original one
	'''
	instances = len(dataX)

	sampleX = []
	sampleY = []

	for i in xrange(instances):
		chosen = random.randint(0, instances-1)
		sampleX.append(dataX[chosen])
		sampleY.append(dataY[chosen])

	return [sampleX, sampleY]
