import random
import math
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

def split(dataX, dataY):
	'''
	divide the whole dataset into positive and negative ones
	'''
	instances = len(dataX)
	positiveX = []
	negativeX = []

	for i in xrange(instances):
		if dataY[i] == 1:
			positiveX.append(dataX[i])
		else:
			negativeX.append(dataX[i])

	return [positiveX, negativeX]

def F_norm(matrix):
	'''
	calculate the Frobenius norm of a matrix i.e |vec(A)|_2
	'''
	squared = map(lambda x: x**2, matrix)
	squared_sum = np.sum(squared)
	return math.sqrt(squared_sum)

def M_norm(matrix, vector):
	'''
	calculate the M-norm of a matrix i.e \sqrt{v.T * M * v}
	'''
	squared = np.dot(vector.T, np.dot(matrix, vector))
	return math.sqrt(squared[0][0])

