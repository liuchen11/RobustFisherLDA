import sys
import numpy as np

import load
import util

def estimate(trainX, trainY, resample_num):
	sample_pos_means = []
	sample_pos_covs = []
	sample_neg_means = []
	sample_neg_covs = []

	for i in xrange(resample_num):
		[sampledX, sampledY] = util.resample(trainX, trainY)
		[positiveX, negativeX] = util.split(sampledX, sampledY)

		sample_pos_means.append(np.mean(positiveX, 0))
		sample_neg_means.append(np.mean(negativeX, 0))
		sample_pos_covs.append(np.cov(np.array(positiveX).T))
		sample_neg_covs.append(np.cov(np.array(negativeX).T))

	nominal_pos_mean = np.mean(sample_pos_means, 0)
	nominal_neg_mean = np.mean(sample_neg_means, 0)
	nominal_pos_cov = np.mean(sample_pos_covs, 0)
	nominal_neg_cov = np.mean(sample_neg_covs, 0)

	sample_pos_means_cov = np.cov(np.array(sample_pos_means).T)
	sample_neg_means_cov = np.cov(np.array(sample_neg_means).T)

	P_pos = np.linalg.inv(sample_pos_means_cov) / len(trainX)
	P_neg = np.linalg.inv(sample_neg_means_cov) / len(trainX)

	rho_pos = 0
	rho_neg = 0

	for cov_matrix in sample_pos_covs:
		dis = util.F_norm(cov_matrix - nominal_pos_cov)
		rho_pos = max(dis, rho_pos)

	for cov_matrix in sample_neg_covs:
		dis = util.F_norm(cov_matrix - nominal_neg_cov)
		rho_neg = max(dis, rho_neg)

	return [nominal_pos_mean, P_pos, nominal_neg_mean, P_neg,
		nominal_pos_cov, rho_pos, nominal_neg_cov, rho_neg]

if __name__ == '__main__':

	if len(sys.argv)<4:
		print 'Usage: python robustFisherLDA.py <dataFile> <dimension> <alpha> (<resample_num>) (<split_token>)'
		exit(0)

	data_file = sys.argv[1]
	dimension = int(sys.argv[2])
	alpha = float(sys.argv[3])
	resample_num = int(sys.argv[4]) if len(sys.argv)>4 else 100
	split_token = sys.argv[5] if len(sys.argv)>5 else ','

	data_loader = load.loader(file_name = data_file, split_token = split_token)
	[dataX, dataY] = data_loader.load()

	[trainX, trainY, testX, testY] = util.divide(dataX, dataY, alpha)

	[pos_mean, pos_P, neg_mean, neg_P, pos_cov, pos_rho, neg_cov, neg_rho] = estimate(trainX, trainY)



