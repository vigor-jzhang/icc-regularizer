# Author: Dr. Jianwei Zhang, Arizona State University
# Date: 10/20/2023
# Code for NeurIPS 2023 paper "
# J. Zhang, S. Jayasuriya, V. Berisha, "Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer"
# PyTorch implementation for ICC regularizer

import torch

def cal_icc(M):
	# ANOVA part
	# class number
	n = M.shape[0]
	# samples number
	t = M.shape[1]
	# grand mean, overall mean
	grand_mean = torch.sum(M) / (n*t)
	# class mean (row mean)
	class_mean = torch.mean(M,1)
	# sample mean (column mean)
	uttr_mean = torch.mean(M,0)
	# SS - the sum of squares
	# SS_between - SS of inter-class
	SS_between = torch.sum((class_mean - grand_mean)**2)*t
	# MSB - the mean square for inter-class variance
	MSB = SS_between / (n-1)
	# SSW - SS of intra-class
	SSW = torch.sum((M - class_mean.unsqueeze(1))**2)
	# MSW - the mean square for intra-class variance
	MSW = SSW / (n*(t-1));
	r = (MSB - MSW) / (MSB + (t-1)*MSW)
	return r

def ICCRegularizer(emb):
	# Need embeddings in shape [class #, samples #, emb dim]
	# class # - the number of class
	# samples # - the number of different samples from each class
	# emb dim - embeddings dimension
	emb_dim = emb.shape[2]
	total_icc = 0
	for i in range(emb_dim):
		icc = cal_icc(emb[:,:,i])
		total_icc += icc
	icc_regularizer = 1 - (total_icc / emb_dim)
	return icc_regularizer
