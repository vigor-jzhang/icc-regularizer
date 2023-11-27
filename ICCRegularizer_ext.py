# Copyright (c) 2023 Jianwei Zhang
#
# E-mail: jianwei.zhang@asu.edu, me@jianwei-zhang.me, zhangjianwei-vigor@outlook.com
#
# All rights reserved.
#
# Code for NeurIPS 2023 paper "
# J. Zhang, S. Jayasuriya, V. Berisha, "Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer"
#
# PyTorch implementation for ICC regularizer extended version, for imblanced classes situation,
# which is described in Appendix A

import torch

def cal_icc(x, label):
	# ANOVA part
	# class number
	label = label.type(torch.long)
	unique_label = torch.unique(label)
	n = unique_label.size(dim=0)
	# grand mean, overall mean
	grand_mean = torch.sum(x) / x.size(dim=0)
	# SS - the sum of squares
	# SS_between - SS of inter-class
	class_mean = torch.zeros(n)
	SS_between = 0
	for idx in range(n):
		now_class = unique_label[idx]
		now_x = x[label==now_class]
		class_mean[idx] = torch.mean(now_x)
		SS_between += now_x.size(dim=0) * (class_mean[idx] - grand_mean)**2
	# MSB - the mean square for inter-class variance
	MSB = SS_between / (n-1)
	# MSW_top - average on all points
	# MSW_bottom - average on class
	MSW_top = 0
	MSW_bottom = 0
	for idx in range(n):
		now_class = unique_label[idx]
		now_mean = class_mean[idx]
		now_x = x[label==now_class]
		now_SS = torch.sum((now_x - now_mean)**2)
		MSW_top += now_SS / (now_x.size(dim=0)-1)
		MSW_bottom += now_SS
	MSW_top = MSW_top / n
	MSW_bottom = MSW_bottom / n
	r = (MSB - MSW_top) / (MSB + MSW_bottom)
	return r

def ICCRegularizer(emb, class_label):
	# Need embeddings in shape [samples #, emb dim]
	# Need class label in shape [sample #]
	# samples # - the number of all samples
	# emb dim - embeddings dimension
	if emb.dim() != 2:
		raise ValueError('input emb must be in shape [samples #, emb dim]')
	if emb.size(dim=0) != class_label.size(dim=0):
		raise ValueError('input emb must have same sample number with class label.')
	emb_dim = emb.shape[1]
	total_icc = 0
	for i in range(emb_dim):
		icc = cal_icc(emb[:,i], class_label)
		total_icc += icc
	icc_regularizer = 1 - (total_icc / emb_dim)
	return icc_regularizer

if __name__ == '__main__':
	x = torch.randn(8, 16)
	class_label = torch.tensor([1, 1, 1, 1, 2, 2, 3, 3], dtype=torch.long)
	print(ICCRegularizer(x, class_label))