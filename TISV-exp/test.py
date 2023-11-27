# This code is referenced:
# https://github.com/HarryVolek/PyTorch_Speaker_Verification
# for TI-SV testing, calculate the EER

import numpy as np
import torch
import os
import random
import glob
import librosa

from hparam import hparam as hp
from dataloader import vox_dataloader
from model import SpeechEncoder
from utils import get_centroids, get_cossim, calc_loss


def test(ckpt_path):
	# Define device
	device = torch.device(hp.device)
	# Define model
	model = SpeechEncoder().to(device)
	# Load pretrained parameters
	checkpoint = torch.load(ckpt_path)
	model.load_state_dict(checkpoint['model'])
	model.eval()
	# Current epoch
	epoch = checkpoint['epoch']
	# calculate EER
	avg_EER = 0
	dataloader = vox_dataloader(hp.test.test_dir, aug = False)
	count = 0
	for mels in dataloader:
		mels = mels.squeeze(0).to(device)
		mels = torch.reshape(mels, (hp.train.sub_n, hp.train.wav_n, mels.size(1), mels.size(2)))
		enrollment_batch, verification_batch = torch.split(mels, int(mels.size(1)/2), dim=1)
		enrollment_batch = torch.reshape(enrollment_batch, (hp.train.sub_n*hp.train.wav_n//2,
			enrollment_batch.size(2), enrollment_batch.size(3)))
		verification_batch = torch.reshape(verification_batch, (hp.train.sub_n*hp.train.wav_n//2,
			verification_batch.size(2), verification_batch.size(3)))
		perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
		unperm = list(perm)
		for i,j in enumerate(perm):
			unperm[j] = i
		verification_batch = verification_batch[perm]
		enrollment_embeddings = model(enrollment_batch)
		verification_embeddings = model(verification_batch)
		verification_embeddings = verification_embeddings[unperm]

		enrollment_embeddings = torch.reshape(enrollment_embeddings,
			(hp.train.sub_n, hp.train.wav_n//2,enrollment_embeddings.size(1)))
		verification_embeddings = torch.reshape(verification_embeddings,
			(hp.train.sub_n, hp.train.wav_n//2, verification_embeddings.size(1)))

		enrollment_centroids = get_centroids(enrollment_embeddings)
		sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
		# calculating EER
		diff = 1
		EER=0
		EER_thresh = 0
		EER_FAR=0
		EER_FRR=0

		for thres in [0.01*i+0.5 for i in range(50)]:
			sim_matrix_thresh = sim_matrix>thres
			FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.train.sub_n))])
				/(hp.train.sub_n-1.0)/(float(hp.train.wav_n/2))/hp.train.sub_n)
			FRR = (sum([hp.train.wav_n/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.train.sub_n))])
				/(float(hp.train.wav_n/2))/hp.train.sub_n)
			# Save threshold when FAR = FRR (=EER)
			if diff > abs(FAR-FRR):
				diff = abs(FAR-FRR)
				EER = (FAR+FRR)/2
				EER_thresh = thres
				EER_FAR = FAR
				EER_FRR = FRR
		avg_EER += EER
		count += 1
	avg_EER = avg_EER.detach().cpu().numpy()
	print('Test EER: ' + str(avg_EER/count))
