import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import random

from hparam import hparam as hp
from dataloader import vox_dataloader
from model import SpeechEncoder
from test import test
from ge2e import GE2ELoss

def create_model(ckpt_path):
	# Define device
	device = torch.device(hp.device)
	# Define model and loss module
	model = SpeechEncoder().to(device)
	ge2e_loss = GE2ELoss(device)
	# Define optimizer
	opt_fn = torch.optim.Adam([
		{'params': model.parameters()},
		{'params': ge2e_loss.parameters()},
	], lr=hp.train.lr, weight_decay=5e-5)
	torch.save({
		'epoch': 0,
		'model': model.state_dict(),
		'ge2e_loss': ge2e_loss.state_dict(),
		'optimizer': opt_fn.state_dict(),
	},ckpt_path)

def train(previous_ckpt_path, writer):
	# Define device
	device = torch.device(hp.device)
	# Define dataloader
	dataloader = vox_dataloader(hp.train.train_dir, aug = False)
	# Define classifier model and loss module
	model = SpeechEncoder().to(device)
	ge2e_loss = GE2ELoss(device)
	# Load pretrained parameters
	checkpoint = torch.load(previous_ckpt_path)
	model.load_state_dict(checkpoint['model'])
	ge2e_loss.load_state_dict(checkpoint['ge2e_loss'])
	# Define optimizer
	opt_fn = torch.optim.Adam([
		{'params': model.parameters()},
		{'params': ge2e_loss.parameters()},
	], lr=hp.train.lr, weight_decay=5e-5)
	opt_fn.load_state_dict(checkpoint['optimizer'])
	# Set to train mode
	model.train()
	start_epoch = checkpoint['epoch']
	# Not sure why this is important
	torch.backends.cudnn.benchmark = True
	# Start training
	for epoch in range(start_epoch,start_epoch+hp.train.ckpt_interval):
		total_loss = 0
		for mels in dataloader:
			# get mels
			mels = mels.squeeze(0).to(device)
			# random arrange
			perm = random.sample(range(0, hp.train.sub_n*hp.train.wav_n), hp.train.sub_n*hp.train.wav_n)
			unperm = list(perm)
			for i,j in enumerate(perm):
				unperm[j] = i
			mels = mels[perm]
			# train network
			opt_fn.zero_grad()
			embs = model(mels)
			# unperm
			embs = embs[unperm]
			# contrastive loss part
			embs = torch.reshape(embs, (hp.train.sub_n, hp.train.wav_n, embs.size(1)))
			loss = ge2e_loss(embs)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
			opt_fn.step()
			# update loss values
			total_loss = total_loss+loss
		# Wrtie total loss to log
		print('Epoch [{}] - total loss: {}'.format(epoch+1, total_loss))
		writer.add_scalar('total loss', total_loss, epoch+1)
	# Save checkpoint
	checkpoint_path = hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(epoch+1)
	print('Saving model in ['+checkpoint_path+']')
	torch.save({
		'epoch': epoch+1,
		'model': model.state_dict(),
		'ge2e_loss': ge2e_loss.state_dict(),
		'optimizer': opt_fn.state_dict(),
	},checkpoint_path)
	print('Saving END')
	return checkpoint_path

if __name__ == '__main__':
	# Write tensorboard
	torch.manual_seed(233)
	os.makedirs(hp.train.ckpt_dir, exist_ok=True)
	writer = SummaryWriter(hp.train.ckpt_dir)
	# Restore ckpt or create new
	if (not hp.train.restoring):
		create_model(hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(0))
		previous_ckpt = hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(0)
	else:
		previous_ckpt = hp.train.restore_path
	# Load epochs
	checkpoint = torch.load(previous_ckpt)
	start_epoch = checkpoint['epoch']
	# Training procedure
	for epoch in range(start_epoch,hp.train.epochs,hp.train.ckpt_interval):
		# Training
		now_ckpt = train(previous_ckpt, writer)
		# testing
		test(now_ckpt)
		previous_ckpt = now_ckpt
		# delete 3 intervals before checkpoint
		temp_path = hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(int(epoch-3*hp.train.ckpt_interval))
		if os.path.exists(temp_path):
			os.remove(temp_path)

