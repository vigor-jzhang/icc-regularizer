import random
import torch
import librosa
import numpy as np
import glob
from hparam import hparam as hp
import pickle

def IR_aug(audio,sr):
	ir_list = glob.glob('IR_aug_16k/*.npy')
	random.shuffle(ir_list)
	h = np.load(ir_list[0])
	# change audio to numpy array
	N = audio.shape[0]
	# convolve the impulse response
	audio_conv = np.convolve(audio,h)
	# downsample the convolved audio
	audio = audio_conv[:N]
	return audio

def aug_part(wav, sr):
	# Define random effect application
	#flag = [random.randint(0,1),random.randint(0,1),random.randint(0,1)]
	flag = [0,random.randint(0,1),random.randint(0,1)]
	if flag[0]==1:
		# Impulse response augmentation
		wav = IR_aug(wav,sr)
	if flag[1]==1:
		wav = 0.9*wav/np.max(np.abs(wav))
		noise_level = random.random()*0.02
		noise = np.random.randn(wav.shape[0],)*noise_level
		wav = wav+noise
	if flag[2]==1:
		bgn_list = glob.glob('noise_wav_16k/*.wav')
		random.shuffle(bgn_list)
		bgn, sr = librosa.core.load(bgn_list[0], sr=hp.data.sr)
		start = random.randint(0, bgn.shape[0]-wav.shape[0])
		end = start+wav.shape[0]
		bgn = bgn[start:end]
		noise_level = random.random()*0.02
		wav = wav+bgn*noise_level
	return wav


def gen_mel_from_wav(f, aug_flag):
	wav, sr = librosa.core.load(f, sr=hp.data.sr)
	if hp.data.sr != sr:
		raise ValueError(f'Invalid sample rate {sr}.')
	wav = 0.9*wav/np.max(np.abs(wav))
	if aug_flag:
		wav = aug_part(wav,sr)
	clip_level = random.random()/2+0.4
	wav = clip_level*wav/np.max(np.abs(wav))
	if aug_flag:
		trim_length = 2.0 + 1.0 * random.random()
	else:
		trim_length = 3.0
	duration = 3.0
	start = random.randint(0, wav.shape[0]-int(hp.data.sr * trim_length))
	end = start+wav.shape[0]
	wav = wav[start:end]
	length = int(hp.data.sr * duration)
	wav = librosa.util.fix_length(wav, size=length)
	# pre emphasis
	wav = librosa.effects.preemphasis(wav)
	# get melspectrogram
	spec = librosa.stft(wav, n_fft=hp.data.n_fft, hop_length=hp.data.hop_size, win_length=hp.data.window_size)
	mag_spec = np.abs(spec)

	mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.n_fft, n_mels=hp.data.n_mels)
	mel_spec = np.dot(mel_basis, mag_spec)
	# db mel spectrogram
	mel_db = librosa.amplitude_to_db(mel_spec).T
	mel_db = torch.from_numpy(mel_db)
	return mel_db

def get_mel(flist, n, aug_flag):
	temp_mel_list = []
	counter = 0
	for item in flist:
		try:
			mel = gen_mel_from_wav(item, aug_flag)
		except:
			continue
		temp_mel_list.append(mel)
		counter += 1
		if counter>n-1:
			break
	# Stack over
	stacked_mel = torch.stack(temp_mel_list)
	return stacked_mel

class VoxDataset(torch.utils.data.Dataset):
	def __init__(self, dir_path, aug = True):
		super().__init__()
		with open(dir_path, 'rb') as fh:
			self.path_dict = pickle.load(fh)
			self.sub_list = list(self.path_dict.keys())
		self.aug = aug

	def __len__(self):
		return int(20)

	def __getitem__(self, idx):
		random.shuffle(self.sub_list)
		mels = []
		for now_id in self.sub_list:
			now_flist = self.path_dict[now_id]
			random.shuffle(now_flist)
			temp_mel = get_mel(now_flist, hp.train.wav_n, self.aug)
			if temp_mel.shape[0] == hp.train.wav_n:
				mels.append(temp_mel)
			if len(mels) >= hp.train.sub_n:
				break
		mels = torch.cat(mels, dim=0)
		return mels.float()

def vox_dataloader(dir_path, aug = True):
	dataloader = VoxDataset(dir_path, aug)
	return torch.utils.data.DataLoader(
		dataloader,
		batch_size=1,
		collate_fn=None,
		shuffle=False,
		num_workers=hp.train.num_workers,
		pin_memory=True,
		drop_last=True)
