import pickle
import glob

if __name__ == '__main__':
	# train datadict
	train_dict = {}
	min_N = 100000
	# vox1 development part
	train_path = 'path/to/vox/vox1-dev/'
	train_sublist = glob.glob(train_path+'*/')
	for nowsub in train_sublist:
		sub_wavlist = glob.glob(nowsub+'*/*.wav')
		if len(sub_wavlist)<min_N:
			min_N = len(sub_wavlist)
		train_dict[nowsub] = sub_wavlist
	# vox2 development part
	train_path = 'path/to/vox/vox2-dev/'
	train_sublist = glob.glob(train_path+'*/')
	for nowsub in train_sublist:
		sub_wavlist = glob.glob(nowsub+'*/*.wav')
		if len(sub_wavlist)<min_N:
			min_N = len(sub_wavlist)
		train_dict[nowsub] = sub_wavlist
	print(min_N)
	with open('vox_train.pickle', 'wb') as fh:
		pickle.dump(train_dict, fh)
	print(len(train_dict))

	# test part
	test_dict = {}
	min_N = 100000
	test_path = 'path/to/vox/vox1-test/'
	test_sublist = glob.glob(test_path+'*/')
	for nowsub in test_sublist:
		sub_wavlist = glob.glob(nowsub+'*/*.wav')
		if len(sub_wavlist)<min_N:
			min_N = len(sub_wavlist)
		test_dict[nowsub] = sub_wavlist
	with open('vox_test.pickle','wb') as fh:
		pickle.dump(test_dict, fh)
	print(len(test_dict.keys()))
