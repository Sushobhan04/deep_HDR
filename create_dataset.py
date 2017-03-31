import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as im
import imageio as hdr
from PIL import Image
import os
import h5py


def plot_arr(arr,name):
	im.imsave(name+'.png',arr,cmap=plt.cm.gray)

def patchify(img,size):
	H = img.shape[0]
	W = img.shape[1]

	batch = []

	for i in range((2*H)//size-1):
		for j in range((2*W)//size-1):
			x = i*size/2
			y = j*size/2
			batch.append(img[x:x+size,y:y+size])
	return np.array(batch)

def batch_fft(batch):
	batch_f = []
	for x in batch:
		batch_f.append(np.absolute(fft.fftshift(fft.fft2(x))))
	return batch_f

def create_dataset(N,source,destination,dataset_name):
	dataset = []
	labelset = []

	for filename in os.listdir(source):
		img = np.asarray(Image.open(source+filename).convert('L'))/255.0
		patches = patchify(img,N)
		patches_fft = batch_fft(patches)

		dataset.extend(patches_fft)
		labelset.extend(patches)


	f = h5py.File(destination+dataset_name+'.h5','w')
	f['data'] = np.array(dataset)
	f['label'] = np.array(labelset)
	f.close()

	print "dataset created"

	plot_arr(labelset[0],'test_img')
	plot_arr(np.clip(dataset[0],0.0,10.0),'test_fft')
	print labelset[0], dataset[0]
	print np.max(dataset[0]), np.min(dataset[0])
	print len(dataset)

def main():
	output_path = '/home/sushobhan/Documents/data/ptychography/datasets/'
	source = '/home/sushobhan/Documents/data/ptychography/data/Set91/'
	N = 32

	dataset_name = 'patch_'+str(N)

	create_dataset(N,source,output_path,dataset_name)



	

	# plot_arr(arr,'test_img')
	# plot_arr(trans,'test_fft',t='c')

if __name__=='__main__':
	main()
