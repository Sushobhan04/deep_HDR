import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as im
from PIL import Image
import imageio as imio
import os
import h5py
import cv2
import matlab.engine


def plot_arr(arr,name):
	im.imsave(name+'.png',arr,cmap=plt.cm.gray)

def patchify(img,size):
	H = img.shape[0]
	W = img.shape[1]

	batch = []

	for i in range(H//size -1):
		for j in range(W//size-1):
			x = i*size
			y = j*size
			batch.append(img[x:x+size,y:y+size,])
	return np.array(batch)

def create_dataset(N,source,destination,dataset_name):
	dataset = []
	labelset = []

	data = []
	for filename in os.listdir(source):
		# print filename
		if filename.endswith(".png"):
			img = cv2.imread(source+filename)/255.0
			# print img.shape
			patches = patchify(img,N)
			# print patches.shape

			data.append(patches)

	data = np.array(data)
	# print data.shape
	dataset = np.transpose(data,(1,0,4,2,3)).reshape(data.shape[1],-1,data.shape[2],data.shape[3])
	print dataset.shape

	label = imio.imread(source+"../memorial.hdr")
	tonemap = cv2.createTonemapDurand(2.2)
	label = tonemap.process(label)
	print label.shape
	labelset = np.transpose(patchify(label,N),(0,3,1,2))
	print labelset.shape



	f = h5py.File(destination+dataset_name+'.h5','w')
	f['data'] = np.array(dataset)
	f['label'] = np.array(labelset)
	print labelset[0,]
	# print dataset[0,]
	f.close()

	print "dataset created"

	# plot_arr(labelset[0],'test_img')
	# plot_arr(np.clip(dataset[0],0.0,10.0),'test_fft')
	# print labelset[0], dataset[0]
	# print np.max(dataset[0]), np.min(dataset[0])
	# print len(dataset)

def main():
	output_path = '/home/sushobhan/Documents/data/deep_hdr/datasets/'
	source = '/home/sushobhan/Documents/data/deep_hdr/Memorial_SourceImages/'
	N = 32

	dataset_name = 'patch_'+str(N)

	create_dataset(N,source,output_path,dataset_name)



	

	# plot_arr(arr,'test_img')
	# plot_arr(trans,'test_fft',t='c')

if __name__=='__main__':
	main()
