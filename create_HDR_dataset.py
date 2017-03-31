import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as im
from PIL import Image
import os
import h5py
import cv2


def plot_arr(arr,name):
	im.imsave(name+'.png',arr,cmap=plt.cm.gray)

def crop(img, cr):
	return img[cr[0]:img.shape[0]-cr[2],cr[1]:img.shape[1]-cr[3],]

def patchify(img,size):
	H = img.shape[0]
	W = img.shape[1]

	batch = []

	for i in range(H//size):
		for j in range(W//size):
			x = i*size
			y = j*size
			batch.append(img[x:x+size,y:y+size,])
	return np.array(batch)

def create_dataset(N,source,destination,dataset_name):
	dataset = []
	labelset = []

	cr = (50,50,50,50) # x,y,-x,-y
	data = []
	data_val = []
	for filename in os.listdir(source):
		# print filename
		if filename.endswith(".png"):
			img = cv2.imread(source+filename)/255.0

			img = crop(img,cr)
			# cv2.imwrite('image'+filename+'.png',img*255)
			# print img.shape
			patches = patchify(img,N)
			# print patches.shape

			data.append(patches)
			data_val.append(img)

	data = np.array(data)
	data_val = np.array(data_val)

	print data.shape
	dataset = np.transpose(data,(1,4,0,2,3))
	print dataset.shape
	cv2.imwrite('orig_data.bmp',dataset[1,:,1,:,:].transpose((1,2,0))*256)

	label = cv2.imread(source+"../memorial.png")/255.0
	label = crop(label,cr)
	# print label.shape
	label_val = label
	labelset = patchify(label,N)
	# cv2.imwrite('orig.bmp',labelset[1,:,:,:]*256)
	labelset = np.transpose(labelset,(0,3,1,2))
	# print labelset.shape
	labelset = labelset.reshape(labelset.shape[0],-1,1,labelset.shape[2],labelset.shape[3])
	print labelset.shape
	l = labelset[1,:,0,:,:].transpose((1,2,0))
	# print l.shape
	# cv2.imwrite('orig1.bmp',l*256)



	f = h5py.File(destination+dataset_name+'.h5','w')
	f['data'] = np.array(dataset)
	f['label'] = np.array(labelset)
	# print labelset[0,]
	# print dataset[0,]
	f.close()

	print "dataset created"

	# plot_arr(labelset[0],'test_img')
	# plot_arr(np.clip(dataset[0],0.0,10.0),'test_fft')
	# print labelset[0], dataset[0]
	# print np.max(dataset[0]), np.min(dataset[0])
	# print len(dataset)
	f = h5py.File(destination+dataset_name+'_val.h5','w')
	print data_val.shape, label_val.shape
	data_val = data_val.transpose(3,0,1,2)
	label_val = label_val.transpose(2,0,1).reshape(-1,1,label_val.shape[0],label_val.shape[1])
	f['data'] = np.array(data_val)
	f['label'] = np.array(label_val)
	print data_val.shape, label_val.shape
	f.close()


def main():
	output_path = '/home/sushobhan/Documents/data/deep_hdr/datasets/'
	source = '/home/sushobhan/Documents/data/deep_hdr/Memorial_SourceImages/'
	N = 64

	dataset_name = 'patch_'+str(N)

	create_dataset(N,source,output_path,dataset_name)



	

	# plot_arr(arr,'test_img')
	# plot_arr(trans,'test_fft',t='c')

if __name__=='__main__':
	main()
