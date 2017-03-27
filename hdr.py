from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    Reshape
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import h5py
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras import callbacks
from keras.callbacks import LearningRateScheduler,EarlyStopping
import math
import sys

K.set_image_dim_ordering('th')

if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]

def BatchGenerator(files,batch_size, net_type = 'conv'):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data['data'])
            label = np.array(curr_data['label'])
            # print data.shape, label.shape

            for i in range((data.shape[0]-1)//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,5:6,]*100
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                yield (data_bat, label_bat)

def SetGenerator(file):
    curr_data = h5py.File(file,'r')
    data = np.array(curr_data['data'])
    label = np.array(curr_data['label'])*100
    # print data.shape, label.shape
    return data,label

def schedule(epoch):
    lr = 0.0001
    if epoch<10:
        return lr
    elif epoch<20:
        return lr/10
    elif epoch<100:
        return lr/100
    else:
        return lr/8

def create_cnn_model(input,output_shape = (1,128,128),border_mode = 'same'):
    kernels = [3,3,3]
    num_ker = [32,32,3]
    # pool_size = (2,2)

    temp = Convolution2D(32, (kernels[0], kernels[0]), border_mode=border_mode, init = 'he_normal')(input)
    # temp = BatchNormalization(mode=0, axis=norm_axis)(temp)
    temp = Activation('relu')(temp)
    # temp = MaxPooling2D(pool_size=pool_size)(temp)
    
    temp = Convolution2D(32, (kernels[1], kernels[1]), border_mode=border_mode, init = 'he_normal')(temp)
    # temp = BatchNormalization(mode=0, axis=norm_axis)(temp)
    temp = Activation('relu')(temp)
    

    temp = Convolution2D(3, (kernels[2], kernels[2]), border_mode=border_mode, init = 'he_normal')(temp)
    # temp = BatchNormalization()(temp)
    # temp = Activation('relu')(temp)


    model = Model(input=input, output=temp)

    return model

def create_dense_model(input,output_shape = (16,16),border_mode='same'):
    temp = Flatten()(input)

    temp = Dense(output_shape[1]*output_shape[2]*2//3,init='he_normal')(temp)
    # temp = BatchNormalization()(temp)
    temp = Activation('relu')(temp)


    # temp = Dense(8*1024,init='he_normal')(temp)
    # # temp = BatchNormalization()(temp)
    # temp = Activation('relu')(temp)

    temp = Dense(output_shape[1]*output_shape[2],init='he_normal')(temp)
    # temp = BatchNormalization()(temp)
    temp = Activation('relu')(temp)

    temp = Reshape(output_shape)(temp)


    model = Model(input=input, output=temp)

    return model


def train_model(path_train,home,model_name,mParam):

    lrate = mParam['lrate']
    epochs = mParam['epochs']
    decay = mParam['decay']
    train_batch_size = mParam['train_batch_size']
    val_batch_size = mParam['val_batch_size']
    samples_per_epoch = mParam['samples_per_epoch']
    nb_val_samples = mParam['nb_val_samples']

    input_shape = mParam['input_shape']
    output_shape = mParam['output_shape']

    net_type = mParam['net_type']

    print input_shape,output_shape

    border_mode = mParam['border_mode']
    norm_axis = 1

    input = Input(shape=input_shape)

    if net_type=='dense':
        model = create_dense_model(input,output_shape=output_shape,border_mode=border_mode)
    elif net_type == 'conv':
        model = create_cnn_model(input,output_shape=output_shape,border_mode=border_mode)


    # sgd = Adadelta(lr=lrate, rho=0.95, epsilon=1e-08, decay=decay)
    train_files = path_train+'datasets/'+'patch_32.h5'
    # val_files = [path_train+'datasets/'+'valset_1.h5']
    data,label = SetGenerator(train_files)
    
    lrate_sch = LearningRateScheduler(schedule)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    callbacks_list = [lrate_sch,early_stop]

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)

    model.compile(loss='mean_squared_error',
              optimizer=sgd)

    model.fit(data,label,validation_split=0.1,nb_epoch = epochs,verbose=1 ,callbacks=callbacks_list)

    model.save(path_train+'models/'+model_name+'.h5')

    # print model.summary()


def main():

    path_train =  "/home/sushobhan/Documents/data/deep_hdr/"
    home = "/home/sushobhan/Documents/research/deep_hdr/"
    model_name = sys.argv[1]

    mParam = {}
    mParam['lrate'] = 0.001
    mParam['epochs'] = 50
    mParam['decay'] = 0.0
    mParam['net_type'] = 'conv'
    mParam['border_mode'] = 'same'

    mParam['input_shape'] = (48,32,32)
    mParam['output_shape'] = (3,32,32)

    mParam['train_batch_size'] = 128
    mParam['val_batch_size'] = 1
    mParam['samples_per_epoch'] = 91
    mParam['nb_val_samples'] = 5

    # if mParam['net_type'] =='conv':
    #     mParam['input_shape'] = (1,mParam['input_shape'][0],mParam['input_shape'][1])

    train_model(path_train,home,model_name,mParam)


if __name__ == '__main__':
    main()