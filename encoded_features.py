from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization,MaxPooling2D
from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate

import keras

import functools

    
'''
Encoder module for end-to-end network. input to the encoder module is an image, which after going
through a series of convolutional, batchnormalizaiton and maxpooling layers; a non-redundant features
are generated.
'''
def encoder(inputImage):
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputImage) #32 x 32 x 32
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)                                 #16 x 16 x 32
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)          #16 x 16 x 64
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)                                 #8 x 8 x 64
    encodedData=p2
    return(encodedData)


#decoder module for end-to-end network (not used). 
def decoder(encodedData):
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(encodedData) #8 x 8 x 128
    c5 = BatchNormalization()(c5)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)           #8 x 8 x 64
    c6 = BatchNormalization()(c6)
    up1 = UpSampling2D((2,2))(c6)                                            #16 x 16 x 64
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)          # 16 x 16 x 32
    c7 = BatchNormalization()(c7)
    up2 = UpSampling2D((2,2))(c7)                                            # 32 x 32 x 32
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2)   # 32 x 32 x 3
    return(decoded)

'''
autoencoder module for seperate-training model. First, the autoencoder module is trained seperately
the code generated from autoencoder module is then transfered to image classification network.
'''
def autoencoder():
    input = Input((32,32,3))                                                          #32 x 32 x 3
    
    #encoder
    encoderLayer = Conv2D(32, (3, 3), activation='relu', padding='same')(input)       #32 x 32 x 32
    encoderLayer = BatchNormalization()(encoderLayer)
    encoderLayer = Conv2D(32, (3, 3), activation='relu', padding='same')(encoderLayer)#32 x 32 x 32 
    encoderLayer = BatchNormalization()(encoderLayer)

    encoderLayer = MaxPool2D(2)(encoderLayer)                                         #16 x 16 x32
    encoderLayer = Conv2D(64, (3, 3), activation='relu', padding='same')(encoderLayer)#16 x 16 x64 
    encoderLayer = BatchNormalization()(encoderLayer)
    encoderLayer = Conv2D(64, (3, 3), activation='relu', padding='same')(encoderLayer)#16 x 16 x64
    encoderLayer = BatchNormalization()(encoderLayer)

    encoderLayer = MaxPool2D(2)(encoderLayer)                                         #8 x 8 x64
    code = BatchNormalization()(encoderLayer)
    
    # Decoder
    decoderLayer = UpSampling2D((2,2))(code)                                          #16 x 16 x64
    
    decoderLayer = Conv2D(64, (3, 3), activation='relu', padding='same')(decoderLayer)#16 x 16 x64 
    decoderLayer = BatchNormalization()(decoderLayer)
    decoderLayer = Conv2D(64, (3, 3), activation='relu', padding='same')(decoderLayer) #16 x 16 x64
    decoderLayer = BatchNormalization()(decoderLayer)

    
    decoderLayer = UpSampling2D((2,2))(decoderLayer)                                    #32 x 32 x64
    decoderLayer = Conv2D(32, (3, 3), activation='relu', padding='same')(decoderLayer) #32 x 32 x32
    decoderLayer = BatchNormalization()(decoderLayer)
    decoderLayer = Conv2D(32, (3, 3), activation='relu', padding='same')(decoderLayer) #32 x 32 x32
    decoderLayer = BatchNormalization()(decoderLayer)

        
    decoderLayer = Conv2D(3, 1)(decoderLayer)                                           #32 x 32 x3
    decoderLayer = Activation("sigmoid")(decoderLayer)                                  #sigmoid activation function
    return Model(input,code), Model(input,decoderLayer)


'''
Encoded non-redundant features are then passed to convolutional neural network.
'''

def classifier_conv(inp):
    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input) #8 x 8 x 32
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                        #4 x 4 x 32
    pool1 = Dropout(0.5)(pool1)                                          #regularizer

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #4 x 4 x 64
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                        #2 x 2 x 64
    pool2 = Dropout(0.5)(pool2)
    flat = Flatten()(pool2)
    den = Dense(128, activation='relu')(flat)
    classify = Dense(10, activation='softmax',name='classification')(den)
    outputs = [classify]
    
    return Model(input,outputs)
'''
Encoded non-redundant features are then passed to convolutional neural network.
'''

def classic_conv():

    input = Input((32,32,3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input) #8 x 8 x 32
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                         #4 x 4 x 32
    pool1 = Dropout(0.5)(pool1)                                           #regularizer

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  #4 x 4 x 64
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                          #2 x 2 x 64
    pool2 = Dropout(0.5)(pool2)
    flat = Flatten()(pool2)
    den = Dense(128, activation='relu')(flat)
    classify = Dense(10, activation='softmax',name='classification')(den)  #softmax activation
    outputs = [classify]
    
    return Model(input,outputs)

'''
End-to-end model is a network that combines encoding module of autoencoder with image
classifier network.
'''

def end_to_end():  
    input = Input((32,32,3))
    encoderOutput=encoder(input)
    decoderOutput=decoder(encoderOutput)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoderOutput)#8 x 8x 32
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                               #4 x 4 x 32
    pool1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)        #4 x 4 x 64
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 
    pool2 = Dropout(0.5)(pool2)
    flat = Flatten()(pool2)
    den = Dense(128, activation='relu')(flat)
    classify = Dense(10, activation='softmax',name='classification')(den)
    outputs = [classify]
    
    return Model(input,outputs)


