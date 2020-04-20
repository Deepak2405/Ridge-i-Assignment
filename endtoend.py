'''
In this code, end-to-end model for supervised-unsupervised learning.
The dataset is trained,validated and tested accordingly. An imbalanced
dataset of standard CIFAR-10 is generated. 20 percent of the training data is
set for validation
'''


from keras.layers import Input
from keras.models import Model 	
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.objectives import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils,plot_model
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from encoded_features import end_to_end
from report import report,show_test_end_to_end
from data_modification import data_modify
import matplotlib.pyplot as plt
import keras.backend as K
import keras
import numpy as np

##callback function
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)                                      
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]

##dataset for task -> CIFAR10 dataset. 5000 images/class and 1000 images/class for training and test set.
(trainX, trainY), (testX, testY) = cifar10.load_data()


##modify the training and test dataset. data_modify() is user defined function written in data_modification.py
trainXModified, trainYModified,testXModified,testYModified = data_modify(trainX, trainY,testX, testY)

##splitting the training dataset into training set and validation set.
trainX, validX, trainYf, validYf = train_test_split(trainXModified, trainYModified, test_size=0.2, random_state=42, shuffle= True)

##converting the classes labels into one-hot classifier
trainY = keras.utils.to_categorical(trainYf, 10)
validY = keras.utils.to_categorical(validYf 10)
testY_one_hot = keras.utils.to_categorical(testYModified, 10)

##design of end-to-end model. end_to_end() is user defined function written in encoded_features.py
model = end_to_end()

##end-to-end model uses log-loss function and uses stochastic gradient descent optimizer to obtain optimal weights.
model.compile(loss = 'categorical_crossentropy',loss_weights = {'classification': 0.9}, optimizer = SGD(lr= 0.01, momentum= 0.9), metrics = {'classification': 'accuracy'})

## since the input to the end-to-end model's classifier network is images. we use image augmentation method for robust training.
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest")


print(model.summary())
##saving the model archietecture.
plot_model(model, to_file='model_end_to_end.png')                                  

history= model.fit_generator(aug.flow(trainX, trainY, batch_size=512),  
	validation_data=(validX, validY), steps_per_epoch=len(trainX) // 512,
	epochs=100)

## After training the model, using the features to predict the classes in the test dataset. 
predictions = model.predict(testXModified)#[1]

##display of scatter plot
test_codes_embedded = TSNE(n_components=2).fit_transform(predictions)
fig6 = plt.figure()
plt.scatter(test_codes_embedded[:,0], test_codes_embedded[:,1], c=testYModified, cmap=plt.get_cmap("tab10"))
plt.colorbar()
fig6.savefig('scatterplot.png')
print(predictions)

## display of confusion matrix and classification report of this end to end model
report(predictions,testY_one_hot,history)

## display of random input images and predicted test results along with their probabilities
show_test_end_to_end(model,testXModified,testY)
