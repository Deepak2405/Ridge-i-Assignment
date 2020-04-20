from keras.layers import Input
from keras.models import Model 	
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.objectives import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils,plot_model
from sklearn.model_selection import train_test_split
from encoded_features import classic_conv
from sklearn.manifold import TSNE
from report import report,show_test_end_to_end
from data_modification import data_modify
import matplotlib.pyplot as plt
import keras.backend as K
import keras
import numpy as np


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainXModified, trainYModified,testXModified,testYModified = data_modify(trainX, trainY,testX, testY)

trainX, validX, trainYf, validYf = train_test_split(trainXModified, trainYModified, test_size=0.2, random_state=42, shuffle= True)
trainY = keras.utils.to_categorical(trainYf, 10)
validY = keras.utils.to_categorical(validYf, 10)
testY_one_hot = keras.utils.to_categorical(testYModified, 10)



model = classic_conv()
model.compile(loss = 'categorical_crossentropy', 
                  loss_weights = {'classification': 0.9}, 
                  optimizer = SGD(lr= 0.01, momentum= 0.9),
                  metrics = {'classification': 'accuracy'})




aug = ImageDataGenerator(rotation_range=10, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest")


print(model.summary())

plot_model(model, to_file='model_classic_conv.png')

history= model.fit_generator(aug.flow(trainX, trainY, batch_size=512),
	validation_data=(validX, validY), steps_per_epoch=len(trainX) // 512,
	epochs=100)


predictions = model.predict(testXModified)#[1]
test_codes_embedded = TSNE(n_components=2).fit_transform(predictions)
fig6 = plt.figure()
plt.scatter(test_codes_embedded[:,0], test_codes_embedded[:,1], c=testYModified, cmap=plt.get_cmap("tab10"))
plt.colorbar()
fig6.savefig('scatterplot_classic_conv.png')
print(predictions)

report(predictions,testY_one_hot,history)
show_test_end_to_end(model,testXModified,testY)
