from keras.layers import Input
from keras.models import Model 	
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.objectives import mean_squared_error
from keras.utils import np_utils,plot_model
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from report import report,show_test_Encoded,visualize
import matplotlib.pyplot as plt
from encoded_features import encoder,autoencoder,classifier_conv,decoder
from data_modification import data_modify
import keras
import numpy as np


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]


(trainX, trainY), (testX, testY) = cifar10.load_data()


trainXModified, trainYModified, testXModified, testYModified = data_modify(trainX, trainY,testX, testY)

trainX, validX, trainYf, validYf = train_test_split(trainXModified, trainYModified, test_size=0.2, random_state=42, shuffle= True)
trainY = keras.utils.to_categorical(trainYf, 10)
validY = keras.utils.to_categorical(validYf, 10)
testY_one_hot = keras.utils.to_categorical(testYModified, 10)


input = Input((32,32,3))


encoder, decoder = autoencoder()

decoder.compile(SGD(lr= 0.01, momentum= 0.9), loss='mse')
print(decoder.summary())
plot_model(decoder, to_file='model_autoencoder.png')


history = decoder.fit(trainX, trainX, batch_size=512, epochs=100, verbose=1, validation_data=(validX, validX),shuffle=True, callbacks=callbacks)


trainCodes = encoder.predict(trainX)
validCodes = encoder.predict(validX)
testCodes = encoder.predict(testXModified)
#plot_model(trainCodes, to_file='model_autoencoder.png')

model = classifier_conv(trainCodes)
for i in range(5):
    img = trainX[i]
    
    visualize(img,trainCodes[i])


model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr= 0.01, momentum= 0.9),metrics =  ['accuracy'])
model.summary()
plot_model(model, to_file='model_classifier.png')

history = model.fit(trainCodes, trainY, batch_size=512, epochs=100,  validation_data = (validCodes, validY), shuffle=True, callbacks=callbacks)

predictions = model.predict(testCodes)

test_codes_embedded = TSNE(n_components=2).fit_transform(predictions)
fig6 = plt.figure()
plt.scatter(test_codes_embedded[:,0], test_codes_embedded[:,1], c=testYModified, cmap=plt.get_cmap("tab10"))
plt.colorbar()
fig6.savefig('scatterplot_classifier.png')
print(predictions)


report(predictions,testY_one_hot,history)

show_test_Encoded(model, testCodes,testXModified,  testYModified)
