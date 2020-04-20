from keras.objectives import mean_squared_error
import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dict = {0: 'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}

def report(predictions,y_test_one_hot,history): ## function used for creating a classification report and confusion matrix
    matrix=confusion_matrix(y_test_one_hot.argmax(axis=1), predictions.argmax(axis=1))
    print("Classification Report:\n")
    report=classification_report(y_test_one_hot.argmax(axis=1),predictions.argmax(axis=1),target_names=list(dict.values()))
    print(report)
    
    fig1 = plt.figure()

    sns.heatmap(matrix, annot=True, xticklabels = list(dict.values()), yticklabels = list(dict.values()), fmt="d")
    fig1.savefig('heatmap_end_to_end.png')
    fig2 = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig2.savefig('accuracy_seperate_classifier.png')
    fig3 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig3.savefig('loss_seperate_classifier.png')
    
def visualize(img,code):
    """Draws original, encoded and decoded images"""
    #code = encoder.predict(img[None])[0]  					# img[None] is the same as img[np.newaxis, :]
    
    fig5 = plt.figure()
    ax = fig5.add_subplot(2, 2, 1)
    ax.grid('off')
    ax.axis('off')
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.title("Original")
    
    ax = fig5.add_subplot(2, 1, 2)
    ax.grid('off')
    ax.axis('off')

    plt.title("Representational Code")
    
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    
    fig5.savefig('image_with_codes.png')
    plt.show()


def show_test_end_to_end(m, d, y_test_final): 					## function used for visualizing the predicted and true labels of test data
    cols = 10
    rows = 2
    fig4 = plt.figure(figsize=(2 * cols - 1, 4 * rows - 1))
    print(m)
    for i in range(cols):
    	for j in range(rows):
            random_index = np.random.randint(0, len(y_test_final))
            ax = fig4.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            test_image = np.expand_dims(d[random_index], axis=0)
            test_result = m.predict(test_image)
                    
            plt.imshow(d[random_index])
            index = np.argsort(test_result[0,:])
            ax.set_title("Pred:{}\nProb:{:.3}\nTrue:{}\n".format(dict[index[9]],test_result[0,index[9]], dict[y_test_final[random_index][0]]))
    fig4.show()
    fig4.savefig('sample_output_classic_conv.png')


def show_test_Encoded(m,features, d, y_test_final):  				## function used for visualizing the predicted and true labels of test data
    cols = 10
    rows = 2
    fig4 = plt.figure(figsize=(2 * cols - 1, 4 * rows - 1))
    print(m)
    for i in range(cols):
    	for j in range(rows):
            random_index = np.random.randint(0, len(y_test_final))
            ax = fig4.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            test_image = np.expand_dims(d[random_index], axis=0)
            test_features=np.expand_dims(features[random_index],axis=0)
            test_result = m.predict(test_features)
                    
            plt.imshow(d[random_index])
            index = np.argsort(test_result[0,:])
            ax.set_title("Pred:{}\nProb:{:.3}\nTrue:{}\n".format(dict[index[9]],test_result[0,index[9]], dict[y_test_final[random_index][0]]))
    fig4.show()
    fig4.savefig('sample_output_seperate_classifier.png')






