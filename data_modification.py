
import numpy as np

'''

Modification of data such that only 50% of the bird, deer and truck training dataset is used.
Rest is used as it is. Once the dataset is modified,it is further normalized.

'''

def data_modify(xtrain,ytrain,xtest,ytest):
   
    testX = []
    testY = []
    trainXFinal = []
    trainYFinal = []
    countLabel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, j in zip(xtrain, ytrain):
        if ((j==2)or(j==4)or(j==9)):
            #print(int(j))
            if(countLabel[int(j)]<2000):
                testX.append(i)
                testY.append(j)
                countLabel[int(j)]+=1
            else:
                trainXFinal.append(i)
                trainYFinal.append(j)
        else:
            trainXFinal.append(i)
            trainYFinal.append(j)
        
    testX  = np.array(testX)
    testY  = np.array(testY)

    trainXFinal = np.array(trainXFinal)
    trainYFinal = np.array(trainYFinal)
    testXFinal = np.append(testX, xtest, axis=0)
    testYFinal = np.append(testY, ytest, axis=0)



    trainXFinal = trainXFinal.astype('float32')
    testXFinal = testXFinal.astype('float32')
    trainXFinal = trainXFinal / 255
    testXFinal = testXFinal / 255
    return(trainXFinal,trainYFinal,testXFinal,testYFinal)

