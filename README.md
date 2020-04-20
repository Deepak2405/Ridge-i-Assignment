# Supervised-Unsupervised Training 

Using Autoencoders for feature extraction on CIFAR-10 dataset and passing the encoded features to image classifier network. Two types of architecture models have been trained.

1. End to end model

   This model combines encoder module of autoencoder with image classifer by avoiding the decoder module of autoencoder. Thus, the generated code of encoder is passed to image classifier. 

2. Seperate-training model

In this model, the autoencoders are trained first and then the encoded data is fed to image classifier network.


## How to use

To run end-to-end model , type
```bash
python3 endtoend.py
```
\
**a. Confusion Matrix Heat Map**\
![alt text](https://github.com/Deepak2405/Ridge-i-Assignment/blob/master/images/heatmap_end_to_end.png)
\
**b. Random Image and its predicted Class**\
![alt text](https://github.com/Deepak2405/Ridge-i-Assignment/blob/master/images/sample_output_end_to_end.png)
\
**c. Model accuracy and Loss**\
![alt-text-1](https://github.com/Deepak2405/Ridge-i-Assignment/blob/master/images/accuracy_end_to_end.png "Model Accuracy") \
![alt-text-2](https://github.com/Deepak2405/Ridge-i-Assignment/blob/master/images/loss_end_to_end.png "Model Loss")
\
To run seperate-training model , type
```bash
python3 seperate-classifier.py
```
To run simple convolution neural network model , type
```bash
python3 classic-conv.py
```



