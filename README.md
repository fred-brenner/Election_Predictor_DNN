# Election Result Prediction using a Small Deep Neural Network

This project demonstrates how a small Deep Neural Network (DNN) can be used to predict election results based on two factors: population size and delegate probability. The project is divided into two parts: training on a small extract of the prediction graph (less than 1000 samples) and extrapolation through the trained model to predict a specific threshold.

## Model Architecture

The model architecture used in this project is simple yet effective for extrapolation. It consists of the following layers:

1. **Input Layer**: The input layers are splitted to give the delegate probability more room in the DNN.

2. **Hidden Layers**: The three hidden layers consist of Rectified Linear Units.

3. **Output Layer**: The output layer produces the election result from 0 to 1 and therefore is a sigmoid layer.

A figure of the model layout is included at the end.

## Usage

Install the necessary dependencies:
Tensorflow; Keras; Pandas; Numpy

## Results

Due to the diminutive differences in the training data, the Hyperparameters have to be tuned carefully. The included ones + automatic adjustement have been found to fit all current known files.

Also the training runs differ a bit. On account of this, all training runs are repeated several times to be sure to get the correct result.


Enjoy predicting election results with the power of deep learning!

## Model Layout

![dnn](https://github.com/fred-brenner/pablo_master/assets/24571823/4cdba4ac-5e01-4c54-a511-e4ea44f1caa5)
