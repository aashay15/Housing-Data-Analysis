---
title: "ReadMe"
author: "Aashay Sharma"
date: "17/08/2020"
---

## Objective 

#### Here we need to analyse and predict the house value situated in US, by using the data provided. We need to check whether several factors like total number of rooms, or households have any significant corelation with median house prices and after exploratory analysis we need to suggest and apply optimum model for the data for further predictions of the median house prices.

## Data 
The data can be downloaded from
[here](https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.csv)

## Neural Network Approach 

### Multilayer Perceptron Model with 3 Layers (excluding input layer)

#### NN Architecture:

   INPUT  LAYER RELU     128 UNITS.     
1# HIDDEN LAYER RELU      64 UNITS.  
2# HIDDEN LAYER RELU      32 UNITS. 
3# OUTPUT LAYER            1   UNIT (regression problem so linear output ). 

OPTIIMIZER    : RMS PROP. 
LEARNING RATE : 0.001. 
LOSS FUNCTION : Mean Squared Error. 
EVALUATION    : Mean Absolute Error. 
NA_VALUES     : OMITTED. 

## Results :

The below errors are the errors after training the model on normalized data rather than original data, as the orgininal data was very much variate in terms of ditribution as well as mean and so normalization was a good option in this particular problem. 

TRAIN SET ERROR      : around 0.2901. 
VALIDATION SET ERROR : around 0.3101. 
TEST SET ERROR       : around 0.3208.   

The model does not overfit a lot and thus there is no significant problem from high variance and the model predictions are off by around $30,000 (plus or minus) and so it is more of a bias problem but that too can be assumed as a unneccesary issue as house values depend on many much properties like crime rate in that particular area, surroundings, Builder or company brand by which the house was built and much more which are not included in this particular data set and thus it limits our observations, but with the given data we can still make our predictions using this model (with some real world complications and errors). 



