---
title: "ReadMe"
author: "Aashay Sharma"
date: "17/08/2020"
output: 
  html_document:
        keed_md: true
---

## Objective 

#### Here we need to analyse and predict the house value situated in US, by using the data provided. We need to check whether several factors like total number of rooms, or households have any significant corelation with median house prices and after exploratory analysis we need to suggest and apply optimum model for the data for further predictions of the median house prices.

## Data 
The data can be downloaded from
[here](https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.csv)

## Neural Network Approach 

### Multilayer Perceptron Model with 3 Layers (excluding input layer)

#### NN Architecture:

   INPUT  LAYER RELU     120 UNITS
1# HIDDEN LAYER SOFTPLUS 60  UNITS
2# HIDDEN LAYER SOFTPLUS 30  UNITS
3# OUTPUT LAYER SOFTPLUS 1   UNIT

OPTIIMIZER    : RMS PROP
LEARNING RATE : 0.001
LOSS FUNCTION : ROOT_MEAN_SQUARED
NA_VALUES     : OMITTED

## Results :

![Train Set and Validation Set Error](/Users/aashaysharma/Desktop/RStudio/Housing-Data-Analysis/screenshot/Screenshot 2020-08-17 at 9.07.48 AM.png)
![Test Set Error and Predicted Example Sample Output](/Users/aashaysharma/Desktop/RStudio/Housing-Data-Analysis/screenshot/Screenshot 2020-08-17 at 9.08.13 AM.png)
