# GPs-with-long-range-forecatsing

## MATLAB code for long-range-forecatsing using GPs.

The main file is runLSM.m, which loads data, initialize experiment setting, and calls all other functions. 

Below are instructions to use the code. As you go through the script, press ctrl+enter to run the code of the subsection.

## 1. Load up gpml

This section always needs to be run once at the start, as it gives access to the Gaussian Processes for Machine Learning (GPML) toolbox.

## 2. Load up data

There are four datasets: airline.mat, rail-miles.mat, monthly-electricity.mat, and pole_telecom.mat. For each dataset, the training inputs and outputs stored in 'xtrain' and 'ytrain', and the withheld testing inputs and outputs stored in 'xtest' and 'ytest'. 

## 3. Global experiment setting for all kernels
Q = 10; % the number of components
numinit = 5;    % initialization times
opt_sm = -1000;   % the optimizing times

## 4. Normalizing data
The users can normalize the data.

## 5. run benchmark kernels, such as SM, LKP, SLSM
the GP model with LKP has two inference methods: RJ-MCMC and exact inference. 

## 6. randinit is a function performing hyper-parameter initilization, component pruning, model inference, and evaluation. 

 6a. Find good initialization by fitting Gaussian/Laplace mixture of models on empirical spectral density.

 6b. Pruning components with weights smaller than 1.

 6c. Infer the GP model on large dataset by using robust BCM. For small dataset, exact inference is preferred.

 6d. Assess the performance of models using MSE and MAE.

 6e. Plot the posterior distribution and learned spectral density.
