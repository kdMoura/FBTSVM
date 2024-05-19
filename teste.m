% Load the dataset
addpath(genpath('/home/livia/Downloads/breast+cancer+wisconsin+diagnostic'));
% Load the .mat file
data = load('data.mat');

% Access the variable containing the data
data_variable = data.data;

% Now you can use data_variable in your MATLAB code