function [X,Y] = regressTrainData()

data = importdata('regress_train.txt');

X = data(1,:);
Y = data(2,:);