import numpy as np
from homework1 import *

'''
Let n be the number of data points
and order be the number of basis terms

Y is a n x 1 column vector
X is a n x 1 column vector (? CONFIRM)
phi is a n x M matrix

Returns a M x 1 column vector
'''
def ridge(X, Y, order, l):
    phi = designMatrix(X,order)
    return np.linalg.inv(l*np.identity(order+1)+phi.T*phi)*phi.T*Y

##X,Y = bishopCurveData()
##print ridge(X,Y,3,0.01)

'''
X_train, Y_train is the dataset we use to find weights
X_val, Y_val is the dataset we use to calculate error and helps us find minimum value
X_test, Y_test is the dataset we use to calculate final error
mupper is the upper limit of the M or "order" values to test
lpower is a tuple of 2 values defining the range of lambda powers to test
    (e.g. specifying [1,3] would test 10^1, 10^2, 10^3)
returns tuple of (minimum error, m of min error, lambda of min error, test error with min parameters)
'''
def findOpt(X_train, Y_train, X_val, Y_val, X_test, Y_test, mupper, lpower):
    min_error = 10000
    min_m = None
    min_lambda = None
    min_w = None
    for i in range(mupper):
        for j in range(lpower[0], lpower[1]):
            w = ridge(X_train,Y_train,i,10**j)
            error = SSE(X_val, Y_val, i, w)
            #error = SSE(X_test, Y_test, i, w)
            if error <= min_error:
                min_error = error
                min_m = i
                min_lambda = j
                min_w = w
    test_error = SSE(X_test, Y_test, min_m, min_w)
    return (min_error, min_m, 10**min_lambda, test_error)

X_train,Y_train = regressTrainData()
X_test,Y_test = regressTestData()
X_val,Y_val = regressValidateData()

print findOpt(X_train, Y_train, X_val, Y_val, X_test, Y_test, 10, [-5,2])
#Minimum error occurs at M=4 and lambda=1 on the validation data
#Actual minimum SSE is M=1 and lambda=1 for the test data


#3.3 BlogFeedback Dataset
X_blogTest, Y_blogTest = regressBlogTestData()
X_blogVal, Y_blogVal = regressBlogValData()
X_blogTrain, Y_blogTrain = regressBlogTrainData()
'''
def ridgeSimple(X,Y,l):
    return np.linalg.inv(l*np.identity(len(X))+X.T*X)*X.T*Y

w=ridgeSimple(XBlogTrain, YBlogTrain,0.01)
'''

#4.1 Outlier and LAD
def LAD(X,Y,order,w):
    return np.linalg.norm((Y-designMatrix(X,order)*w),1);

def LADDer(X,Y,order,w):
    

order = 3;
f = lambda w:LAD(X,Y,order,w);
#g = lambda w: num_gradient(f,w,0.001);
g = lambda w: SSEDer(X,Y,order,w);
print decent(f,g,0.001,np.matrix([0,10,-20,20]).T,0.000001);

