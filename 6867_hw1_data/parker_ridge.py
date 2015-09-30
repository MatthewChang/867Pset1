import numpy as np
import matplotlib.pyplot as plt
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

X,Y = bishopCurveData()
print ridge(X,Y,3,0.01)
plt.figure(1)
plt.subplot(1,2,1)
#plots for different lambdas on Bishop
#bishop points
plt.scatter(X,Y, c=[[0,0.8,0.8]])

#sin(2*pi*x)
x = np.linspace(0,1,100)
y = np.sin(2*np.pi*x)
plt.plot(x,y,'r')

w2 = ridge(X,Y,3,0.1)
w3 = ridge(X,Y,3,0.01)
w5 = ridge(X,Y,3,0)

plt.plot(X,designMatrix(X,3)*w2,'b')
plt.plot(X,designMatrix(X,3)*w3,'m')
plt.plot(X,designMatrix(X,3)*w5,'k')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1,2,2)
plt.scatter(X,Y, c=[[0,0.8,0.8]])

#sin(2*pi*x)
x = np.linspace(0,1,100)
y = np.sin(2*np.pi*x)
plt.plot(x,y,'r')

w2 = ridge(X,Y,9,0.1)
w3 = ridge(X,Y,9,0.01)
w5 = ridge(X,Y,9,0)

plt.plot(X,designMatrix(X,9)*w2,'b')
plt.plot(X,designMatrix(X,9)*w3,'m')
plt.plot(X,designMatrix(X,9)*w5,'k')

plt.xlabel('X')
plt.ylabel('Y')

plt.show()

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
    error_full = []
    for i in range(2,mupper):
        error_mat = []
        for j in range(lpower[0], lpower[1]):
            print "("+str(i)+","+str(j)+")"
            w = ridge(X_train,Y_train, i,10**j)
            error = SSE(X_val, Y_val, i, w)
            error_mat.append(error)
            if error <= min_error:
                min_error = error
                min_m = i
                min_lambda = j
                min_w = w
        error_full.append(error_mat)
        error_mat = []
    test_error = SSE(X_test, Y_test, min_m, min_w)
    return (min_error, min_m, 10**min_lambda, test_error, error_full)

X_train,Y_train = regressTrainData()
X_test,Y_test = regressTestData()
X_val,Y_val = regressValidateData()

#get = findOpt(X_train, Y_train, X_val, Y_val, X_test, Y_test, 5, [-5,10])
#errors = get[4]
#a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#for elem in errors:
#    plt.plot(a,elem)
#plt.xlabel("Log of Lambdas")
#plt.ylabel("SSE")
#plt.show()

#Minimum error occurs at M=4 and lambda=1 on the validation data
#Actual minimum SSE is M=1 and lambda=1 for the test data

#4.1 Outlier and LAD
def LAD(X,Y,order,w):
    return np.linalg.norm((Y-designMatrix(X,order)*w),1);

def LADDer(X,Y,order,w):
    return 1

def ridgeSimple(X,Y,l):
    X = np.asmatrix(X);
    Y = np.asmatrix(Y);
    return np.linalg.inv(l*np.identity(X.shape[1])+X.T*X)*X.T*Y.T

#3.3 BlogFeedback Dataset
X_blogTest, Y_blogTest = regressBlogTestData()
X_blogVal, Y_blogVal = regressBlogValData()
X_blogTrain, Y_blogTrain = regressBlogTrainData()
#preprocessing for feature scaling - takes in matrix and scales based off of norm of feature vector
def featureScaling(X):
    X = np.asmatrix(X)
    X_t = X.T
    for i in range(X_t.shape[0]):
        norm = np.linalg.norm(X_t[i],1)
        if norm == 0: norm = 0.0001
        X_t[i] = X_t[i]/norm
    return X_t.T

X_blogTest = featureScaling(X_blogTest)
X_blogVal = featureScaling(X_blogVal)
X_blogTrain = featureScaling(X_blogTrain)
#modified opt for linear model on blog data
def findOptSimple(X_train, Y_train, X_val, Y_val, X_test, Y_test,lpower):
    #increased min error since errors were so high
    min_error = 10000000000000000
    min_lambda = None
    min_w = None
    for j in [0.5,0.2,0.1,1,2,9,10, 20]:
        w = ridgeSimple(X_train,Y_train, j)
        error = SSESimple(X_val, Y_val, w)
        print error
        if error <= min_error:
            min_error = error
            min_lambda = j
            min_w = w
    test_error = SSESimple(X_test, Y_test, min_w)
    return (min_error, min_lambda, test_error)
#modified SSE for linear model on blog data
def SSESimple(X,Y,w):
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)
    return np.linalg.norm(Y.T-X*w,2)**2
#print findOptSimple(X_blogTrain, Y_blogTrain, X_blogVal, Y_blogVal, X_blogTest, Y_blogTest, [-10,10])


##order = 3;
##f = lambda w:LAD(X,Y,order,w);
###g = lambda w: num_gradient(f,w,0.001);
##g = lambda w: SSEDer(X,Y,order,w);
##print decent(f,g,0.001,np.matrix([0,10,-20,20]).T,0.000001);