import numpy as np
import matplotlib.pyplot as plt
from homework1 import *

'''
Let n be the number of data points
and order be the number of basis terms

Y is a n x 1 column vector
X is a n x 1 column vector 
phi is a n x M matrix

Returns a M x 1 column vector
'''
def ridge(X, Y, order, l):
    phi = designMatrix(X,order)
    return np.linalg.inv(l*np.identity(order+1)+phi.T*phi)*phi.T*Y

def prob31():
    X,Y = bishopCurveData()
    plt.figure(1)
    plt.subplot(1,2,1)
    #plots for different lambdas on Bishop
    #bishop points
    plt.scatter(X,Y, c=[[0,0.8,0.8]])

    #sin(2*pi*x)
    x = np.asmatrix(np.linspace(0,1,100)).T
    y = np.sin(2*np.pi*x)
    plt.plot(x,y,'r')

    order1 = 1
    w2 = ridge(X,Y,order1,1)
    w3 = ridge(X,Y,order1,0.1)
    w5 = ridge(X,Y,order1,0)

    plt.plot(x,designMatrix(x,order1)*w2,'b')
    plt.plot(x,designMatrix(x,order1)*w3,'m')
    plt.plot(x,designMatrix(x,order1)*w5,'k')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1,2,2)
    plt.scatter(X,Y, c=[[0,0.8,0.8]])

    #sin(2*pi*x)
    x = np.asmatrix(np.linspace(0,1,100)).T
    y = np.sin(2*np.pi*x)
    plt.plot(x,y,'r')

    order2 = 9

    w2 = ridge(X,Y,order2,1)
    w3 = ridge(X,Y,order2,0.1)
    w5 = ridge(X,Y,order2,0)

    plt.plot(x,designMatrix(x,order2)*w2,'b')
    plt.plot(x,designMatrix(x,order2)*w3,'m')
    plt.plot(x,designMatrix(x,order2)*w5,'k')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()
    return

'''
X_train, Y_train is the dataset we use to find weights
X_val, Y_val is the dataset we use to calculate error and helps us find minimum value
X_test, Y_test is the dataset we use to calculate final error
mupper is the upper limit of the M or "order" values to test
lpower is a tuple of 2 values defining the range of lambda powers to test
    (e.g. specifying [1,3] would test 10^1, 10^2, 10^3)
returns tuple of (minimum error, m of min error, lambda of min error, test error with min parameters, matrix of all errors)
'''
def findOpt(X_train, Y_train, X_val, Y_val, X_test, Y_test, mupper, lpower):
    min_error = 10000
    min_m = None
    min_lambda = None
    min_w = None
    error_full = []
    for i in mupper:
        error_mat = []
        for j in range(lpower[0], lpower[1]):
            w = ridge(X_train,Y_train, i,10**j)
            error = SSE(X_test, Y_test, i, w)
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

'''
get = findOpt(X_train, Y_train, X_val, Y_val, X_test, Y_test, [0,2,4], [-5,10])
errors = get[4]
a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for elem in errors:
    plt.plot(a,elem)
plt.xlabel("Log of Lambdas")
plt.ylabel("SSE")
plt.show()

for i in get[4]:
    print i
'''

#Minimum error occurs at M=4 and lambda=1 on the validation data
#Actual minimum SSE is M=1 and lambda=1 for the test data
def ridgeSimple(X,Y,l):
    print "ridgeSimple"
    X = np.asmatrix(X);
    Y = np.asmatrix(Y);
    return np.linalg.inv(l*np.identity(X.shape[1])+X.T*X)*X.T*Y.T

#modified SSE for linear model on blog data
def SSESimple(X,Y,w):
    print "SSESimple"
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)
    return 0.5*np.linalg.norm(Y.T-X*w,2)**2

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
def findOptSimple(X_train, Y_train, X_val, Y_val, X_test, Y_test, lpower):
    #increased min error since errors were so high
    min_error = 10000000000000000
    min_lambda = None
    min_w = None
    error_all = []
    for j in range(1,40,1):
        j=j/2.0
        w = ridgeSimple(X_train,Y_train, j)
        error = SSESimple(X_val, Y_val, w).item(0)
        error_all.append(error)
        print error
        if error <= min_error:
            min_error = error
            min_lambda = j
            min_w = w
    test_error = SSESimple(X_test, Y_test, min_w)
    print "test_error:"
    print test_error
    return (min_error, min_lambda, test_error, error_all)

'''
y = findOptSimple(X_blogTrain, Y_blogTrain, X_blogVal, Y_blogVal, X_blogTest, Y_blogTest, [-10,10])
x = range(1,40,1)
#y = np.sin(2*np.pi*x)
plt.plot(x,y[3],'r')

plt.xlabel('X')
plt.ylabel('Y')

plt.show()
'''

##order = 3;
##f = lambda w:LAD(X,Y,order,w);
###g = lambda w: num_gradient(f,w,0.001);
##g = lambda w: SSEDer(X,Y,order,w);
##print decent(f,g,0.001,np.matrix([0,10,-20,20]).T,0.000001);

#4.1 Outlier and LAD
def LAD(X,Y,order,w):
    return np.linalg.norm((Y-designMatrix(X,order)*w),1);

def LADError(X,Y,order,w,l):
    return LAD(X,Y,order,w) + l*np.linalg.norm(w,2)

# plot LAD curves

#X,Y = bishopCurveData()
def findOptimumPair(X_train, Y_train, X_val, Y_val, order_list, lambda_list):
    min_error = float("inf")
    min_order = None
    min_lambda = None
    error_all = []
    for m in order_list:
        error_list = []
        for l in lambda_list:
            f = lambda w: LADError(X_train,Y_train,m,np.matrix(w).T,l)
            w = fmin_bfgs(f,[1]*(m+1))
            error = SSE(X_val, Y_val, m, np.matrix(w).T)
            error_list.append(error)
            if error <= min_error:
                min_error = error
                min_order = m
                min_lambda = l
        error_all.append(error_list)
        error_list = []
    return (min_error, min_order, min_lambda, error_all)
                
#min_error, min_order, min_lambda, error_all = findOptimumPair(X_train, Y_train, X_val, Y_val, range(10),[0.01, 0.1, 0.5, 1])

order = 10;
l = 0.1
f = lambda w: LADError(X_train,Y_train,order,np.matrix(w).T,l);
w = fmin_bfgs(f,[1]*(order+1))

plt.scatter(X_val,Y_val, c=[[0,0.8,0.8]])

x = np.asmatrix(np.linspace(-2.6,2,100)).T
#y = np.sin(2*np.pi*x)
#plt.plot(x,y,'r')

plt.plot(x,designMatrix(x,order)*(np.matrix(w).T),'k')

plt.xlabel('X')
plt.ylabel('Y')

plt.show()

