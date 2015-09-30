import pdb
import random
import pylab as pl
import numpy as np
from numpy import genfromtxt
from scipy.optimize import fmin_bfgs
from Descent import decent, num_gradient

# X is an array of N data points (one dimensional for now), that is, Nx1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def designMatrix(X,order):
    Z = X.T.tolist()[0]
    return np.matrix([ [x**r for r in range(0,order+1)]  for x in Z]);

def SSE(X,Y,order,w):
    return np.linalg.norm((Y-designMatrix(X,order)*w),2)**2;

def SSEDer(X,Y,order,w):
    s = [0]*(order+1);
    phi = designMatrix(X,order);
    for d in range(0,order+1):
        for i in range(len(X)):
            s[d] += 2*(Y[i]-phi[i]*w)*(-phi[i,d]);
    s = [x[0,0] for x in s]
    return np.matrix(s).T;

def regressionFit(X,Y,phi):
    return np.linalg.inv((phi.T*phi))*phi.T*Y

def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    print 'w', w

    print SSE(X,Y,order,w);
    #print SSEDer(X,Y,order,w);
    #print num_gradient(lambda w: SSE(X,Y,order,w),w,0.001);
    
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    A=  np.matrix(pts)
    Yp = pl.dot(w.T, designMatrix(A, order).T)
    pl.plot(pts, Yp.tolist()[0])

def regressionPlotDescent(X, Y, order, guess):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    wo = regressionFit(X, Y, phi)
    print 'optimal w', wo
    f = lambda w:SSE(X,Y,order,w);
    g = lambda w: np.matrix(num_gradient(f,w,0.001)).T;
    w = decent(f,g,0.01,guess,0.0001);

    print 'descent w', w

    print SSE(X,Y,order,w);
    #print SSEDer(X,Y,order,w);
    #print num_gradient(lambda w: SSE(X,Y,order,w),w,0.001);
    
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    A=  np.matrix(pts)
    Yp = pl.dot(w.T, designMatrix(A, order).T)
    pl.plot(pts, Yp.tolist()[0])

def regressionPlotDescentBuiltin(X, Y, order, guess):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    wo = regressionFit(X, Y, phi)
    print 'optimal w', wo
    def f(w):
        #print np.matrix(w).T
        return SSE(X,Y,order,np.matrix(w).T)
    w = fmin_bfgs(f,guess);
    w = np.matrix(w).T

    print 'descent w', w

    print SSE(X,Y,order,w);
    #print SSEDer(X,Y,order,w);
    #print num_gradient(lambda w: SSE(X,Y,order,w),w,0.001);
    
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    A=  np.matrix(pts)
    Yp = pl.dot(w.T, designMatrix(A, order).T)
    pl.plot(pts, Yp.tolist()[0])
   

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressTrainData():
    return getData('regress_train.txt')

def regressValidateData():
    return getData('regress_validate.txt')

def regressTestData():
    return getData('regress_test.txt')

def regressBlogTestData():
    return (genfromtxt('BlogFeedback_data/x_test.csv'), genfromtxt('BlogFeedback_data/y_test.csv'))

def regressBlogValData():
    return (genfromtxt('BlogFeedback_data/x_val.csv'), genfromtxt('BlogFeedback_data/y_val.csv'))

def regressBlogTrainData():
    return (genfromtxt('BlogFeedback_data/x_train.csv'), genfromtxt('BlogFeedback_data/y_train.csv'))

X , Y = bishopCurveData();
##regressionPlot(X,Y,0);
##regressionPlot(X,Y,1);
##regressionPlot(X,Y,3);
##regressionPlot(X,Y,9);
##pl.show()

regressionPlotDescent(X,Y,0,np.matrix([1]*1).T);
regressionPlotDescent(X,Y,1,np.matrix([1]*2).T);
regressionPlotDescent(X,Y,3,np.matrix([0]*4).T);
regressionPlotDescent(X,Y,9,np.matrix([1]*10).T);

##regressionPlotDescentBuiltin(X,Y,0,np.matrix([1]*1).T);
##regressionPlotDescentBuiltin(X,Y,1,np.matrix([1]*2).T);
##regressionPlotDescentBuiltin(X,Y,3,np.matrix([0]*4).T);
##regressionPlotDescentBuiltin(X,Y,9,np.matrix([1]*10).T);
pl.show();


##order = 1;
##f = lambda w:SSE(X,Y,order,w);
##g = lambda w: SSEDer(X,Y,order,w);
##w = np.matrix([100,10]).T
###print np.linalg.norm(g(w) - np.matrix(num_gradient(f,w,0.01)).T,2);
##print decent(f,g,0.001,np.matrix([0,1]).T,0.000001);

