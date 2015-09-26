import numpy as np;
import math;

def decent(f,g,step,guess,thresh):
    x = guess;
    previous = None;
    while(previous == None or abs(f(previous)-f(x)) > thresh):
        previous = x;
        x = x - step*g(x);
    return x;

def num_gradient(f,x,step):
    x = x.astype('float64');
    res = [0]*len(x)
    for i in range(len(x)):
        forward = np.array(x,copy=True);
        forward[i] = forward[i] + step/2.0;
        back = np.array(x,copy=True);
        back[i] = back[i]- step/2.0;
        res[i] = (f(forward)-f(back))/step;
    return np.array(res);

##f = lambda x: x**2;
##g = lambda x: 2*x;
##print decent(f,g,0.01, 100, 0.001)
##print g(5)
##print num_gradient(f,np.array([5.0]),0.0001);

##f = lambda x: x[0]*x[1];
##g = lambda x: np.array([x[1],x[0]]);
##print decent(f,g,0.01, np.array([-100,50]), 0.001)
##print g(np.array([-100,50])), num_gradient(f,np.array([-100.0,50.0]),0.00001);

##f = lambda x: (x[0]-5)**2+(x[1]+3)**2;
##g = lambda x: np.array([2*(x[0]-5),2*(x[1]+3)]);
##print decent(f,g,0.01, np.array([-100,50]), 0.00001)
##print g(np.array([3,3])), num_gradient(f,np.array([3,3]),0.000001);


###Multi minima
##f = lambda x: (x**2 + 3*x + 2)**2;
##g = lambda x: 2*(2*x + 3)*(x**2 + 3*x + 2);
##print decent(f,g,0.01, -1.5, 0.00001)
##print decent(f,g,0.01, 2, 0.00001)
##print decent(f,g,0.01, -3, 0.00001)
###print decent(f,g,0.01, -100, 0.00001) #crashes because unstable
##print decent(f,g,0.0001, -10, 0.00001)
##print g(np.array(-9)), num_gradient(f,np.array([-9]),0.01);


'''
def gaussian(x, mu, sig):
    return np.exp(-0.5*(x - mu)*np.linalg.inv(sig)*(x-mu).transpose())

def gen_gaussian(mu,sig):
    return lambda x: gaussian(x,mu,sig);

f = gen_gaussian(np.matrix([0,0]),np.matrix('1 0;0 1'));
g = np.gradient(f,);
print f(np.matrix([0,0]));
'''


'''
def gen_gaussian(u,o):
    def f(x):
        return (1.0/math.sqrt(2*math.pi*(o**2)))*math.exp(-(x-u)**2/(2*o**2));
    return f;
'''
#f = gen_gaussian(0,1);
#print f(0);
#print (np.array([0,0])-np.array([0,0]))*np.linalg.inv(np.matrix('1 0; 0 1'))*np.matrix([0,0]).transpose();

#print gaussian_m(np.matrix([0,0]),np.matrix([0,0]),np.matrix('1 0; 0 1'))
