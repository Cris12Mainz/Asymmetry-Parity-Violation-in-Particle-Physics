import numpy as np
import matplotlib.pyplot as plt
import math as m
import csv
from random import gauss
from scipy.optimize import fmin,fmin_cg,curve_fit
from scipy import misc
from sympy import diff
from sympy import symbols
from sympy import lambdify
from sympy import evalf
import scipy as sp
from sympy import symbols, Matrix, Function, simplify, exp, hessian, solve, init_printing
init_printing()
from sympy import lambdify

file = open('solid.csv')
csvreader = csv.reader(file)
x = []
Q2 = []
APV_err = []
for row in csvreader:
    x.append(float(row[0]))
    Q2.append(float(row[1]))
    APV_err.append(float(row[2])/100)  
    
x = np.array(x)
Q2 = np.array(Q2) 
APV_err = np.array(APV_err)

# Here I will consider the statistical errors given by the paper
# as statistical error from Adpv since I will assume that we are using SM + something.

def err(s):
    
    y = np.full( (15,15) , s**2 )
    
    return y

# We defined an systematic error matrix considering the same correlated error for all 
# the measurements and symmetric.
    
def Apvew(Q2,x,a1,a3): #Apv from the Standard Model, Ekectroweak sector
    
    mp = 0.93828 #GeV
    s = 11 #GeV
    GF = 1.1663787*(10**-5) #GeV^-2
    sc = 0.007297351 #constant
    coeff = ( GF*Q2 )/( 4*np.sqrt(2)*np.pi*sc ) # coefficient 
    y = ( Q2/x )*( 1/( s-mp**2 ) ) # from deep inelastic scattering relations 
    z = coeff*(a1 + a3*( (1 - (1-y)**2 )/( 1 + (1-y)**2) ) )
    
    return z

a10 = 0.864   
a30 = 0.48 

# Values for a1 and a3 from the Standard Model from Solid paper

i = 0
MVpvew = [] 
for i in range(15):
    MVpvew.append( Apvew( Q2[i] , x[i] , a10 , a30 ) )

MVpvew = np.array(MVpvew)

#print(MVpvew) ### Possible error calculating Apvew not found yet
#print(APV_err) #####Warning

#we compute the mean value of APV with the values of C1's and C2's given 
#in the SOLID paper

#plt.hist(MVpvew,5)  

APV_err = np.array(MVpvew*APV_err)

#print(APV_err)

# We print the values for our errors so we can see the order of magnitude

squared_APV_err = [APV_err ** 2 for APV_err in APV_err]

# We square the error to put them as the Standard deviation squared in the covariance matrix

#print(APV_err[0])
#print(np.sqrt(squared_APV_err[0]))

stat = np.diag( squared_APV_err ) 

#print(stat)

# We form a diagonal matrix with the error given by the paper squared. Here we consider the errors as gaussian. 
# Therefore we are using them as the Standard Deviation squared.

i = 0 
Adpv = []
for i in range(15):
     Adpv.append( gauss(MVpvew[i], APV_err[i])) #New way of defining our simulated values
     #Adpv.append( MVpvew[i] + APV_err[i])  ######Warning!
     
#print(gauss(0, MVpvew[0]*APV_err[0]))     

Adpv = np.array(Adpv)    

#print(Adpv)

# possible error is that we are generating Adpv with Apvew + error but these errors are sometimes bigger than
# the values of Apvew that change the sign of the entries.
   
# Here we are going to simulate the possible values for Adpv as the value of the SM plus
# a value given by random Gaussian values generated with a mean zero and standard deviation
# equation to the error given in the Solid paper Adpv = Apvew + delta Adpv 

def Adpvf( Q2 , x , a1 , a3 , bht, bcsv ):
    
    x1 = 1/((1-x)**3)
    
    x2 = 1 + bht*x1*(1/Q2) + bcsv*(x**2)
    
    x3 =  Apvew(Q2,x,a1,a3)*x2

    return x3 

def cov(s):
    
    y = stat + err(s)
    
    return y

# Here we define the covariance matrix between the measurements where in the diagonal entries
# We find the elements to be like sigma^2 + s^2 , that is, statistical plus systematic
# While in the off-diagonal terms we find the terms to be s^2, systematic errors squared.
#print(stat + err(1))
    
def incov(s):
    
    y = np.linalg.inv( cov(s) )
    
    return y

#print(icov(0))

# Here we define the inverse of our covariance matrix
    
def chi2(bht , bcsv , a1 , a3, s ):
    
    y = Adpv-Adpvf(Q2,x,a1,a3,bht,bcsv)
    p = 0
    
    for i in range(15):
        for j in range(15):
            p += y[i]*incov(s)[i,j]*y[j]
            
    return p

# We have defined our /Chi^2 function, summing over all the elements of the matrix (y.T,V^-1,y) 
# with y the difference between measured data and the fit function, while V is the covariance matrix

#print(chi2(1,1,1,.5,.6))    
    
s=0

def best(cbht , cbcsv , ca1 , ca3 , s):
    
    bestbht,bestbcsv,besta1,besta3 =fmin(lambda x,x1: chi2(x[0],x[1],x[2],x[3],x1),
                                           [cbht,cbcsv,ca1,ca3],args=(s,))
    
    p = bestbht,bestbcsv,besta1,besta3
    
    chi2_min = chi2(bestbht,bestbcsv,besta1,besta3,s)

    print(f" The best value for bht is: {bestbht}")
    
    print(f" The best value for bcsv is: {bestbcsv}")
    
    print(f" The best value for bcsv is: {besta1}")
    
    print(f" The best value for bcsv is: {besta3}")
        
    print(f" The value of chi2_min is: {chi2_min}")
     
    
    return p
    
bvalue = best(0.,0.,a10,a30,s)

#bbht = best(0.,0.,a10,a30,s)[0]
#bbcsv = best(0.,0.,a10,a30,s)[1]
#ba1 = best(0.,0.,a10,a30,s)[2]
#ba3 = best(0.,0.,a10,a30,s)[3]


# Here we have minimized our chi2 function for bht and bcsv, fixing a1 and a3 to the SM values.
# We have use as central values bht=0 and bcsv=0.
# We are allow to use this code to find the best value for bht and bcsv.


# Here we have defined our fit function as the expression proposed in the Solid Paper
# We can see that we can leave open the possibility for corrections to a1 and a3
# In case we just want to fit bht and bcsv we will set a1 and a3 equal to the SM values.


x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
f, g, h = symbols('f g h', cls=Function)

X = Matrix([x1,x2,x3,x4])

def chi2(x1,x2,x3,x4):
    
    y = Adpv-Adpvf(Q2,x,x4,x3,x1,x2)
    p = 0
    
    for i in range(15):
        for j in range(15):
            p += y[i]*incov(0)[i,j]*y[j]
            
    return Matrix([p])
    
h = simplify(hessian(chi2(x1,x2,x3,x4), X))
s = (x1, x2, x3, x4)
g_func = lambdify(s, h, modules='numpy')

#U = np.linalg.inv(0.5*g_func(bvalue[0],bvalue[1],bvalue[2],bvalue[3]))
print(f"The covariance matrix is: {np.matrix(np.linalg.inv(0.5*g_func(0,0,.7,.5)))}")
#plt.imshow(np.log(np.abs(U)))
#plt.colorbar()
#plt.title('Covariance matrix fitted parameters', fontweight ="bold")
#plt.plot()
#plt.savefig('covarince matrix.png', dpi=600)
print(f"The error of bht , bcsv, a1 , a3 respectively are : {np.sqrt(np.diag(U))}")
    
