import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

#Non-regularised Linear Regression
def multivarlinreg(X, y):
    step1 = np.linalg.inv(np.dot(np.transpose(X), X)) # (D+1)xN * Nx(D+1) --> (D+1)x(D+1)
    step2 = np.dot(step1, np.transpose(X))            # (D+1)x(D+1) * (D+1)xN --> (D+1)xN
    step3 = np.dot(step2, y)                          # (D+1)xN * Nx1 --> (D+1)x1

    wOut = np.transpose(step3)                        # (D+1)x1
    return wOut

#Regularised Linear Regression
def Regmultivarlinreg(X, y,lam):
    #step1 = np.linalg.inv(np.dot(np.transpose(X), X)+np.dot(np.identity(len(X[1])),len(X[1])*lam))
    #step2 = np.dot(step1, np.transpose(X))
    #step3 = np.dot(step2, y)

    #linalg.solve is better than linalg.inv (less rounding errors)
    A = np.dot(np.transpose(X), X)+np.dot(np.identity(len(X[1])),len(X[1])*lam)
    b = np.dot(np.transpose(X), y)
    solved = np.linalg.solve(A,b)

    wOut = np.transpose(solved)
    return wOut

#Leave One Out Cross Validation
def LOOCV (X,t,lam):
    X = np.c_[np.ones(len(X)), X]
    LOOL = 0
    for i in range(0,len(X)):
        XTemp = np.delete(X,i,0)
        tTemp = np.delete(t,i,0)
        w = Regmultivarlinreg(XTemp,tTemp,lam)
        LOOL += (t[i]-np.dot(np.transpose(w),X[i]))**2
    return LOOL/len(X)

#------------------------------------------------------------
#Below is an example, using the .txt file
#------------------------------------------------------------
data = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

X = data[:,0]                #Get the first column in the matrix (input)
XSq = np.c_[X,X**2]          #Add another column, 2nd power of 1st column
XThree = np.c_[XSq,X**3]     #Add another column, 3rd power of 1st column
XFour = np.c_[XThree,X**4]   #Add another column, 3rd power of 1st column
t = data[:,1]                #Get the second column in the matrix (output)

print("1st degree weights with lambda = 0:")
print(Regmultivarlinreg(np.c_[np.ones(len(X)), X],t,0))
j=(np.arange(1896., 2012, 1))
plt.plot(j, 3.64164559e+01 -1.33308857e-02*j, lw=2, label='Fitted polynomial')
plt.plot(X,t)
plt.title('1st degree polynomial fitted to data')
plt.ylabel('Time')
plt.xlabel('Year')
plt.plot(X,t, label='Data points connected', color='k')
plt.scatter(X,t, label='Data points', color='k')
plt.legend(loc='upper right')
plt.show()

print("4th degree weights with lambda = 0, 1, 0.0001156397, respectively:")
print(multivarlinreg(np.c_[np.ones(len(XFour)), XFour],t))
print(Regmultivarlinreg(np.c_[np.ones(len(XFour)), XFour],t,1))
print(Regmultivarlinreg(np.c_[np.ones(len(XFour)), XFour],t,0.0001156397))
plt.plot(j, 4.70486896e+05 -9.52076893e+02*j +7.22426138e-01*j**2 -2.43602743e-04*j**3 + 3.07996246e-08*j**4 , lw=2,label='Lambda = 0')
#plt.plot(j, 1.61286136e-04 + 7.13048020e-02*j + 3.42586043e-05*j**2 - 8.22180435e-08*j**3 + 2.42488415e-11*j**4)
plt.plot(j, 8.02021118e-06 + 3.90482702e-03*j + 1.37834316e-04*j**2 - 1.35264392e-07*j**3 + 3.33031062e-11*j**4, lw=2,label='Lambda = 1')
plt.plot(j, 1.86259631e-02 + 9.01750159e+00*j - 1.37136517e-02*j**2 + 6.95878189e-06*j**3 -1.17755099e-09*j**4, lw=2,label='Lambda = 0.0001156397')
plt.title('4th degree polynomials fitted to data')
plt.ylabel('Time')
plt.xlabel('Year')
plt.plot(X,t, label='Data points connected', color='k')
plt.scatter(X,t, label='Data points', color='k')
plt.ylim([9.5,12.1])
plt.legend(loc='upper right')
plt.show()

#------------------------------------------------------------
#1st Degree Polynomial - lambda in [0,1]
#------------------------------------------------------------
#Calculated the LOOCV loss for various lambda
lam = (np.arange(0., 1, 0.001))
x_out = []
for i in range(0,len(lam)):
    x_out.append(LOOCV(X,t,lam[i]))

#Plot of LOOCV loss vs lambda
plt.plot(lam,x_out)
plt.ylim([-0.01,0.6])
plt.xlim([-0.01,1.05])
plt.title('LOOCV(lambda)')
plt.ylabel('LOOCV')
plt.xlabel('Lambda')
plt.show()

#------------------------------------------------------------
#4st Degree Polynomial - lambda in [0,1]
#------------------------------------------------------------
#Calculated the LOOCV loss for various lambda
lam = (np.arange(0., 1, 0.001))
output = []
for i in range(0,len(lam)):
    temp = LOOCV(XFour,t,lam[i])
    output.append([lam[i], temp])
output = np.asarray(output)

#Plot of LOOCV loss vs lambda
plt.plot(output[:,0],output[:,1])
plt.ylim([0.054,0.06])
plt.xlim([-0.01,1.01])
plt.title('LOOCV(lambda)')
plt.ylabel('LOOCV')
plt.xlabel('Lambda')
plt.show()

#------------------------------------------------------------
#4st Degree Polynomial - lambda in [0,001]
#------------------------------------------------------------
#Calculated the LOOCV loss for various lambda
lam = (np.arange(0.0, 0.001, 0.00001))
output = []
for i in range(0,len(lam)):
    temp = LOOCV(XFour,t,lam[i])
    output.append([lam[i], temp])
output = np.asarray(output)

#Plot of LOOCV loss vs lambda
plt.plot(output[:,0],output[:,1])
plt.ylim([0.050,0.06])
plt.xlim([-0.00001,00.00101])
plt.title('LOOCV(lambda)')
plt.ylabel('LOOCV')
plt.xlabel('Lambda')
plt.show()
