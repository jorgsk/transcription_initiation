## perform least squares fit
#    A_final, cov_x, infodict, msg, ier = leastsq(residuals, x0, args=param, full_output=True, warning=True)
##    if ier != 1:
##        print "No fit!"
##        print msg
##        return
#    y_final = exponential(A_final, x_indep)
#    chi2 = sum((y_dep-y_final)**2 / y_final)

## print resulting parameters and their std. deviations
#    print "Optimized parameters:"
#    for i in range(len(A_final)):
#        print 'nothing'
##        print "A[%d]  =%8.3f +- %.4f" % (i, A_final[i], np.sqrt(cov_x[i,i]))
#    print "chi^2 =", chi2
#    1/0

# This script fits the mean and +/- standard deviations of the incoming values
# to an exponential function and plots the resulting curves.

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def residuals(p,y,x):
    err = y-(p[0] + p[1]*np.exp(-p[2]*x))
    return err

def peval(x, p):
    return p[0] + p[1]*np.exp(-p[2]*x)

def Expfit(x_ind,y_dep, stdv):
    n = 2
    x_ind = np.array(x_ind)
    y_dep = np.array(y_dep)
    stmin = np.array(y_dep) - n*np.array(stdv)
    stmax = np.array(y_dep) + n*np.array(stdv)
    p0 = np.array([0.5, 11, -1])
    meanlsq, cov_x, infodict, msg, ier = leastsq(residuals, p0, args=(y_dep,
                                                                      x_ind),
                                                 full_output=True, warning=True)
    y_est = peval(x_ind,meanlsq)
    chi2 = sum((y_dep-y_est)**2 / y_est)
    print "Optimized parameters:"
    for i in range(len(meanlsq)):
        print 'nothing'
        print "A[%d]  =%8.3f +- %.4f" % (i, meanlsq[i], np.sqrt(cov_x[i,i]))
    print "chi^2 =", chi2
    minlsq = leastsq(residuals, p0, args=(stmin,x_ind))
    maxlsq = leastsq(residuals, p0, args=(stmax,x_ind))
    min_est = peval(x_ind,minlsq[0])
    max_est = peval(x_ind,maxlsq[0])
    plt.plot(x_ind,y_est,'*',x_ind,min_est,'s',x_ind,max_est,'>',x_ind,y_dep,'o')
    plt.xlabel('Normalized DNA/RNA binding energy', fontsize=20)
    plt.ylabel('Original PY and extrapolated PY', fontsize=20)
    plt.title(r'Least-squares fit of RNA/DNA data to '
              '$a_0+a_1\exp(-a_2 x)$ ', fontsize=20)
    plt.legend(['Mean', 'SD minus','SD plus','original PY'])
    plt.show()
