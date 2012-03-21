"""
Define as set of models to optimize toward
"""
import numpy as np

# MAIN MODEL (which is kind of wrong :S because Keq can't be considered that
# way. But maybe you can treat k1 or k_1 as arrhenius and then get away with it.
def mainM(parameters, variables):
    """
    PY = c1*exp(b1*x1 + b2*x2 + ... + bn*xn)
    """
    c1 = parameters[0]
    # note, not dot product, more like
    if len(parameters) > 2:
        return c1*np.exp(np.dot(parameters[1:], variables))
    else:
        return c1*np.exp(parameters[1]*variables)

def residuals_mainM(parameters, variables, PY):

    return PY - mainM(parameters, variables)
