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

def logKeq(parameters, variables):
    """
    #PY = c1*exp(b1*x1 - (b2)ln(x2))
    PY = c1*exp(b1*x1 - ln(x2) + b2)

    Try to model using the logarithm of Keq
    This is the only model i get out of equilibrium kinetics. The expression
    divides by x2 and then I take log(ln)
    """
    keq, rna_dna = variables

    if len(parameters) == 4:
        c1, b1, b2, b3 = parameters
        return c1*np.exp(b1*rna_dna - b2*np.log(b3*keq))

    if len(parameters) == 3:
        c1, b1, b2 = parameters
        return c1*np.exp(b1*rna_dna - b2*np.log(keq))

    elif len(parameters) == 2:
        c1, b1 = parameters
        return c1*np.exp(rna_dna - b1*np.log(keq))

def residuals_logKeq(parameters, variables, PY):

    return PY - logKeq(parameters, variables)

def logRT(parameters, variables):
    """
    PY = c1*exp(x1/RT + c2*log(x2))
    """
    rna_dna, keq = variables
    RT = 0.62

    if len(parameters) == 3:
        c1, c2, c3 = parameters
        return c1*np.exp(c2*rna_dna/RT - c3*np.log(keq))

    elif len(parameters) == 2:
        c1, c2 = parameters
        return c1*np.exp(c2*rna_dna/RT - np.log(keq))

    elif len(parameters) == 1:
        c1 = parameters[0]
        #return c1*np.exp(rna_dna/RT - np.log(keq))
        return c1*np.exp(rna_dna/RT - np.log(keq))


def residuals_logRT(parameters, variables, PY):

    return PY - logRT(parameters, variables)
