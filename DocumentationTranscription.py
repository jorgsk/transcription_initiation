""" Calculate correlations between RNA/DNA and DNA/DNA binding energies and PY. """
# NOTE this file is a simplified version of 'Transcription.py' to generate data for the Hsu-paper

# Python modules
from __future__ import division
# You need future division even in numpy
import os
import re
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import scipy.stats
from scipy import optimize
from scipy.stats import spearmanr, pearsonr
import scipy.interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
import operator
from matplotlib import rc
import itertools
import time
# My own modules
import Energycalc as Ec
import Workhouse
import Filereader
import Models
from glob import glob
import time

from operator import itemgetter

import multiprocessing

#import transformations as transf

# Make numpy easier to read
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

from numpy import dot

def run_from_ipython():
    try:
        __IPYTHON__active
        return True
    except NameError:
        return False

if run_from_ipython():
    from IPython.Debugger import Tracer
    #from IPython.core.debugger import Tracer
    debug = Tracer()
else:
    def debug():
        pass

#matplotlib.rc('text', usetex=True)  # Using latex in labels in plot
#matplotlib.rc('font', family='serif')  # Setting font family in plot text

# Locations of input data
hsu1 = '/Hsu/csvHsu'
hsu2 = '/Hsu/csvHsuNewPY'
hsu3 = '/Hsu/csvHsuOmit3'
hsu4 = '/Hsu/csvHsu2008'
hsu5 = '/Hsu/csvHsu2008_full'

# Figure directory for rna_dna analysis
# The path to the directory the script is located in
here = os.path.dirname(os.path.realpath(__file__))
fig_dir1 = os.path.join(here, 'figures')
#fig_dir2 = '/home/jorgsk/phdproject/The-Tome/my_papers/rna-dna-paper/figures'
fig_dir2 = '/home/jorgsk/Dropbox/The-Tome/my_papers/rna-dna-paper/figures'

fig_dirs = (fig_dir1, fig_dir2)

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

def welchs_approximate_ttest(n1, mean1, sem1, n2, mean2, sem2, alpha):
    """
    Got this from the scipy mailinglist, guy named Angus

    n1 = # of samples in sample 1
    mean1 = mean of sample 1
    sem1 = standard error of mean of sample 1
    sem1 is the sample standard deviation divided by the square root of n1

    sem1 = np.std(n1)/sqrt(n1)
    """

    # calculate standard variances of the means
    svm1 = sem1**2 * n1
    svm2 = sem2**2 * n2
    print "standard variance of the mean 1: %0.4f" % svm1
    print "standard variance of the mean 2: %0.4f" % svm2
    print ""
    t_s_prime = (mean1 - mean2)/np.sqrt(svm1/n1+svm2/n2)
    print "t'_s = %0.4f" % t_s_prime
    print ""
    t_alpha_df1 = scipy.stats.t.ppf(1-alpha/2, n1 - 1)
    t_alpha_df2 = scipy.stats.t.ppf(1-alpha/2, n2 - 1)
    print "t_alpha[%d] = %0.4f" % (n1-1, t_alpha_df1)
    print "t_alpha[%d] = %0.4f" % (n2-1, t_alpha_df2)
    print ""
    t_alpha_prime = (t_alpha_df1 * sem1**2 + t_alpha_df2 * sem2**2) / \
                    (sem1**2 + sem2**2)
    print "t'_alpha = %0.4f" % t_alpha_prime
    print ""
    if abs(t_s_prime) > t_alpha_prime:
        print "Significantly different"
        return True
    else:
        print "Not significantly different"
        return False

def SimpleCorr(seqdata, ran='no', rev='no', maxlen=20):
    """Calculate the correlation between RNA-DNA and DNA-DNA energies with PY
    for incremental positions of the correlation window (0:3) to (0:20). Two
    versions are returned: one where sequence-average expected energies are
    added after msat and one where nothing is done for msat.
    The rev='yes' option only gives meaningful result for
    incremental without adding expected values."""

    rowdata = [[row[val] for row in seqdata] for val in range(len(seqdata[0]))]
    seqs = rowdata[1]

    if ran == 'yes':
        nrseq = len(seqdata) #how many sequences. (might remove some)
        seqs = ITS_generator(nrseq)
#    labels = ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst','APR','MSAT','R']

    PY = rowdata[2]
#    PY = rowdata[8] # the correlation between RNA/DNA and R must be commented
#    upon.

    msat = rowdata[-2]
    # Calculating incremental energies from 3 to 20 with and without expected
    # energies added after msat in incr[1]. incr[0] has incremental energies
    # without adding expected energies after msat. NOTE E(4nt)+E(5nt)=E(10nt)
    # causes diffLen+1

    incrEnsRNA = [[], []] # 0 is withOut exp, 1 is With exp
    incrEnsDNA = [[], []]
    start = 3 #nan for start 0,1, and 2. start =3 -> first is seq[0:3] (0,1,2)
    for index, sequence in enumerate(seqs):
        incrRNA = [[], []]
        incrDNA = [[], []]

        for top in range(start, maxlen+1):
            # Setting the subsequences from which energy should be calculated
            rnaRan = sequence[0:top]
            dnaRan = sequence[0:top+1]

            if rev == 'yes':
                rnaRan = sequence[top-start:]

            tempRNA = Ec.RNA_DNAenergy(rnaRan)
            tempDNA = Ec.PhysicalDNA(dnaRan)[-1][0]

            incrRNA[0].append(tempRNA)
            incrDNA[0].append(tempDNA)

            # you are beyond msat -> calculate average energy
            if top > msat[index]:
                diffLen = top-msat[index]
                #RNA
                baseEnRNA = Ec.RNA_DNAenergy(sequence[:int(msat[index])])
                diffEnRNA = Ec.RNA_DNAexpected(diffLen+1)
                incrRNA[1].append(baseEnRNA + diffEnRNA)

                #DNA
                baseEnDNA = Ec.DNA_DNAenergy(sequence[:int(msat[index])])
                diffEnDNA = Ec.DNA_DNAexpected(diffLen+1)
                incrDNA[1].append(baseEnDNA + diffEnDNA)
            else:
                incrRNA[1].append(tempRNA)
                incrDNA[1].append(tempDNA)
        #RNA
        incrEnsRNA[0].append(incrRNA[0])
        incrEnsRNA[1].append(incrRNA[1])

        #DNA
        incrEnsDNA[0].append(incrDNA[0])
        incrEnsDNA[1].append(incrDNA[1])

    #RNA
    incrEnsRNA[0] = np.array(incrEnsRNA[0]).transpose() #transposing
    incrEnsRNA[1] = np.array(incrEnsRNA[1]).transpose() #transposing

    #DNA
    incrEnsDNA[0] = np.array(incrEnsDNA[0]).transpose() #transposing
    incrEnsDNA[1] = np.array(incrEnsDNA[1]).transpose () #transposing
    # Calculating the different statistics

    # RNA + DNA without expected energy
    incrEnsRNADNA = incrEnsRNA[0] + incrEnsDNA[0]

    RNADNA = []
    for index in range(len(incrEnsRNADNA)):
        RNADNA.append(spearmanr(incrEnsRNADNA[index], PY))

    #RNA
    incrWithExp20RNA = []
    incrWithoExp20RNA = []
    for index in range(len(incrEnsRNA[0])):
        incrWithoExp20RNA.append(spearmanr(incrEnsRNA[0][index], PY))
        incrWithExp20RNA.append(spearmanr(incrEnsRNA[1][index], PY))

    #DNA
    incrWithExp20DNA = []
    incrWithoExp20DNA = []
    for index in range(len(incrEnsDNA[0])):
        incrWithoExp20DNA.append(spearmanr(incrEnsDNA[0][index], PY))
        incrWithExp20DNA.append(spearmanr(incrEnsDNA[1][index], PY))

    arne = [incrWithExp20RNA, incrWithExp20DNA, incrWithoExp20RNA, incrWithoExp20DNA]

    output = {}
    output['RNA_{0} msat-corrected'.format(maxlen)] = arne[0]
    output['DNA_{0} msat-corrected'.format(maxlen)] = arne[1]

    output['RNA_{0} uncorrected'.format(maxlen)] = arne[2]
    output['DNA_{0} uncorrected'.format(maxlen)] = arne[3]

    output['RNADNA_{0} uncorrected'.format(maxlen)] = RNADNA

    return output

def ITS_generator(nrseq, length=18, ATstart=True):
    """Generate list of nrseq RNA random sequences. Seq with length. """

    gatc = list('GATC')

    if ATstart:
        beg = 'AT'
        length = length-2
    else:
        beg = ''

    return [beg + ''.join([random.choice(gatc) for dummy1 in range(length)])
            for dummy2 in range(nrseq)]

def LadderScrutinizer(lizt, n=10):
    """ Calculate sets of size n of random sequences, perform RNA/DNA energy
    calculation for incremental ITS sublenghts, and evaluate the "monotonicity"
    of the correlation ladder; evaluation depends on number k of allowed
    deviations from monotonicity."""
    msatyes = 'no'
    rowdata = [[row[val] for row in lizt] for val in range(len(lizt[0]))]
    msat = rowdata[-2]
    py = rowdata[2]
    # The real correlation ladder
    arne = SimpleCorr(lizt, maxlen=20)
    laddA = [tup[0] for tup in arne[2]][:13]
    # Consider the ladder up to arne[6][15-3]. Rank these. Subtract a range(12).
    # This will be deviation from perfect linear increase. This is the score.
    ladd = np.array(Orderrank.Order(laddA[:13]))
    perf = np.array(range(13))
    Zcore = sum(abs(ladd-perf))
    # The real sequence has a Zcore of 2
    # Randomize
    nrseq = len(rowdata[0]) #how many sequences. (might remove some)
    # Generate random sequences and correlate their RNA/DNA values with PY
    noter = [[],[],[]]
    maxcor = max(laddA)

    for dummy in range(n):
        # Generate 39-43 random sequences of length 20
        ranITS = ITS_generator(nrseq)
        # Calculate the energy
        enRNA = [Ec.PhysicalRNA(seq, msat, msatyes) for seq in ranITS]
        RNAen = [[[row[val][0] for row in enRNA], row[val][1]] for val in
                  range(len(enRNA[0]))]
        corrz = [spearmanr(enRNArow[0],py) for enRNArow in RNAen]
        onlyCorr = np.array([tup[0] for tup in corrz])
        # Rank the correlation ladder
        ranLa = Orderrank.Order(onlyCorr[:13])
        Zcore = sum(abs(ranLa-perf))
        bigr = [1 for value in onlyCorr if value >= maxcor]
        noter[0].append(Zcore)
        noter[1].append(sum(bigr))
    return noter

def ITSgenerator():
    """Generate a random ITS """
    gatc = list('GATC')
    return 'AT'+ ''.join([random.choice(gatc) for dummy1 in range(18)])

def PvalDeterminer(toplot):
    """ Take a list of list of [Corr, pval] and return an interpolated Corr
    value that should correspond to pval = 0.05. """
    toplot = sorted(toplot, key=operator.itemgetter(1)) # increasing pvals like needed
    pvals = [tup[1] for tup in toplot]
    corrs = [tup[0] for tup in toplot]
    f = scipy.interpolate.interp1d(pvals, corrs)
    return f(0.05)

def Purine_RNADNA(repnr=100, ranNr=39, rand='biased', upto=20):
    """ Calculate the correlation coefficient between a DNA sequence's RNA-DNA
    energy and its purine content. """
    # NOTE should this be done with ITS sequence of the same probability
    # distribution as found in Hsu's data? Probably not.
    # RESULT: with 100000 runs of 39 sequences: mean = 0.29, sigma=0.15
    # RESULT: with biased sequences I get for 100000 runs 0.35 p/m 0.15 -> 0.20 to 0.50,
    # which contains the 0.46! 
    # This means that the correlation between purines and RNADNA energy in hsu's
    # 39 sequences (0.46)

    cor_vals = []
    p_vals = []
    for dummy in range(repnr):
        purTable = np.zeros(ranNr)
        enTable = np.zeros(ranNr)
        for nr in range(ranNr):
            sequence = ITS_generator(1)[0][:upto]
            purTable[nr] = sequence.count('G') + sequence.count('A')
            enTable[nr] = Ec.RNA_DNAenergy(sequence)
        cor, pval = scipy.stats.spearmanr(purTable, enTable)
        cor_vals.append(cor)
        p_vals.append(pval)

    pval_mean, pval_sigma = scipy.stats.norm.fit(p_vals)
    cor_mean, cor_sigma = scipy.stats.norm.fit(cor_vals)
    #return 'Mean (corr, pval): ', cor_mean, pval_mean, ' -- Sigma: ', cor_sigma, pval_sigma
    return 'Mean corr : ', cor_mean, ' -- Sigma: ', cor_sigma

class Result(object):
    """
    Placeholder for results for plotting
    These are results from the simulation for a given its_length; they are the
    top 20 correlation coefficients, pvals, fitted parameters (c1, c2, c3, c4)
    and the final values

    You get in slightly different things here for Normal and Random results.

    Normal results are the (max) 20 best for each its-length. Random results are
    the X*20 best. For random, mean and std are what's interesting (the mean and
    std of the best for each random sample)
    """
    def __init__(self, corr=[np.nan], pvals=np.nan, params=np.nan,
                 finals=np.nan):

        self.corr = corr
        self.pvals = pvals
        self.params = params
        self.finals = finals

        # calculate the mean, std, max, and min values for correlation
        # coefficients, pvalues, and finals simulation results (concentration in
        # its_len)
        self.corr_mean = np.mean(corr)
        self.corr_std = np.std(corr)
        max_indx = np.abs(corr).argmax() # correlation may be negative

        if len(corr) == 1 and np.isnan(corr[0]):
            self.corr_max = np.nan
        else:
            self.corr_max = corr[max_indx]

        self.pvals_mean = np.mean(pvals)
        self.pvals_std = np.std(pvals)
        self.pvals_min = np.min(pvals)

        self.finals_mean = np.mean(finals)
        self.finals_std = np.std(finals)
        self.finals_max = np.max(finals)
        self.finals_min = np.min(finals)

        # The parameters are not so easy to do something with; the should be
        # supplied in a dictionray [c1,c2,c3,c4] = arrays; so make mean, std,
        # min, max for them as well

        # handle nan
        if params is np.nan:
            params = {'c1':[np.nan],
                      'c2':[np.nan],
                      'c3':[np.nan],
                      'c4':[np.nan],
                      'c5':[np.nan]}

        self.params_mean = dict((k, np.mean(v)) for k,v in params.items())
        self.params_std = dict((k, np.std(v)) for k,v in params.items())
        self.params_max = dict((k, np.max(v)) for k,v in params.items())
        self.params_min = dict((k, np.min(v)) for k,v in params.items())

        # You need the parameters for the highest correlation coefficient, since
        # that's what you're going to plot (but also plot the mean and std)
        # You rely on that v is sorted in the same order as corr
        self.params_best = dict((k, v[0]) for k,v in params.items())


class ITS(object):
    """ Storing the ITSs in a class for better handling. """
    def __init__(self, sequence, name='noname', PY=-1, PY_std=-1, apr=-1, msat=-1):
        # Set the initial important data.
        self.name = name
        #self.sequence = sequence
        self.sequence = sequence + 'TAAATATGGC'
        self.PY = PY
        self.PY_std = PY_std
        self.APR = apr
        self.msat = int(msat)

        # Redlisted sequences 
        self.redlist = []
        self.redlisted = False

        __seqlen = len(self.sequence)
        __indiv = list(self.sequence)
        __dinucs = [__indiv[c] + __indiv[c+1] for c in range(__seqlen-1)]

        # Make di-nucleotide vectors for all the energy parameters
        self.rna_dna_di = [Ec.NNRD[di] for di in __dinucs]
        self.dna_dna_di = [Ec.NNDD[di] for di in __dinucs]
        self.keq_delta_di = [Ec.delta_keq[di] for di in __dinucs]

    def __repr__(self):
        return "{0}, PY: {1}".format(self.name, self.PY)

def RedlistInvestigator(ITSs):
    """ Read minus ten elements from Filereader and add redlisted sequence
    elements to the ITS class instances."""

    # IDEA look for not just :6 but also 1:7, 2:8 in this list. It might be
    # different. Or, rahter, see where in that list the consensus appears most
    # often. Then you will know where the minus 10 is. OR WILL YOU? Alternative:
    # take all of them, and show for each sequence the sum of these elements.
    # Then correlate this sum to PY. Might work. As well, find out the exact
    # positions of the curated elements and look for these elements in these
    # positions only.
    # RESULT: TATAAT consensus appears in 2:8 90% of the time. Even if I look at
    # these sequences, I see no pattern. Looking at 1:8 there is no patterin
    # either. This is busted.
    # Should you look at extented consensus? T_GT_TATAAT? Perhaps count the
    # number of matches? In any case, since Hsu's TATAAT consensus had no effect
    # on the abortive ladder, this is probably a bind alley!

    minus10s = Filereader.MinusTen()
    modMin = [row[1:] for row in minus10s] # using only the last six nts of the -10
    modMin = list(set(modMin)) # Removing duplicates!!!!!!!!!!!!!!
    curated = ['ACGAT', 'ACGCT', 'TAACGCT', 'TAATTTT', 'ATTGT', 'TAGGA', 'TATAAT']
#    repeats = [letter*x for letter in ['G','A','T','C'] for x in range(3, 21)]
    polyA = ['A'*x for x in range(3, 21)]
    polyT = ['T'*x for x in range(3, 21)]

    def avoidappender(itr, avoidpath, avoidme, name):
        """ If 'avoidme' is found in the itr sequence,  """
        if avoidme in itr.sequence:
            locations = Workhouse.SubSeqLocater(itr.sequence, avoidme)
            avoidpath.append([avoidme, locations, name])

    # Goes through all itrs and all avoids. If an avoid for a specific
    # avoid-class is found in an itr, the avoid is addes to the avoidclass
    # redlist of the itr, and the boolean 'contains redlist'
    # is set to True. Finally, if at least one avoid has been found in the itr,
    # the boolean 'itr.redlisted' is set to True (False by default).
    for itr in ITSs:

        #shortcut variable
        redlist = itr.redlist

        for avoidme in modMin:
            avoidappender(itr, redlist, avoidme, 'Minus 10')

#        for avoidme in curated:
#            avoidappender(itr, redlist, avoidme, 'Curated')

#        for avoidme in polyA:
#            avoidappender(itr, redlist, avoidme, 'PolyA')
#        # If there are more than one PolyA and they are part of the same
#        # subsequence, keep only the longest one

#        for avoidme in polyT:
#            avoidappender(itr, redlist, avoidme, 'PolyT')

        if itr.redlist != []:
            itr.redlisted = True
            itr.redlist = Polyremover(itr.redlist)

#    fig = plt.figure()
#    ax = fig.add_subplot(111)
##    ax.scatter(red[0],red[1],'r',notRed[0], notRed[1],'b')
#    ax.scatter(red[0],red[1], c='r')
#    ax.scatter(notRed[0], notRed[1], c='b')

def Polyremover(redlist):
    typelist = [row[2] for row in redlist] # list of the 'Minus 10', 'PolyT' etc
    delrows = [] # rows to delete from redarr
    pols = ['PolyT', 'PolyA']
    for pol in pols:
        if typelist.count(pol) > 1: # if empty, count() will return 0! :)
            start = []
            for index, row in enumerate(redlist):
                start.append(row[1][0][0])
                if index == 0:
                    continue # special case for first row
                one = (row[2] == pol)               # current type is polyT
                two = (redlist[index-1][2] == pol)  # last type also polyT
                three = (start[index] == start[index-1]) # both have same start
                if one and two and three:
                    delrows.append(index-1)             # append previous row for deletion
    if delrows != []:
        delrows.sort(reverse=True) # reverse sort for recursive rowdel
        for row in delrows:
            del(redlist[row])
    return redlist

def RedlistPrinter(ITSs):
    """ Give a printed output of what ITSs have what redlisted sequence
    in them at which location. """
    # What I would like to see (sorted by PY value): Name of promoter, PY value, redlisted sequences
    # according to type with location in brackets.
    _header = '{0:10} {1:5} {2:15} {3:8} {4:10} {5:10}'.format('Promoter', 'PY value', 'Sequence', 'Redlisted '
                                                          'sequence', 'Location on ITS','Redlist type')
    #print(header)
    # Sort ITSs according to PY value
    ITSs.sort(key=operator.attrgetter('PY'), reverse=True)
    for itr in ITSs:
        promoter = itr.name
        PY = itr.PY
        sequence = itr.sequence
        rnadna = itr.rna_dna1_15
        if not itr.redlisted:
            print ('{0:10} {1:5} {2:20} {3:9} {4:19} {5:10} '
                   '{6:4}'.format(promoter, PY, sequence, 'No match', '', '',
                                 rnadna))
        else:
            for index, avoid in enumerate(itr.redlist):
                name = avoid[0]
                location = avoid[1]
                if len(location) == 1: location = location[0]
                kind = avoid[2]
                if index == 0:
                    print('{0:10} {1:5} {2:20} {3:9} {4:19} {5:10} '
                          '{6:4}'.format(promoter, PY, sequence, name, location,
                          kind, rnadna))
                else:
                    if kind == itr.redlist[index-1][2]:
                        print('{0:10} '
                              '{1:5} {2:20} {3:9} {4:19} {5:10}'.format('', '', '',
                                                                        name,
                                                                        location,
                                                                       '   --'))
                    else:
                        print('{0:10} '
                              '{1:5} {2:20} {3:9} {4:19} {5:10}'.format('', '', '', name, location, kind))

def HsuRandomTester(ITSs):
    """ Test the distribution of nucleotides in Hsu's data. Also correlate
    purines with RNA-DNA energi."""
    gatc = {'G':[], 'A':[], 'T':[], 'C':[]}

    bitest = scipy.stats.binom_test # shortcut to function

    for itr in ITSs:
        gatc['G'].append(itr.sequence.count('G'))
        gatc['A'].append(itr.sequence.count('A'))
        gatc['T'].append(itr.sequence.count('T'))
        gatc['C'].append(itr.sequence.count('C'))

        seq = itr.sequence
        thisone = np.array([seq[2:].count('G'), seq[2:].count('A'), seq[2:].count('T'),
                     seq[2:].count('C')])
        for ind, freq in enumerate(thisone):
            pval = bitest(freq, 18, 1/4)
            if pval < 0.05:
                print itr.name, bitest(freq, 18, 1/4)
                print gatc.keys()[ind]
                print itr.PY

    sumgatc = np.array([sum(gatc['G']), sum(gatc['A']), sum(gatc['T']),
                        sum(gatc['C'])])

    # purine vs rna-dna energies
    purine_levs = [itr.sequence.count('G') + itr.sequence.count('A') for itr in ITSs]
    rna_dna15 = [itr.rna_dna1_15 for itr in ITSs]
    rna_dna20 = [itr.rna_dna1_20 for itr in ITSs]
    rho15, p15 = spearmanr(purine_levs, rna_dna15)
    rho20, p20 = spearmanr(purine_levs, rna_dna20)

    print sumgatc
    total = sum(sumgatc) # Should be 43*20 = 860
    fracs = sumgatc/total
    print fracs

    # Two-tailed binomial test of the 0-hypothesis that the probability to have
    # G, A, T, and C are 1/4
    pvalsgatc = [bitest(sumgatc[x], total, 1/4) for x in range(4)]
    print pvalsgatc
    # p-value: gives the probability that we reject the 0-hypothesis by mistake. 
    # RESULT: the distribution of G,A,T,C is not random. There is more A and T
    # than G and C. Consequences...?
    # RESULT however, the purine/pyrimidine fractions are 50:50
    # RESULT in all the sequences that have non-random nucleotide distributions,
    # 6/8 have high PY, correlated iwth lack of mostly C.

def PurineLadder(ITSs):
    """ Create a ladder of correlations between purine vs PY and energy."""
    pur_ladd_corr = []
    # calculate the purine ladders
    for itr in ITSs:
        seq = itr.sequence
        pur_ladder = [seq[:stop].count('A') + seq[:stop].count('G') for stop in
               range(2,20)]
        a_ladder = [seq[:stop].count('G') for stop in range(2,20)]
        energies = [Ec.RNA_DNAenergy(seq[:stp]) for stp in range(2,20)]
        itr.energy_ladder = energies
        itr.pur_ladder = pur_ladder
        itr.a_ladder = a_ladder
    # put the ladders in an array so that each column can be obtained
    pur_ladd = np.array([itr.pur_ladder for itr in ITSs])
    a_ladd = np.array([itr.a_ladder for itr in ITSs])
    PYs = np.array([itr.PY for itr in ITSs])
    energies = np.array([itr.energy_ladder for itr in ITSs])
    pur_corr = [spearmanr(pur_ladd[:,row], PYs) for row in range(18)]
    a_corr = [spearmanr(a_ladd[:,row], PYs) for row in range(18)]
    en_corr = [spearmanr(a_ladd[:,row], energies[:,row]) for row in range(18)]
    # The correlation between purines and the Hsu R-D energies (quoted as 0.457
    # for up to 20)
    pur_en_corr = [spearmanr(pur_ladd[:,row], energies[:,row]) for row in range(18)]

    for row in pur_en_corr:
        print row

    # RESULT the energy ladder is very strong against PY values. There's even a
    # peak correlation for G+A, so both are needed to explain the data, not just
    # the As, although the As stand for most of the variation. Is there any way
    # of explaining this??? :( purine-rna_dna correlation is more than expected
    # by chance -- but this is only due to the As. There is no correlation to
    # Gs.
    # The purine-PY correlation follows the RNA-DNA energy perfectly in
    # terms of auto-correlation. Thus it can easily be argued that the RNA-DNA
    # energy link is just a byproduct of the influence of the As in the
    # purine-PY correlation, since RNA-DNA energy fairly strongly correlated to
    # purine levels the whole way.
    #
    # OK OK OK. The reason the strong link exists between purines and RNA-DNA
    # energy is of course that these sequences have a disproportional high
    # amount of As.

    # Yet why is there a correlation between Gs and PY???

    # If you want to do real science, you have to test if it is the RNA-DNA
    # energy and not just the purine content that matters. Thus you should
    # investigate 40 ITS sequences that vary in both their RNA-DNA binding
    # energies and also in their purine content.
    # QUESTION: is the overrepresentation of As constant throughout the
    # sequences? Can be calculated by calculating the expected frequency of As
    # for successive segments of the 42 seqs. The same as calling the function
    # you already have but many times.
    # NOTE TODO this is what you were working on just as you left trondheim. You
    # were doing a deeper analysis of the relationship between Purines, PYs, and
    # rna-dna energies. It is clear that the purine-PY link is much stronger
    # than the rna-dna energy-link. It is also clear that the purine--rna-dna
    # link is within one standard deviation of what you would expect from random
    # sequences -- hinting that the rna-dna correlation is not just some fluke
    # because 
    en15s = [Ec.RNA_DNAenergy(itr.sequence[:15]) for itr in ITSs]
    pur15s = [itr.sequence[:15].count('A') + itr.sequence[:15].count('G') for itr in ITSs]
    fck_corr = spearmanr(en15s, pur15s)

def ReadAndFixData():
    """ Read Hsu paper-data and Hsu normalized data. """

    # labels of Hsu data
    #_labels = ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst',
    #           'APR','APRst','MSAT','R']

    # Selecting the dataset you want to use
    #
    lazt = Filereader.PYHsu(hsu1) # Unmodified Hsu data
    #lazt = Filereader.PYHsu(hsu2) # Normalized Hsu data by scaling
    #lazt = Filereader.PYHsu(hsu3) # Normalized Hsu data by omitting experiment3
    #lazt = Filereader.PYHsu(hsu4) # 2007 data, skipping N25, N25anti
    #lazt = Filereader.PYHsu(hsu5) # 2007 data, including N25, N25anti
    #biotech = '/Rahmi/full_sequences_standardModified'

    # Selecting the columns I want from Hsu. Removing all st except PYst.
    # list=[Name 0, Sequence 1, PY 2, PYstd 3, RPY 4, RIFT 5, APR 6, MSAT 7, R 8]
    lizt = [[row[0], row[1], row[2], row[3], row[4], row[6], row[8], row[10],
             row[11]] for row in lazt]

    # name seq PY X X X X X X
    # From Stepanova 2009
    #lizt = [['WT', 'GCCCTCGATATGGGGATTTTTA', 46, 0, 0, 0, 0, 0, 0],
            #['T7', 'GTCGAGAGGGACACGGCGAATA', 89, 0, 0, 0, 0, 0, 0],
            #['WT', 'GTCTGAGATATGGGGATTTTA', 95, 0, 0, 0, 0, 0, 0],
            #['WT', 'GCCCTGTCGCGGGGGATTTTA', 53, 0, 0, 0, 0, 0, 0],
            #['WT', 'GCCCTGGATATGTCAGTTTTA', 76, 0, 0, 0, 0, 0, 0],
            #['WT', 'GCCCTGGATATGGGGATGGCA', 62, 0, 0, 0, 0, 0, 0]]

    # Making a list of instances of the ITS class. Each instance is an itr.
    # Storing Name, sequence, and PY.
    ITSs = []
    for row in lizt:

        ITSs.append(ITS(row[1], row[0], row[2], row[3], row[6], row[7]))
        #ITSs.append(ITS(row[1][5:], row[0], row[2], row[3], row[6], row[7]))

    return ITSs

def genome_wide():
    """
    Check if there is an RNA-DNA correlation for the ITS in the genome in
    general. Compare the RNA-DNA energy for the different promoters with

    1) Random sites in the e coli genome
    2) Shuffled +1 to +20(15) sites

    Most are promoters are compuattionally annotated

     11 [HIPP|W|Human inference of promoter position],[AIPP|W|Automated inference of promoter position]
     11 [HIPP|W|Human inference of promoter position],[ICA|W|Inferred by computational analysis]
     11 [TIM|S|Transcription initiation mapping],[HIPP|W|Human inference of promoter position],[IMP|W|Inferred from mutant phenotype]
     13 [TIM|S|Transcription initiation mapping],[IMP|W|Inferred from mutant phenotype]
     15 [HIPP|W|Human inference of promoter position],[IEP|W|Inferred from expression pattern]
     15 [IDA|W|Inferred from direct assay],[IEP|W|Inferred from expression pattern]
     20 [TIM|S|Transcription initiation mapping],[HTTIM|S|High-throughput transcription initiation mapping]
     27 [TIM|S|Transcription initiation mapping],[ICA|W|Inferred by computational analysis],[AIPP|W|Automated inference of promoter position]
     33 [TIM|S|Transcription initiation mapping],[AIPP|W|Automated inference of promoter position]
     61 [IEP|W|Inferred from expression pattern]
     96 [AIPP|W|Automated inference of promoter position]
    134 [HIPP|W|Human inference of promoter position]
    153 [HIPP|W|Human inference of promoter position],[TIM|S|Transcription initiation mapping]
    186 [TIM|S|Transcription initiation mapping],[HIPP|W|Human inference of promoter position]
    261 [HTTIM|S|High-throughput transcription initiation mapping]
    579 [TIM|S|Transcription initiation mapping]
   1777 [ICWHO|W|Inferred computationally without human oversight]
    """

    # 1) Get the RNA-DNA energies for the different sigma-promoters, both std
    # and mean
    sigprom_dir = 'sequence_data/ecoli/sigma_promoters'
    sig_paths = glob(sigprom_dir+'/*')

    sig_dict = {}
    for path in sig_paths:
        fdir, fname = os.path.split(path)
        sig_dict[fname[8:15]] = path

    # parse the promoters and get the energies
    sig_energies = {}
    sig_jumbl_energies = {}
    #
    clen = 20

    gc_count_sigma = []

    all_energies = []

    for sigmaType, sigpath in sig_dict.items():
        at_count = []
        energies = []
        jumbl_energies = []
        for line in open(sigpath, 'rb'):
            # skip the silly header

            if line == '\n' or line.startswith('#') or line.startswith('\t'):
                continue

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            # one sequence like this 
            if seq == '':
                continue

            #if 'TIM' not in evidence:
                #continue

            if clen == 20:
                its = seq[-20:].upper()
            else:
                its = seq[-20:-(20-clen)].upper()

            # count gc content
            gc_count_sigma.append(its.count('G') + its.count('C'))

            jumbl_its = ''.join([its[np.random.randint(clen)] for v in
                                 range(clen)])
            energies.append(Ec.RNA_DNAenergy(its))
            all_energies.append(Ec.RNA_DNAenergy(its))

            jumbl_energies.append(Ec.RNA_DNAenergy(jumbl_its))

            at_count.append(its.count('A') + its.count('T'))

        sig_energies[sigmaType] = (np.mean(energies), np.std(energies))
        sig_jumbl_energies[sigmaType] = (np.mean(jumbl_energies), np.std(jumbl_energies))

        print sigmaType, np.mean(at_count), np.std(at_count)

    # get energies for 10000 random locations in e coli genome
    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))
    drawlen = len(ecoli_seq) - clen

    gc_count_rand = []

    debug()
    rand_at = []
    randenergies = []
    jumbl_randenergies = []
    for rnr in range(100000):
        rand_pos = np.random.random_integers(drawlen)
        randseq = ecoli_seq[rand_pos:rand_pos+clen]

        gc_count_sigma.append(its.count('G') + its.count('C'))
        gc_count_rand.append(randseq.count('G') + randseq.count('C'))

        rand_at.append(randseq.count('A') + randseq.count('T'))

        randenergies.append(Ec.RNA_DNAenergy(randseq))

        jumbl_rs = ''.join([randseq[np.random.randint(clen)] for v in range(clen)])
        jumbl_randenergies.append(Ec.RNA_DNAenergy(jumbl_rs))

    print 'random at count', np.mean(rand_at)
    print 'std at count', np.std(rand_at)
    print ''
    print 'random mean energies', np.mean(randenergies)
    print 'random std energies', np.std(randenergies)
    print ''
    print 'random jumbl mean energies', np.mean(jumbl_randenergies)
    print 'random jumbl std energies', np.std(jumbl_randenergies)
    print ''

    #print 'jumbled mean energies', np.mean(randenergies)
    #print 'jumbled std energies', np.std(randenergies)

    print ''
    print 'all promoters mean and std', np.mean(all_energies), np.std(all_energies)
    print ''

    for sigm in sig_energies.keys():
        print sigm, 'original mean and std', sig_energies[sigm]
        print sigm, 'jumbled mean and std', sig_jumbl_energies[sigm]

    print 'sigma gc rate', sum(gc_count_sigma)/(len(gc_count_sigma)*clen)
    print 'random gc rate', sum(gc_count_rand)/(len(gc_count_rand)*clen)

    fig, ax = plt.subplots(1)
    ax.hist(randenergies, bins=40, alpha=0.5, color='g', label='Random',
            normed=True)
    ax.hist(all_energies, bins=40, alpha=0.85, color='b', label='Promoter ITS',
           normed=True)

    #ax.set_title = '\mu_random = -22.5, \mu_ITS = -20.1'

    ax.legend()

    for fig_dir in fig_dirs:
        for formt in ['pdf', 'eps', 'png']:

            name = 'ITS_rand_distribution_comparison.' + formt
            odir = os.path.join(fig_dir, formt)

            if not os.path.isdir(odir):
                os.makedirs(odir)

            fig.savefig(os.path.join(odir, name), transparent=True, format=formt)

    alpha = 0.001
    n1 = len(all_energies)
    n2 = len(randenergies)
    mean1 = np.mean(all_energies)
    mean2 = np.mean(randenergies)

    # standard error of the mean is std / sqrt(n)
    sem1 = np.std(all_energies) / np.sqrt(n1)
    sem2 = np.std(randenergies) / np.sqrt(n1)

    welchs_approximate_ttest(n1, mean1, sem1, n2, mean2, sem2, alpha)

def dimer_calc(energy_function, seq, operators):
    """
    Use the dictionary 'energy_function' to cacluate the value of seq
    """

    indiv = [l for l in list(seq)] # splitting sequence into individual letters

    # getting dinucleotides
    neigh = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]

    # If only one energy term just return the sum
    if not operators:
        return sum([energy_function[nei] for nei in neigh])
    else:

        def sig(sign, val):
            if sign == '+':
                return val
            elif sign == '-':
                return -val

        vals = []
        # zip the operators with the enery function and change sign after
        # applying the energy function
        for nei in neigh:
            # add all the energy terms there
            ens = [sig(sign, enf[nei]) for enf, sign in zip(energy_function,
                                                              operators)]
            # apply the operators to the terms
            vals.append(sum(ens))

        return sum(vals)


def NewSimpleCorr(seqdata, energy_function, maxlen=20, operators=False):
    """Calculate the correlation between RNA-DNA and DNA-DNA energies with PY
    for incremental positions of the correlation window (0:3) to (0:20)

    If operators != False, it must be (+,+), (+,-,+) etc, which specify the sign
    of the energy functions.
    """

    rowdata = [[row[val] for row in seqdata] for val in range(len(seqdata[0]))]
    seqs = rowdata[1]

    PY = rowdata[2]

    # Calculating incremental energies from 3 to 20 with and without expected
    # energies added after msat in incr[1]. incr[0] has incremental energies
    # without adding expected energies after msat. NOTE E(4nt)+E(5nt)=E(10nt)
    # causes diffLen+1
    start = 3 #nan for start 0,1, and 2. start =3 -> first is seq[0:3] (0,1,2)

    incrEnsRNA = []
    for index, sequence in enumerate(seqs):
        incrRNA = []

        for top in range(start, maxlen+1):
            # Setting the subsequences from which energy should be calculated
            rnaRan = sequence[0:top]

            # add the 'energy'
            incrRNA.append(dimer_calc(energy_function, rnaRan, operators))

        incrEnsRNA.append(incrRNA)

    #RNA
    incrEnsRNA = np.array(incrEnsRNA).transpose() #transposing

    # Calculating the different statistics

    #RNA
    stats = []
    for index in range(len(incrEnsRNA)):
        stats.append(spearmanr(incrEnsRNA[index], PY))

    return stats

def new_ladder(lizt):
    """ Printing the probability ladder for the ITS data. When nt = 5 on the
    x-axis the correlation coefficient of the corresponding y-value is the
    correlation coefficient of the binding energies of the ITS[0:5] sequences with PY.
    nt = 20 is the full length ITS binding energy-PY correlation coefficient.  """
    maxlen = 20
    pline = 'yes'

    # how will it go with 26 random ones?
    # Pick out 26 random positions in lizt -- the result is zolid.
    rands = set([])
    while len(rands)<26:
        rands.add(random.randrange(0,43))
    lizt = [lizt[i] for i in rands]

    # use all energy functions from new article
    #from dinucleotide_values import resistant_fraction, k1, kminus1, Keq_EC8_EC9

    from Ec import reKeq, NNRD, NNDD, super_en

    name2func = [('RNA-DNA', NNRD), ('DNA-DNA', NNDD), ('Translocation', reKeq),
                ('RNA-DNA - Translocation', super_en)]
    # The r_f, k1, and K_eq correlate (r_f positively and k1 and K_eq negatively

    #plt.ion()
    fig, ax = plt.subplots()

    colors = ['b', 'g', 'c', 'k']

    for indx, (name, energy_func) in enumerate(name2func):
        corr = NewSimpleCorr(lizt, energy_func, maxlen=maxlen)

        # The first element is the energy of the first 3 nucleotides
        start = 3
        end = maxlen+1
        incrX = range(start, end)

        r_vals = [tup[0] for tup in corr]

        ax.plot(incrX, r_vals, label=name, linewidth=2, color=colors[indx])

    xticklabels = [str(integer) for integer in range(3,21)]
    yticklabels = [str(integer) for integer in np.arange(-1, 1.1, 0.1)]
    #yticklabels = [str(integer) for integer in np.arange(0, 1, 0.1)]
    # make the almost-zero into a zero
    yticklabels[10] = '0'

    ax.set_xticks(range(3,21))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("RNA length (nt)", size=26)
    ax.set_ylabel("Correlation coefficient, $r$", size=26)

    if pline == 'yes':
        pval = PvalDeterminer(corr)
        ax.axhline(y=pval, ls='--', color='r', label='p = 0.05 threshold', linewidth=2)
        ax.axhline(y=-pval, ls='--', color='r', linewidth=2)

    ax.legend(loc='lower left')
    ax.set_yticks(np.arange(-1, 1.1, 0.1))
    #ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_yticklabels(yticklabels)

    # awkward way of setting the tick font sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)

    fig.set_figwidth(9)
    fig.set_figheight(10)

    # you need a grid to see your awsome 0.8 + correlation coefficient
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

    ax.set_title('Cummulative correlation of dinucletide parameters with abortive '\
             'initiation')

    #for fig_dir in fig_dirs:
        #for formt in ['pdf', 'eps', 'png']:
            #name = 'simplified_ladder.' + formt

            #odir = os.path.join(fig_dir, formt)

            #if not os.path.isdir(odir):
                #os.makedirs(odir)

            #fig.savefig(os.path.join(odir, name), transparent=True, format=formt)


def new_genome():
    """
    For the -20,+20 of promoters, calculate the average of all your
    scoring-dicts for each position.
    """
    sigprom_dir = 'sequence_data/ecoli/sigma_promoters'
    sig_paths = glob(sigprom_dir+'/*')

    sig_dict = {}
    for path in sig_paths:
        fdir, fname = os.path.split(path)
        sig_dict[fname[8:15]] = path

    #from dinucleotide_values import resistant_fraction, k1, kminus1, Keq_EC8_EC9
    from Ec import NNRD, NNDD, super_en, reKeq

    name2func = [('Keq', reKeq), ('RNA-DNA', NNRD), ('DNA-DNA', NNDD), ('Combo',
                                                                       super_en)]
    names = [name for name, func in name2func]

    ## parse through all promoters, adding energy
    #rprm = ['rpsJ', 'rplK', 'rplN', 'rpsM', 'rpsL', 'rpsA', 'rpsA', 'rpsP', 'rpsT',
            #'rpsT', 'thrS', 'infC', 'infC', 'rpmI', 'rpoZ', 'rplJ', 'rpsU', 'rrnB']

    housekeeping = ('pgi', 'icd', 'arcA', 'aroE', 'rpoS', 'mdh', 'mtlD',
                    'gyrB')

    plt.ion()

    # collect the gene-> energy
    energies = {}

    its = 10
    for sigma, sigpath in sig_dict.items():

        if sigma not in ['Sigma70']:
            continue

        # you need to know the colum number for the 2d array
        row_nr = sum((1 for line in open(sigpath, 'rb')))
        # a shorter col_nr than 80 makes the sequence shorter in the 5' end
        col_nr = 80

        # dictionary for names with matrix for values
        vals = dict((name, np.zeros((row_nr, col_nr-1))) for name in names)
        fulls = dict((name, []) for name in names)

        # seq_logo = open(sigma+'_logo.fasta', 'wb')
        for row_pos, line in enumerate(open(sigpath, 'rb')):

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            if seq == '':
                continue

            #seq_logo.write('>{0}\n{1}\n'.format(pr_id, seq.upper()))
            #if 'without human' in evidence:
                #continue

            #if name[:3] in housekeeping or name[:4] in housekeeping:
                #pass
            #else:
                #continue

            # try to average if more than 1 promoter assigned to a gene
            if name.endswith('p'):
                energies[name[:-1]] = Ec.super_f(seq[-20:-10].upper())

            # there are 80 nt in the strand
            indiv = list(seq.upper()[-col_nr:]) # list of individual nucleotides
            dinucs = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]

            # calculate its of each promoter
            for name, func in name2func:
                if its < 20:
                    fulls[name].append(sum((func[din] for din in dinucs[-20:its-20])))
                elif its == 20:
                    fulls[name].append(sum((func[din] for din in dinucs[-20:])))


            # calculate the energies of all promoter dinucleotides
            for col_pos, dinuc in enumerate(dinucs):
                for name, func in name2func:
                    vals[name][row_pos, col_pos] = func[dinuc]

        #seq_logo.close()

        # get random its-len energies
        #rand_di_en, rand_its_en = get_random_energies(names, name2func, its)

        #for name, data in fulls.items():
            #print name, np.mean(data), np.std(data)

        #fug, ux = plt.subplots()
        #ux.hist(fulls['Combo'], bins=51, label = 'Combo')
        #ux.hist(fulls['Keq'], bins=51, label = 'Keq (translocation)')
        #ux.hist(fulls['RNA-DNA'], bins=51, label = 'RNA-DNA')
        #ux.set_title('"Energies" of first {0} nt of ITS'.format(its))
        #ux.legend()
        #print len([g for g in fulls['Combo'] if g < -31])/float(len(fulls['Combo']))
        #debug()

        # plot the dinucleotide energy of the promoters
        #dinuc_en_plot(vals, rand_di_en, sigma)


    # correlation between propensities and gene expression
    #expression_plot(energies)

    #ontology_plot(energies)

def ontology_plot(energies):

    ontology_handle = open('sequence_data/ecoli/genes.col', 'rb')
    ontology_handle.next() # skip header

    term2gene = {}
    gene2term = {}
    terms = {}

    for line in ontology_handle:
        #descr = ' '.join(line.split('\t')[-4:])
        term = line.split('\t')[-4]
        gene = line.split('\t')[2]

        if term in term2gene:
            term2gene[term].append(gene)
        else:
            term2gene[term] = [gene]

        if gene in gene2term:
            gene2term[gene].append(term)
        else:
            gene2term[gene] = [term]

        #print line.split('\t')
        #debug()
        if term in terms:
            terms[term] += 1
        else:
            terms[term] = 1

        #print line.split('\t')

    for t, v in terms.items():
        if v > 30:
            print t, v

    term2en = {}

    for term, genes in term2gene.items():
        if terms[term] > 30:

            # skip the empty one
            if term == '':
                continue

            if term not in term2en:
                term2en[term] = []

            for gene in genes:
                if gene in energies:
                    term2en[term].append(energies[gene])

    # filter out those with less than xx
    filteredterm2en = {}

    for term, ens in term2en.items():
        if len(ens) > 15:
            filteredterm2en[term] = ens

    boxes = filteredterm2en.values()
    boxes.append(energies.values())

    titles = filteredterm2en.keys()
    titles.append('All')

    fig, ax = plt.subplots()
    ax.boxplot(boxes)
    ax.set_xticklabels(titles, rotation=14)

    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

def expression_plot(energies):
    # get ribomomal genes
    ribs = []
    rib_handle = open('sequence_data/ecoli/ribosomal_genes.txt', 'rb')

    for line in rib_handle:
        if line.split()[4] == 'ribosomal':
            ribs.append(line.split()[2])

    # Read the gene -> DPKM values
    dpkm_handle = open('sequence_data/ecoli/expression/dpkm.txt', 'rb')
    dpkm_handle.next() # skip the header

    dpkms = {}

    for line in dpkm_handle:
        # some lines have 6 and some have 3 entries
        try:
            gene, dpkm, rpkm, d, d, d = line.split()
        except ValueError:
            gene, dpkm, rpkm = line.split()

        dpkms[gene] = float(dpkm)

    set1 = set(dpkms.keys())
    set2 = set(energies.keys())

    both = set1.intersection(set2)

    print 'dpkms len ', len(set1)
    print 'energies len ', len(set2)
    print 'both len ', len(both)

    # get the values of both
    ens = []
    dpk = []
    cols = []

    for name, en in energies.items():
        if name in dpkms:
            #if dpkms[name] > 500:
            ens.append(en)
            dpk.append(dpkms[name])

            # classify the ribosomes
            if name in ribs:
                cols.append(1)
            else:
                cols.append(0)
        else:
            ens.append(en)
            # add 0 if not found
            dpk.append(0)

            if name in ribs:
                cols.append(1)
            else:
                cols.append(0)

    # annotate the figure and move on to the ribosomal genes
    fig, ax = plt.subplots()

    color_en = [e for indx, e in enumerate(ens) if cols[indx]]
    color_dp = [e for indx, e in enumerate(dpk) if cols[indx]]

    corrs = spearmanr(ens, dpk)
    header = 'r = {0:.2f}, p = {1:.3f}'.format(corrs[0], corrs[1])
    ax.set_title(header, size=22)
    ax.set_xlabel('ITS abortive-initiation propensity', size=20)
    ax.set_ylabel('DPKM expression measure', size=20)
    fig.suptitle('No relationship between in vitro abortive propensity and'\
                 ' gene expression\n (ribosomal genes in red)', size=22)

    ax.scatter(ens, dpk)

    ax.scatter(color_en, color_dp, c='red')

    fig2, ax2 = plt.subplots()
    ax2.hist(ens, bins=40)

    # TODO read the gene file and extract ontological terms for each gene
    # plot the propensities for each class as box plots
    # when that is done you can be sure there is nothing more

def dinuc_en_plot(vals, rand_di_en, sigma):
    """
    """
    fig, ax = plt.subplots()

    for name, data_matrix in vals.items():
        means = np.mean(data_matrix, axis=0)

        # subtract the mean from the mean!
        # NOTE you should subtract the genome-wide mean
        norm_mean = means - rand_di_en[name][0]

        ax.plot(norm_mean, label=name, linewidth=2)

        #stds = np.std(data_matrix, axis=0)
        #ax.plot(stds, label=name + '_std')

    ax.set_xticks(range(0,81,10))
    xticklabels = [str(integer) for integer in range(0,81, 10)]
    # remove most vals
    newticks = []
    for tick in xticklabels:
        if int(tick) < 60:
            newticks.append(str(int(tick)-60))
        elif int(tick) >= 60:
            newticks.append('+'+str(int(tick)-60))

    newticks[6] = 'TSS'
    ax.set_xticklabels(newticks)

    ax.legend(loc='upper left')
    ax.set_title('Position averaged "energies" all E coli '+ sigma + ' promoters')

def get_random_energies(names, name2func, its):
    """
    Return the mean and std of energy functions at random dinucleotide sites and
    15-mers
    """
    # get energies for 10000 random locations in e coli genome
    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))
    drawlen = len(ecoli_seq) - its

    sample_nr = 10000
    random_di_vals = dict((name, []) for name in names)
    random_15_vals = dict((name, []) for name in names)

    for rnr in range(sample_nr):
        rand_pos = np.random.random_integers(drawlen)
        randDi = ecoli_seq[rand_pos:rand_pos+2]
        rand15 = ecoli_seq[rand_pos:rand_pos+its+1] # adjust by 1 because of dinucs

        for nme, func in name2func:
            random_di_vals[nme].append(func[randDi])

            # split up 
            indiv = list(rand15) # list of individual nucleotides
            dinucs = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]

            random_15_vals[nme].append(sum([func[nei] for nei in dinucs]))

    outp_di = {}
    outp_15 = {}
    for name in names:
        di_vals = random_di_vals[name]
        f15_vals = random_15_vals[name]

        outp_di[name] = (np.mean(di_vals), np.std(di_vals))
        outp_15[name] = (np.mean(f15_vals), np.std(f15_vals))

    return outp_di, outp_15

def make_models(ITSs, maxnuc):
    """
    Return basic DNA-DNA, RNA-DNA, Keq values + 3 models:

        1) Add 'RNA-DNA' and 'Keq' at the 3' end
        2) Add 'RNA-DNA' and 'Keq' at the 3' end and remove 'RNA-DNA' at the 5' end
        3) Add 'Keq' at the 3' end and remove 'RNA-DNA' at the 5' end
        4) Add 'RNA-DNA', 'Keq', and 'DNA-DNA' at the 3' end

    """

    if maxnuc == 'msat':
        rna_dna = [Ec.RNA_DNAenergy(s.sequence[:s.msat])/float(s.msat)
                   for s in ITSs]
        dna_dna = [Ec.DNA_DNAenergy(s.sequence[:s.msat])/float(s.msat)
                   for s in ITSs]
        keq = [Ec.Keq(s.sequence[:s.msat])/float(s.msat) for s in ITSs]

    else:
        rna_dna = [Ec.RNA_DNAenergy(s.sequence[:maxnuc]) for s in ITSs]
        dna_dna = [Ec.DNA_DNAenergy(s.sequence[:maxnuc]) for s in ITSs]
        keq = [Ec.Keq(s.sequence[:maxnuc]) for s in ITSs]

    its_nr = len(ITSs)

    # Model 1 is the sum of rna_dna
    model1 = [rna_dna[i] - keq[i] for i in range(its_nr)]

    # With model 2 you have to subtract the maxnuc-9 first RNA-DNA values
    # First calculate those values, then subtract them from model 1
    sub_me = [Ec.RNA_DNAenergy(s.sequence[:maxnuc-9]) for s in ITSs]
    model2 = [model1[i] - sub_me[i] for i in range(its_nr)]

    # With model 3 you should subtract sub_me only from keq
    model3 = [-keq[i] + sub_me[i] for i in range(its_nr)]

    # With model 4 add the dna-dna energy to model 1
    model4 = [model1[i] + dna_dna[i] for i in range(its_nr)]

    return rna_dna, dna_dna, keq, model1, model2, model3, model4

def new_scatter(lizt, ITSs):
    """
    Scatter plots at 5, 10, 13, 15, and 20

    Test 3 simple models:
        1) Add 'RNA-DNA' and Keq at the 3' end
        2) Add 'RNA-DNA' and Keq at the 3' end and remove 'RNA-DNA' at the 5' end
        3) Add only 'Keq' at the 3' end and remove 'RNA-DNA' at the 5' end
        4) Add 'DNA-DNA' to 1)

    Result: model 1) fares best.
    """

    #stds = 'no'
    stds = 'yes'

    plt.ion()

    #rows = [5, 10, 15, 20, 'msat']
    rows = [10, 15, 20]
    fig, axes = plt.subplots(len(rows), 3, sharey=True)

    for row_nr, maxnuc in enumerate(rows):
        name = '1_{0}_scatter_comparison'.format(maxnuc)

        rna_dna, dna_dna, keq, model1, model2, model3, model4 = make_models(ITSs, maxnuc)

        energies = [('RNA-DNA', rna_dna), ('Translocation', keq),
                    ('RNA-DNA - Translocation', model1)]
        #energies = [('Model1', model1), ('Model4', model4),
                    #('Model3', model3)]

        PYs = [itr.PY for itr in ITSs]
        PYs_std = [itr.PY_std for itr in ITSs]

        fmts = ['ro', 'go', 'bo']

        for col_nr, (name, data) in enumerate(energies):

            ax = axes[row_nr, col_nr]
            # can't use [0,4] notation when only 1 row ..
            if len(rows) == 1:
                ax = axes[col_nr]
            else:
                ax = axes[row_nr, col_nr]

            if stds == 'yes':
                ax.errorbar(data, PYs, yerr=PYs_std, fmt=fmts[col_nr])
            else:
                ax.scatter(data, PYs, color=fmts[col_nr][:1])

            corrs = spearmanr(data, PYs)

            if col_nr == 0:
                ax.set_ylabel("PY ({0} nt of ITS)".format(maxnuc), size=15)
            if row_nr == 0:
                #ax.set_xlabel("Abortive propensity", size=20)
                #header = '{0}\nr = {1:.2f}, p = {2:.1e}'.format(name, corrs[0], corrs[1])
                header = 'Spearman: r = {1:.2f}, p = {2:.1e}'.format(name, corrs[0], corrs[1])
                ax.set_title(header, size=15)
            else:
                header = 'r = {0:.2f}, p = {1:.1e}'.format(corrs[0], corrs[1])
                ax.set_title(header, size=15)

            # awkward way of setting the tick sizes
            for l in ax.get_xticklabels():
                l.set_fontsize(12)
                #l.set_fontsize(6)
            for l in ax.get_yticklabels():
                l.set_fontsize(12)
                #l.set_fontsize(6)

        fig.set_figwidth(2)
        fig.set_figheight(6)

    #for formt in ['pdf', 'eps', 'png']:

        #if stds == 'yes':
            #fname = name + '_stds.' + formt
        #else:
            #fname = name + '.' + formt

        #for fig_dir in fig_dirs:
            #odir = os.path.join(fig_dir, formt)

            #if not os.path.isdir(odir):
                #os.makedirs(odir)

            #fpath = os.path.join(odir, fname)

            #fig.savefig(fpath, transparent=True, format=formt)

def get_start_codons():
    """
    Return start codons sorted in two dicts
    """
    startC_path = '/home/jorgsk/phdproject/5UTR/hsuVitro/sequence_data/'\
            'ecoli/gtf/e_coli_start_codons.bed'

    mehdict = {'+': [], '-': []}

    for line in open(startC_path, 'rb'):
        (d, beg, end, strand) = line.split()
        if strand == '+':
            mehdict[strand].append(int(end))
        if strand == '-':
            mehdict[strand].append(int(beg))

    out_dict = {}
    for strand, starts in mehdict.items():
        out_dict[strand] = sorted(starts)

    return out_dict

def get_candidates(sigpath, start_codons, min_len=0, max_len=400):
    """
    Return for each promoter all the transcription start sites between min_len
    and max_len
    """

    candidates = {}
    for line in open(sigpath, 'rb'):

        (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
        if seq == '':
            continue

        # skip purely computationally derived ones
        #if "without human" in evidence:
            #continue

        candidates[name] = []

        TTS = int(pos)

        if strand == 'forward':
            # start_codons is sorted by lowest first
            for val in start_codons['+']:
                # skip until you get to the TTS + maybe a min len
                if val < TTS + min_len:
                    continue
                else:
                    # If you passed by more than max_len nt, just stop
                    if val - TTS > max_len:
                        break
                    else:
                        if val not in candidates[name]:
                            candidates[name].append(val)

        if strand == 'reverse':
            for val in reversed(start_codons['-']):
                # skip until you have reached min_len downstream the TTS
                if val > TTS - min_len:
                    continue
                else:
                    # if too far away, stop
                    if TTS - val > max_len:
                        break
                    else:
                        if val not in candidates[name]:
                            candidates[name].append(val)
                            # don't add twice

    return candidates

def new_long5UTR(lizt):
    """
    For each promoter's TTS, find the closest start codon. If it's further away
    than 40 nt, keep it for analysis.
    """
    sigprom_dir = 'sequence_data/ecoli/sigma_promoters'
    sig_paths = glob(sigprom_dir+'/*')

    # e coli genome
    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))

    # strand-dictionary for position of all start_codons in e coli
    start_codons = get_start_codons()

    #favored = ['TA', 'TC', 'TG']
    not_favored = ['AT', 'CT', 'GT']
    oppos = ['TA', 'TC', 'TG']

    dinucfreq = {}
    dinucfreq_random = {}
    counts = 0

    # Keq energy

    # paths for the sigma files
    sig_dict = {}
    for path in sig_paths:
        fdir, fname = os.path.split(path)
        sig_dict[fname[8:15]] = path

    for sigma, sigpath in sig_dict.items():

        candidates = get_candidates(sigpath, start_codons)

        favs = [0, 0, 0, 0]
        opposite = [0, 0, 0, 0]
        non_favs = [0, 0, 0, 0]
        #energy = [0, 0, 0, 0]

        for line in open(sigpath, 'rb'):

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            if seq == '':
                continue

            # skip if not there if not human curated or not found
            if name not in candidates:
                continue

            # parse over the regions (1,20), (21,40), (41, 60), (61, 80)
            # TODO ignore the first nucleotide; it's got nothing to ... no,
            # that's not right. The first will be important.
            # the first G/A is definitely driving your values, since
            # G/A-starting dinucleotides are generally not favored.
            # When I don't consider the first one, the bias is gone and there
            # are slightly more favored than non-favored
            # WHen I ignore the first nucleotide there is no difference
            # whatsoever.
            # NOTE hey, there is actually still a difference in the energy -->
            # its higher even if I exclude the 0th base. Anyway, find a way to
            # present this for each position and leave it like that. It's enough
            # that you explained Hsu's dataset.

            # Q: what about PPi concentations in vitro (hsu + translocatino
            # expreiment) contra in vivo?

            # What can you do about that? Maybe there is no such force acting.
            # This could mean that it plays a role in abortive initiation, but
            # gre-factors could rescue it, relieving the evolutionary pressure
            # on the ITS region.

            pos = int(pos)
            #for indx, reg in enumerate([(-1, 19), (19, 39), (39, 59), (59, 79)]):
            #for indx, reg in enumerate([(-1, 9), (9, 19), (19, 29), (29, 39)]):
            for indx, reg in enumerate([(0, 10), (10, 20), (20, 30), (30, 40)]):
                if strand == 'forward':
                    subseq = ecoli_seq[pos+reg[0]:pos+reg[1]]
                else:
                    continue

                if indx == 0 and subseq not in seq.upper():
                    debug()

                #energy[indx] += Ec.Keq(subseq)
                #energy[indx] += Ec.res_frac(subseq)

                idv = list(subseq) # splitting sequence into individual letters
                dins = [idv[cnt] + idv[cnt+1] for cnt in range(9)]

                for di in dins:
                    if di in not_favored:
                        non_favs[indx] += 1
                    if di in oppos:
                        opposite[indx] += 1
                    else:
                        favs[indx] += 1

                    if indx == 1:
                        if di in dinucfreq:
                            dinucfreq[di] += 1
                        else:
                            dinucfreq[di] = 1

                        counts += 1

                if indx == 1:
                    # shuffle idv nucletodies to shuffle the dinucletodies
                    random.shuffle(idv)
                    sdins = [idv[cnt] + idv[cnt+1] for cnt in range(9)]

                    for sd in sdins:
                        if di in dinucfreq_random:
                            dinucfreq_random[di] += 1
                        else:
                            dinucfreq_random[di] = 1

                # count how many of the favored dinucleotides
                #for f in favored:
                    #favs[indx] += din.count(f)

                #for n in not_favored:
                    #non_favs[indx] += din.count(n)


        #print sigma
        #print 'favs', favs
        #print 'non-favs', non_favs
        #print 'opposite', opposite
        #print 'Non-fav ratios:'
        #print np.array(non_favs)/np.array(favs)
        #print 'Opposite non-fav ratios:'
        #print np.array(opposite)/np.array(favs)
        #print ''
        #print energy
        #print ''
        #print ''

    # Do the correlation with both Keq and super for the dinucleotides from +1
    # to +10


    from Ec import resistant_fraction, super_en

    # check both original and shuffled data
    for dinucdict, headr in [(dinucfreq, 'Dinucleotides'), (dinucfreq_random,
                                                            'Shuffled dinucleotides')]:

        fig, axes = plt.subplots(2)
        titles = ['Resistant fraction', 'RNA-DNA + Translocation']
        labels = dinucfreq.keys()

        for ax, ydict, title in zip(axes, (resistant_fraction, super_en), titles):
            x, y = dinucfreq.values(), ydict.values()
            #x, y = dinucfreq_random.values(), ydict.values()
            ax.scatter(x, y)

            corrs = spearmanr(x, y)
            header = 'r = {0:.2f}, p = {1:.3f}'.format(corrs[0], corrs[1])
            ax.set_title(header, size=22)

            ax.set_ylabel(title, size=22)

            ymin, ymax = ax.get_ylim()
            ylen = np.abs(ymax - ymin)
            ax.set_ylim((ymin - ylen*0.1, ymax +ylen*0.4))

            # set circles
            for label, xx, yy in zip(labels, x, y):
                label = label.replace('T', 'U')
                colr = 'yellow'
                if label.endswith('U'):
                    colr = 'red'
                ax.annotate(label,
                        xy=(xx, yy), xytext=(-20,20),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round, pad=0.5', fc=colr, alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )

            # dinucleotide count at bottom
            axes[1].set_xlabel('Dinucleotide count +1,+10 of E coli promoters', size=22)

            fig.suptitle('{0}: no correlation between dinucleotide counts and resistant'\
                         ' fraction or optimal parameters'.format(headr), size=20)

def new_AP(ITSs):
    """
    Check of the abortive probabilities correlate with your dinucleotide
    propensities
    """
    abortive_path = 'Hsu_original_data/AbortiveProbabilities/'\
            'abortiveProbabilities__FIXED.csv'

    abort_handle = open(abortive_path, 'rb')

    promoters = abort_handle.next()
    prom_order = [p for p in promoters.split(',')[:-1] if p != '']

    abortProbs = dict((p, []) for p in prom_order)

    dummy = abort_handle.next()

    for line in abort_handle:
        # iterate over the promoter - AP data
        for prom, datum in zip(prom_order, line.split(',')[1::2]):
            if datum == '':
                abortProbs[prom].append('-')
            else:
                abortProbs[prom].append(float(datum))

    # add all abortives and all propensities
    #APs = []
    #props = []
    #pys = []

    nr= 0
    for its in ITSs:
        nr +=1
        if nr >10:
            break
        ap = []

        # not always agreement between ITS.msat and abortive probs
        abortz = [a for a in abortProbs[its.name] if a != '-']

        #get the keqs up to the length of abortiz
        keqs = its.keq_delta_di[:len(abortz)]

        #APs.append(np.std(ap))
        #props.append(np.std(keqs))
        #pys.append(its.PY)

        fig, ax = plt.subplots()
        ax.scatter(abortz, keqs)
        ax.set_title(its.name)

        #print spearmanr(ap, keqs)


def new_promoter_strength():
    """
    Promoter consensus scores have been published. Use them to compare with the
    propensity of the downstream ITS

    Map the transcription start site of the Prediction file to the stat sites of
    the verified promoters. Then just compare scores. You
    """

    promoter_scores = 'sequence_data/ecoli/PromoterPredictionSigma70Set.txt'
    promoters =  'sequence_data/ecoli/sigma_promoters/PromoterSigma70Set.txt'

    # First get the verified ones, and index them by their start site

    realz = {}

    for line in open(promoters, 'rb'):
        (reg_id, prom_id, strand, TSS, promoter, seq, info) = line.split('\t')
        realz[TSS] = (prom_id, strand, seq)

    checkme = {}
    for line in open(promoter_scores, 'rb'):
        if line.startswith('#'):
            continue

        (beg, end, name, strand, prom_name, TSS, score, d, d, d) = line.split()

        min1 = str(int(TSS)-1)
        min2 = str(int(TSS)-2)
        plus1 = str(int(TSS)+1)
        plus2 = str(int(TSS)+2)

        if TSS in realz:
            checkme[prom_name] = (score, realz[TSS][2])
        elif min1 in realz:
            checkme[prom_name] = (score, realz[min1][2])
        elif min2 in realz:
            checkme[prom_name] = (score, realz[min2][2])
        elif plus1 in realz:
            checkme[prom_name] = (score, realz[plus1][2])
        elif plus2 in realz:
            checkme[prom_name] = (score, realz[plus2][2])

    scores = []
    props = []

    for prom, (score, seq) in checkme.items():
        scores.append(float(score))

        propensity = Ec.super_f(seq[-20:-10].upper())

        props.append(propensity)

    print spearmanr(scores, props)
    debug()

def get_pr_nr(sig_paths, promoter):
    """
    Count the number of promoters with a 1promoter-1gene connection
    """
    proms = set([])

    for path in sig_paths:

        # 'promoter' is either the specific type of promoter you want, or it's
        # all promoters
        if 'all' in promoter:
            pass
        elif promoter not in path:
            continue

        for line in open(path, 'rb'):
            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            if seq == '':
                continue

            # add the gene name if if only 1 promoter for this gene
            if name.endswith('p'):
                proms.add(name[:-1])

    return proms

def get_expression(expr_path):
    """
    Return a dict with gene -> dpkm
    """
    # Read the gene -> DPKM values
    dpkm_handle = open(expr_path, 'rb')
    dpkm_handle.next() # skip the header

    dpkms = {}

    for line in dpkm_handle:
        # some lines have 6 and some have 3 entries
        try:
            gene, dpkm, rpkm, d, d, d = line.split()
        except ValueError:
            gene, dpkm, rpkm = line.split()

        dpkms[gene] = float(dpkm)

    return dpkms

def reverseComplement(sequence):
    complement = {'A':'T','C':'G','G':'C','T':'A','N':'N'}
    return "".join([complement[nt] for nt in sequence[::-1]])

def get_plus_20(pos, strand, ecoli):
    """
    Return the sequence +20 in the 3' direction of ops
    """

    if strand == 'forward':
        return ecoli[pos+20:pos+40]
    if strand == 'reverse':
        return reverseComplement(ecoli[pos-40:pos-20])

def get_reversed_data(promoter, enDict):
    """
    Return energy data for all promoters in a m X n matrix, where m is the
    number of promoters and n is the measure you want. This is to support a
    promoter-as-variable PCA.

    Proposed variables:

        x1 : promoter 1 ...
        .
        .
        .
        xn : promoter n

    Proposed samples:

        Energy of dinucleotide1
        Energy of dinucleotide2
        .
        .
        .
        Energy of dinucleotide39

    That means that the vectors I'll end up with will be linear combinations of
    genes, which explain the largest variation in the samples. This is some kind
    of clustering approach. They will be clustered if they have special patterns
    in the dinucleotide energies. Well well.
    """

    promoters_dir =  'sequence_data/ecoli/sigma_promoters/'
    sig_paths = glob(promoters_dir+'/*')
    expression_path = 'sequence_data/ecoli/expression/dpkm.txt'

    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))

    from Ec import super_en, Keq_EC8_EC9

    # get genes for which there is expression data
    expression = get_expression(expression_path)
    # TODO start here and get the new matrix up and running

    # get promoters for which there is a 1-1 promoter-gene relationsihp 
    promoters = get_pr_nr(sig_paths, promoter)

    # Get the promoters that have expressed genes
    #pr_remain = promoters.intersection(set(expression.keys()))

    # Optionally ignore the expression thing -- I think it's not related 
    pr_remain = promoters

    # You should return a m (promoter) x n (measures) matrix. Let's use the
    # Keq of the first 20 dinucleotdies, which will be 19

    #ind = list(sequence) # splitting sequence into individual letters
    #neigh = [ind[c] + ind[c+1] for c in range(len(ind)-1)]

    en_dict = enDict

    empty_matrix = np.zeros([len(pr_remain), 38])

    expre = []

    names = []

    # seq_logo = open(sigma+'_logo.fasta', 'wb')
    row_pos = 0
    for sigpath in sig_paths:
        for line in open(sigpath, 'rb'):

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')

            gene_name = name[:-1]
            pos = int(pos)

            if seq == '' or gene_name not in pr_remain:
                continue

            names.append(gene_name)

            ITSseq = seq.upper()[-20:]
            dins = [ITSseq[b] + ITSseq[b+1] for b in range(19)]

            ITSseqp20 = get_plus_20(pos, strand, ecoli_seq)
            dinsp20 = [ITSseqp20[b] + ITSseqp20[b+1] for b in range(19)]

            ens = [en_dict[d] for d in dins + dinsp20]

            empty_matrix[row_pos] = np.array(ens)

            if gene_name in expression:
                expre.append(expression[gene_name])
            else:
                expre.append(0)


            row_pos += 1

    return expre, empty_matrix, names


def its_data():
    """
    Just return the ITS data.

    Now outputting x1 separately from the others

    OUTDATED INFO:
    Do PCA on the promoter data. It's not sure what you're going to get out, but
    hopefully it will satisfy Martin to move on.

    Each column in your data-matrix should be some variable that could be
    connected. Proposed variables:

        x1  : Expression of downstream gene
        x2  : Keq of first 20 nucleotides
        x3  : RNA-DNA energy of first 20 nucleotides
        x4  : DNA-DNA energy of first 20 nucleotides
        x5  : *U-dinucleotide frequency in first 20 nucleotides
        x6  : Keq of +20 to +40 nucleotides
        x7  : RNA-DNA energy of +20 to +40 nucleotides
        x8  : DNA-DNA energy of +20 to +40 nucleotides
        x9  : *U-dinucleotide frequency of the +20 to +40 nucleotides

    Critisism of PCA approach:

        1) RNA-DNA and Keq are not normally distributed across promoters
        2) (Check if gene expression is normally distributed)
        3) *U Frequency is not normally distributed
        4) From in-vitro data: the relationship between RNA-DNA/Keq and abortive
        initiation is exponential -- not linear.

        PCA assumes each variable to be gaussian in distribution -- it assumes
        that the mean and variance are enough to explain the data. This is not
        true for the energy parameters.

        PCA also assumes a linear relationship between the variables. In vitro,
        the relationship between Keq and abortive initiation is nonlinear.

        Therefore, several of the assumptions behind the mathematics of PCA are
        broken, and we cannot trust the result as much.

        Further, the variance of the different measurements vary a lot.
    """

    promoters_dir =  'sequence_data/ecoli/sigma_promoters/'
    sig_paths = glob(promoters_dir+'/*')
    expression_path = 'sequence_data/ecoli/expression/dpkm.txt'

    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))

    from Ec import Keq, RNA_DNAenergy, DNA_DNAenergy

    # Prepare a data-matrix with 8 columns and as many sigma70 promoters as
    # there is expression data from for which only a single promoter is known

    # get genes for which there is expression data
    expression = get_expression(expression_path)

    # get promoters for which there is a 1-1 promoter-gene relationsihp 
    promoters = get_pr_nr(sig_paths, 'all')

    # Get the promoters that have expressed genes
    #pr_remain = promoters.intersection(set(expression.keys()))

    # Optionally ignore the expression thing -- I think it's not related 
    pr_remain = promoters

    empty_matrix = np.zeros([len(pr_remain), 9])

    names = []

    # seq_logo = open(sigma+'_logo.fasta', 'wb')
    row_pos = 0
    for sigpath in sig_paths:
        for line in open(sigpath, 'rb'):

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')

            gene_name = name[:-1]
            pos = int(pos)

            if seq == '' or gene_name not in pr_remain:
                continue

            names.append(gene_name)

            ITSseq = seq.upper()[-20:]
            dins = [ITSseq[b] + ITSseq[b+1] for b in range(19)]

            ITSseqp20 = get_plus_20(pos, strand, ecoli_seq)
            dinsp20 = [ITSseqp20[b] + ITSseqp20[b+1] for b in range(19)]

            # 1 expression of gene
            empty_matrix[row_pos,0] = expression[gene_name]
            #empty_matrix[row_pos,0] = 1 # ignoring this parameter anyway

            # 2 Keq of first 20 nucs
            empty_matrix[row_pos,1] = Keq(ITSseq)

            # 3 RNA-DNA of first 20 nucs
            empty_matrix[row_pos,2] = RNA_DNAenergy(ITSseq)

            # 4 DNA-DNA of first 20 nucs
            empty_matrix[row_pos,3] = DNA_DNAenergy(ITSseq)

            # 5 *U dinuc frequency in first 20 nucs
            empty_matrix[row_pos,4] = sum((1 for d in dins if d[1] == 'T'))/19

            # 6 Keq of 20:40 nucs
            empty_matrix[row_pos,5] = Keq(ITSseqp20)

            # 7 RNA-DNA of 20:40 nucs
            empty_matrix[row_pos,6] = RNA_DNAenergy(ITSseqp20)

            # 8 DNA-DNA of 20:40 nucs
            empty_matrix[row_pos,7] = DNA_DNAenergy(ITSseqp20)

            # 9 *U dinuc frequency in 20:40 nucs
            empty_matrix[row_pos,8] = sum((1 for d in dinsp20 if d[1] == 'T'))/19

            row_pos += 1 # increment to the next gene/promoter

    # Return as mXn matrix, with expression separate
    return names, empty_matrix[:,0].T, empty_matrix[:,1:].T

def frequency_change():
    """
    Difference in nucleotide and dinucleotide concentration in +1 to +10 and
    +10 to +20 and +20 to +40

    On a personal note, is there a preference for repeats in the sequences
    compared to normal? I see many repeats. How's this compared to random e coli
    seq? + 20? Check 3-4-5 repeats
    """
    # e coli sequence
    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))

    sigprom_dir = 'sequence_data/ecoli/sigma_promoters'
    sig_paths = glob(sigprom_dir+'/*')

    sig_dict = {}
    for path in sig_paths:
        fdir, fname = os.path.split(path)
        sig_dict[fname[8:15]] = path

    # Separate high-expressors from low-expressors
    # Read the gene -> DPKM values
    dpkm_handle = open('sequence_data/ecoli/expression/dpkm.txt', 'rb')
    dpkm_handle.next() # skip the header

    dpkms = {}

    for line in dpkm_handle:
        # some lines have 6 and some have 3 entries
        try:
            gene, dpkm, rpkm, d, d, d = line.split()
        except ValueError:
            gene, dpkm, rpkm = line.split()

        dpkms[gene] = float(dpkm)

    mean_dpkm = np.mean(dpkms.values())
    std_dpkm = np.std(dpkms.values())

    #print sum((1 for i in dpkms.values() if i < 100))

    # remember to do the forward/reverse thing ...
    seqs = []

    nr = 0

    repeaters = {}
    apeaters = {}

    for sigma, sigpath in sig_dict.items():

        #if sigma not in ['Sigma70']:
            #continue

        for row_pos, line in enumerate(open(sigpath, 'rb')):

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            if seq == '':
                continue

            # get only those with 1 promoter <-> 1 gene
            if not name.endswith('p'):
                continue

            gene_name = name[:-1]

            #if 'TTTTT' in seq.upper() and gene_name in dpkms:
                #repeaters[gene_name] = dpkms[gene_name]

            #if 'AAAAA' in seq.upper() and gene_name in dpkms:
                #apeaters[gene_name] = dpkms[gene_name]

            # get the expressed ones and those with low dpkm
            #if gene_name not in dpkms or dpkms[gene_name] < mean_dpkm:
            #if gene_name not in dpkms or dpkms[gene_name] > mean_dpkm:
                #continue

            nr += 1

            pos = int(pos)

            if strand == 'forward':
                seqs.append(ecoli_seq[pos-1:pos+40])

            elif strand == 'reverse':
                seqs.append(reverseComplement(ecoli_seq[pos-40:pos]))

    print np.mean(repeaters.values()), 'Ts'
    print np.mean(apeaters.values()), 'As'
    print mean_dpkm, 'mean'

    print nr
    # compare nucleotide and dinucleotide frequency between first +1, +10 and
    # +10 to +20, and then 

    # nucs
    nucs = ('G', 'A', 'T', 'C')

    # dinucs
    dinucs = tuple(set([''.join([n1, n2]) for n1 in nucs for n2 in nucs]))

    # repeats
    repeats = tuple((''.join([n for val in range(i)]) for n in nucs
                     for i in range(3,6)))

    # for each of the groups, make
    # make a 2X3 plot; each one with two bar-plots of the change in frequency of
    # all these parameters
    # No compare the ITS to +20, +40, shuffled, random.
    # put in dict like this nucs, dinusc, repeats are individual dicts with the
    # same keys :['ITS compared to +20 to +40'], ['ITS compared to shuffled
    # ITS'], ['ITS compared to random e coli 20-mers']
    # Actually make a 3X3, it will be easier to vizuzalizizuze

    # Note this is all very messy.
    data = dict()
    data['Nucleotides'] = [[0 for i in range(len(nucs))] for i in range(4)]
    data['Di-nucleotides'] = [[0 for i in range(len(dinucs))] for i in range(4)]
    data['Repeats'] = [[0 for i in range(len(repeats))] for i in range(4)]

    # Relate the data-entries with their respective dicst
    data2tuple = (('Nucleotides', nucs), ('Di-nucleotides', dinucs),
                 ('Repeats', repeats))

    # Test the +1 to +20 vs +10 to +20

    # caclulate the ITS, +20 to +40 and shuffled
    # Rely on the order of the original tuples for the order in the lists
    for seq in seqs:

        # Fill up 0 the 1 then 2 then 3 (cols first then row in 'data')
        #seq_parts = [seq[:20], seq[-20:]]
        seq_parts = [seq[:10], seq[10:20]]
        #seq_parts = [seq[:10], seq[-10:]]

        for indx, seq_part in enumerate(seq_parts):

            for key, tup in data2tuple:
                # add the count, but you must modify the list ...
                if key == 'Di-nucleotides':
                    i = list(seq_part) # splitting sequence into individual letters
                    lookup = [i[cnt] + i[cnt+1] for cnt in range(len(i)-1)]
                else:
                    lookup = seq_part

                for tup_ind, entry in enumerate(tup):
                    data[key][indx][tup_ind] += lookup.count(entry)

    # shuffle each of the ITS 'shufs' times and do the average count
    shufs = 50
    for seq in seqs:
        #its = list(seq[:20])
        its = list(seq[:10])

        for key, tup in data2tuple:
            # shuffle 100 times
            temp_shuff = [0 for i in range(len(tup))]
            for d in range(shufs):
                random.shuffle(its)
                if key == 'Di-nucleotides':
                    i = list(its) # splitting sequence into individual letters
                    lookup = [i[cnt] + i[cnt+1] for cnt in range(len(i)-1)]

                for tup_ind, entry in enumerate(tup):
                    temp_shuff[tup_ind] += ''.join(its).count(entry)

            # append the average count
            for tup_ind, entry in enumerate(tup):
                data[key][2][tup_ind] += temp_shuff[tup_ind]/float(shufs)

    # Average by seqlen
    seqslen = float(len(seqs))
    avrg_data = dict()
    for key, lists2avrg in data.items():
        avrg_data[key] = []
        for entry in lists2avrg:
            avrg_data[key].append([v/seqslen for v in entry])

    # Sample 10k random 20 mers and add them to the averaged ensemble
    rand_nr = 50000
    elen = len(ecoli_seq)-20
    randpos = (random.randrange(elen) for i in range(rand_nr))

    # make a generator of random sequences
    #rand_seqs = [ecoli_seq[pos:pos+20] for pos in randpos]
    rand_seqs = [ecoli_seq[pos:pos+10] for pos in randpos]

    for key, tup in data2tuple:
        temp_one = [0 for i in range(len(tup))]

        for rand_seq in rand_seqs:
            # add the count
            for tup_ind, entry in enumerate(tup):
                temp_one[tup_ind] += rand_seq.count(entry)

        for tup_ind, val in enumerate(temp_one):
            avrg_data[key][3][tup_ind] = val/float(rand_nr)

    # PLOT THE DAMN THINGS
    plt.ion()
    #plt.ioff()

    #fig, axes = plt.subplots(3,3)
    fig, axes = plt.subplots(2,3)

    #ind2title = ((0, 'ITS'), (1, '+20 to +40'), (2, 'Shuffled ITS'),
                  #(3, 'Random 20-mer'))
    #ind2title = ((0, '+20 to +40'), (1, 'Shuffled ITS'), (2, 'Random 20-mer'))
    ind2title = ((0, '+20 to +40'), (1, 'Shuffled ITS'))

    ind3title = dict((t1, t2) for t1, t2 in ind2title)
    da2tup = dict((t1, t2) for t1, t2 in data2tuple)

    for col_nr, (key, listz) in enumerate(avrg_data.items()):
        its_data = listz[0]

        for row_nr, title in ind2title:

            ax = axes[row_nr, col_nr]
            other_data = listz[row_nr+1]

            x_center = range(1, len(its_data)+1)

            # 16 ones. I will plot bars at 0.5 and 1 of with 0.5.
            its_xpos = [c-0.25 for c in x_center]

            ax.bar(its_xpos, its_data, width=0.25, label='ITS', color='g')

            if row_nr == 0:
                colr = 'c'
            if row_nr == 1:
                colr = 'b'

            ax.bar(x_center, other_data, width=0.25, label='ITS', color=colr)

            ax.set_xticks(x_center)

            ax.set_xticklabels(da2tup[key])

            ax.set_yticks([])

            if row_nr == 0:
                ax.set_title(key, size=20)

            if col_nr == 0:
                ax.set_ylabel(ind3title[row_nr], color=colr, size=18)

            if col_nr == 2:
                ax.set_xticklabels(da2tup[key], rotation=20)


def hsu_pca(lizt):
    """
    Run a PCA on RNA-DNA, DNA-DNA, Keq and Purine %
    """
    from operator import attrgetter
    # these are the attributes you want to extract
    #attrs = ('purine_count', 'DNADNA_en', 'RNADNA_en', 'super_en')
    #attrs = ('DNADNA_en', 'RNADNA_en', 'keq_en')
    attrs = ('RNADNA_en', 'keq_en')
    f = attrgetter(*attrs)
    seqs = {}

    # make a n x m matrix (m = 4, n = len(seqs)
    X = np.zeros([len(lizt), len(attrs)])

    its_len = 15

    for li in lizt:
        name = li[0]
        seq = li[1][:its_len]
        py = li[2]

        seqs[name] = [Sequence(seq, name), py]

    pys = []
    for row_nr, (name, (seqObj, py)) in enumerate(seqs.items()):
        # can you inita again witha shorter seq?

        # extract the relevant attributes and make as row in matrix
        X[row_nr] = np.array(f(seqObj))
        pys.append(py)

    # subtract the mean of nXm matrix and transpose
    Xn = (X - np.mean(X, axis=0)).T

    plt.scatter(*X.T, c=pys)
    #debug()

    # calculate covariance
    covXn = np.cov(Xn)

    # get eigenvectors of covariance
    eigvals, eigvecs = np.linalg.eig(covXn)

    E = eigvecs.T

    new_X = dot(E, X.T)

    plt.scatter(*new_X, c=pys)

    # plot the data in the new directions and color with py values

    # can we make weights based on the  PCA?
    # Then it should be
    # array([ 16.84,   0.36,   7.5 ])
    # array([[-0.32, -0.31, -0.9 ],
       #[-0.85,  0.5 ,  0.13],
       #[-0.41, -0.81,  0.42]])
       #array([[-0.97,  0.23],

    # just for 2 :
    #array([[-0.97,  0.23],
       #[-0.23, -0.97]])
    #ipdb> eigvals
    #array([  6.09,  15.39])


    #_sum = 15.39 + 6.1
    # 15.39/_sum = w1
    # 6.1/_sum = w2

    #super_f = w1*(rna_term*(-0.23) + Keq_term*(-0.97)
    # + w2*(rna_term*(-0.97) + Keq_term*(0.23)
    # =  rna_term*(w1(-0.23) + w2*(-0.97)) + Keq_term*(w1*(-0.97) + w2*(0.23)
    # RESULT no benefit from doing PCA weighting of the RNA-DNA and Keq
    # variables (or, you did it wrong, but hey, I did it.)

def get_activdowns():
    """
    Return lists of activated and downregulated genes for both wt and
    overexpressed GreA
    """
    activ_both = 'sequence_data/greA/activated_grA_native_and_overexpressed.txt'
    activ_onlyOver = 'sequence_data/greA/activated_grA_only_overexpressed.txt'
    down_both = 'sequence_data/greA/downRegulated_grA_native_and_overexpressed.txt'
    down_onlyOver ='sequence_data/greA/downRegulated_grA_only_overexpressed.txt'

    # get activated genes 
    activs = set([])
    for line in open(activ_both, 'rb'):
        activs.add(line.split()[0])
    for line in open(activ_onlyOver, 'rb'):
        activs.add(line.split()[0])

    # get downregulated genes
    downs = set([])
    for line in open(down_both, 'rb'):
        downs.add(line.split()[0])
    for line in open(down_onlyOver, 'rb'):
        downs.add(line.split()[0])

    return list(activs), list(downs)


def greA_filter():
    """
    Check the promoters for the genes which are sensitve to GreA. Maybe you'll
    find some super_f pattern there which you otherwise don't see. Compare both
    up and down groups to each other and to the genome average
    """

    # get GreA activated and downregulated genes
    activated, downregulated = get_activdowns()

    # get sigma-promoters
    sigprom_dir = 'sequence_data/ecoli/sigma_promoters'
    sig_paths = glob(sigprom_dir+'/*')

    sig_dict = {}
    for path in sig_paths:
        fdir, fname = os.path.split(path)
        sig_dict[fname[8:15]] = path

    # the length of the ITS you should consider
    its = 10
    energies = {'activated': [], 'down-regulated': [], 'others': []}
    sObjs = {}
    #genz = ['tnaA', 'cspA', 'cspD', 'rplK', 'rpsA', 'rpsU', 'lacZ', 'ompX']
    genz = ['ompX']

    dupes = {'activated': {}, 'down-regulated': {}, 'others': {}}
    # all start codons
    start_codons = get_start_codons()

    for sigma, sigpath in sig_dict.items():

        # get those genes that have a long (+40 nt) 5UTR
        # as not to compete with codon sequence and/or shine-dalgarno
        threshold = 40
        candidates = get_candidates(sigpath, start_codons, threshold)

        for row_pos, line in enumerate(open(sigpath, 'rb')):

            dupe = False

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            if seq == '':
                continue

            if name not in candidates:
                continue
            if candidates[name] == []:
                continue

            # as a first approximation, skip those that have more than 1
            # promoter
            if not name.endswith('p'):
                dupe = True

            # do a double check to see if you have acrDp, acrDp2 or acrDp12
            if name[-1] != 'p':
                gene_name = name[:-2]
                if gene_name[-1] == 'p':
                    gene_name = gene_name[:-1]

            else:
                gene_name = name[:-1]

            sequence = seq[-21:].upper()

            seqObj = Sequence(sequence[:its], name=gene_name, shuffle=False)

            sObjs[seqObj.name] = seqObj

            if not dupe:
                if gene_name in activated:
                    energies['activated'].append(seqObj.super_en)
                elif gene_name in downregulated:
                    energies['down-regulated'].append(seqObj.super_en)
                else:
                    energies['others'].append(seqObj.super_en)
            else:
                if gene_name in activated:
                    if gene_name in dupes['activated']:
                        dupes['activated'][gene_name].append(seqObj.super_en)
                    else:
                        dupes['activated'][gene_name] = [seqObj.super_en]

                elif gene_name in downregulated:
                    if gene_name in dupes['down-regulated']:
                        dupes['down-regulated'][gene_name].append(seqObj.super_en)
                    else:
                        dupes['down-regulated'][gene_name] = [seqObj.super_en]
                else:
                    if gene_name in dupes['others']:
                        dupes['others'][gene_name].append(seqObj.super_en)
                    else:
                        dupes['others'][gene_name] = [seqObj.super_en]
                # add to dupe dict for later resolving

    mean_std= dict((k, (np.mean(vals), np.std(vals))) for k, vals in
                 energies.items())
    median_std= dict((k, (np.median(vals), np.std(vals))) for k, vals in
                 energies.items())

    # NOTE there is a small difference, but it's not much.
    # Idea for testing also the genes with more than 1 promoter. Choose the one
    # with highest/lowst super_en, but do it in a way that is comparable with
    # the others! Like: for all of them, choose the lowest or highest. Maybe
    # then ... if you even out the effect, you'll get the p-value benefit.
    #mins = {}
    #maxes = {}
    #for key, subdict in dupes.items():
        #mins[key] = []
        #maxes[key] = []
        #for gene, vals in subdict.items():
            #mins[key].append(min(vals))
            #maxes[key].append(max(vals))

    fig1, ax1 = plt.subplots()

    cols = ['b', 'g', 'r']

    for ind, (name, ens) in enumerate(energies.items()):
        if cols[ind] == 'r':
            al = 0.2
        else:
            al = 0.9
        ax1.hist(ens, color=cols[ind], label=name, alpha=al)

    ax1.legend()

    # tnaA, cspA, cspD, rplK, rpsA and rpsU as well as lacZ. How do these stand
    # out? Result: no clear pattern. 
    # NOTE: The activated are the ones that are activated in the greA+, which
    # means that GreA are important for their expression. This makes the most
    # sense.

    alpha = 0.02

    from itertools import combinations
    testers = ['activated', 'others', 'down-regulated']

    plotdict = {}

    for key1, key2 in combinations(testers, r=2):

        print "{0} vs {1}".format(key1, key2)
        n1 = len(energies[key1])
        mean1, std1 = mean_std[key1]
        sem1 = std1/np.sqrt(n1)
        n2 = len(energies[key2])
        mean2, std2 = mean_std[key2]
        sem2 = std2/np.sqrt(n2)
        welchs_approximate_ttest(n1, mean1, sem1, n2, mean2, sem2, alpha)

        plotdict[key1] = (mean1, std1)
        plotdict[key2] = (mean2, std2)
        print ''
        print ''

    order = testers
    heightsYerr = [plotdict[o] for o in order]
    heights, errs = zip(*heightsYerr)
    xpos = range(1,4)
    wid = 0.5

    plt.ion()

    heights = [-h for h in heights]

    fig, ax = plt.subplots()
    rects = ax.bar(xpos, heights, width=wid, yerr=errs, align='center')

    ax.set_xlim(1-wid,3+wid)

    ax.set_xticks(xpos)
    ax.set_xticklabels(order)

    stardict = {'activated': '**\n*', 'others': '*', 'down-regulated': '**'}

    for indx, name in enumerate(order):
        xpos = indx + 1
        height = -plotdict[name][0]
        std = plotdict[name][1]

        ax.text(xpos, height+std + 2, stardict[name], ha='center', va='center')

    ax.set_ylim(0,30)

    ax.set_title('Comparison of abortive initiation scores for E coli'\
                 ' promoters\nsignificance: * $p = 0.02 and ** $p = 0.005$')
    ax.set_ylabel('Abortive initiation score')
    ax.set_xlabel('GreA activated, other, and GreA down-regulated genes')

    #def autolabel(rects):
    ## attach some text labels
    #for rect in rects:
        #height = rect.get_height()
        #ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                #ha='center', va='bottom')

    # The result is not strong, but you can probably find that it's
    # "statistically significant". Or actually I'm not sure: both results are
    # well within the standard deviations of both.
    # RESULT you see something that makes sense finally.

    # the attributes you'd like
    #attrs = ('DNADNA_en', 'RNADNA_en', 'keq_en', 'super_en')

    # are the genes regulated by GreA different in the # of promoters compared
    # to random genes?
    debug()

def my_pca():
    """
    PCA. My way.

    UPD: add the information about expression

    UPD2: Martin actually meant that I should use the promoters as VARIABLES!
    This would make the energy calculations the samples.

    The logic isn't so straight forward, but I imagine it like this:

        instead of X1 = Energy, X1(P1) = Energy of P1

        I have to think like this:

            P1 = Promoter

            P1(X1) = X1(P1) = Energy of P1

        Maybe it doesn't matter for the mathematics, but it's kinda crazy

    NOTE: for some reason the eig doesn't give the same eigenvalues as svd.
    sticking with svd as recommended by people.
    """

    # Get the data matrx; it should be in m X n orientation
    # where m are the variables (energy ...) and n are the samples (promoters
    # ...)
    #X = np.array([[11,3,66,7,4,5], [33,4,2,66,7,8], [55,7,2,99,22,2]])
    #expression, X = its_data()

    # XXX new, get data where things are reversed. Now, this m X n matrix should
    # have m variables -- the promoters! And n samples -- the measures!
    # Personally this is very strange, but it could find co-varying patterns in
    # the genes, but really, it's nuts. The interpretation is not clear at all.

    from Energycalc import super_en, Keq_EC8_EC9
    expression, X, names = get_reversed_data('all', super_en)

    log2exp = np.log2(expression)

    m, n = X.shape
    # Transpose to subtract means easily. Then revert back.
    #Xn = (X.T - np.mean(X.T, axis=0)).T
    Xn = (X.T - np.mean(X.T, axis=0))

    U,sgma,V = np.linalg.svd(Xn, full_matrices=False)
    #U,sgma,V = np.linalg.svd(CovXn, full_matrices=False)
     #Then U.T is the same as rowEig
    var_sum = sum(sgma)
    cum_eig = [float(format(sum(sgma[:i])/var_sum, '.2f'))
               for i in range(1, len(sgma)-1)]

    ########## Write the dataset for Martin ##########
    #for promoter in ['Sigma70', 'all_promoters']:
        #for energy_term, enDict in [('Translocation', Keq_EC8_EC9),
                                    #('Combination', super_en)]:

            #expression, X, names = get_reversed_data(promoter, enDict)
            #debug()

            #outfile = 'output/{0}_{1}.txt'.format(energy_term, promoter)
            #handle = open(outfile, 'wb')
            #for row, name in zip(X, names):
                #handle.write('{0}\t'.format(name))
                #handle.write('\t'.join([str(r) for r in row]) + '\n')

            #handle.close()

    #debug()


    ######################################

    debug()
    # new method
    #from pca_class import PCA, Center
    #pc = PCA(CovXn, fraction=1)
    #pc = PCA(Xn, fraction=1)

    # Re-represent the data in the new basis
    #Y = np.dot(U, X.T)

    plt.ion()
    fig, axes = plt.subplots(2)

    #axes[0].scatter(Y[0,:], Y[1,:], c=log2exp)
    axes[0].scatter(Y[0,:], Y[1,:])
    axes[0].set_ylabel('PC 1')
    axes[0].set_xlabel('PC 2')

    #axes[1].scatter(Y[2,:], Y[3,:], c=log2exp)
    axes[1].scatter(Y[2,:], Y[3,:])
    axes[1].set_ylabel('PC 3')
    axes[1].set_xlabel('PC 4')

    fig2, ax2 = plt.subplots()
    ax2.plot((range(1, len(cum_eig)+1)), [1-float(c) for c in cum_eig])
    ax2.set_xlabel('Principal component')
    ax2.set_ylabel('Fraction of variance explained')
    ax2.set_ylim(0,1)
    ax2.set_xlim(0.5,len(cum_eig)+0.5)

#PCe1 = [ 0.67,  0.37,  0.35,  0.02,  0.39,  0.28,  0.26,  0.01]
#PCe2 = [ 0.49,  0.18,  0.16,  0.01, -0.64, -0.41, -0.35, -0.01]
#PCe3 = [-0.34,  0.42,  0.23, -0.01, -0.56,  0.52,  0.26, -0.01]
#PCe4 = [-0.43,  0.59,  0.34, -0.01,  0.33, -0.45, -0.21,  0.01]


    # Plot the first two components against each other,
    # Then the second two

def pca_eig(orig_data):
    data = np.array(orig_data)
    data = (data - data.mean(axis=0))
    #C = np.corrcoef(data, rowvar=0)
    C = np.cov(data, rowvar=0)
    w, v = np.linalg.eig(C)
    print "Using numpy.linalg.eig"
    print w
    #print v

def pca_svd(orig_data):
    data = np.array(orig_data)
    data = (data - data.mean(axis=0))
    C = np.corrcoef(data, rowvar=0)
    u, s, v = np.linalg.svd(C)
    print "Using numpy.linalg.svd"
    #print u
    print s
    #print v

def new_designer(lizt):
    """
    Design sequences to span PY from 0 to 10 in the 0-13 range

    Is there a way to quantify when 90% of the abortive transcription is over?
    Maybe by the imagequant? Maybe you'll find a correlation. What would that
    prove exactly? The msat? How? At each point there is a stochastic chance
    of falling off. MSAT means that the active site has not fallen off yet. It
    could be that MSAT is higher for those with weak PY? But then I should
    have seen it. Or maybe not use MSAT but use 90% of MSAT. You might then
    find that 
    What do I expect, biochemically? I expect that MSAT is limited by an
    energy barrier that RNAP is unlikely to pass while still attached to
    promoter.
    This suggests that it is not the "free energy" held up in scrunching, but

    The model is that RNAP has at any point a chance of aborting, as long as
    it's attached to SIGMA. Mayby thermodynamic fluctuations play a part,
    making the process stochastic in nature. The determining factor then
    becomes the ease at which RNAP synthisizes the first XX nucleotides. Then
    sigma release is also stochastic. Sigma is pulling backward and RNAP is
    pulling forward. The probability of each one winning is the ease at which
    RNAP translocates the first 10-15 nucleotiedes. But pulling? RNAP is
    standing still at the promoter. However, if the DNA-DNA buildup is not the
    case, what is? I should re-read that article. That's for T7 though. What
    is the stressor? Why escape? Is this important only for strong promoters?
    If RNAP continues straight from initiation to escape, what is escape? Why
    does sigma fall off? Or why does sigma dissasociate from promoter
    contacts? That's the most relevant question.

    They say that rnap moves by random fluctuations that occasionally becomes
    blocked backwards by nucleotide incorporations. The free energy of nt
    incorporation should also somehow be a driving force, even if it's not
    well explained.

    Design sequences which are both AT rich but have opposite effects on
    transcription initiation, showing that it's not just the AT-richness, but
    the specific order of the AT-richness
    What about that -10 like element? Is that a particularly weak one? Would
    be interesting if it were the weakest possible association of these
    values. It's TATAAT.

    The 10 pause site had more variation in RNA-DNA than in super_en, so I'm
    ruling it out as a super_en effect.

    RNAP is using force from nucleotide incorporation to propel itself?
    Conclusion of 1998 Science.

    How many possible combinations are there? If I vary from 3 to 15? 12. 4**12
    is 1.6 million :S But actually that is not so much. I'm doing dinucleotiding
    and dictionary lookup. I think I can check them all out.

    PyPy would be great for this! PyPy speeds the process up 2fold. (Which is
    like getting a 8hz processor ... ) 2fold is not enough for a serious speed
    increase.
    """

    plt.ion()

    beg = 'AT'
    #N25 = 'ATAAATTTGAGAGAGGAGTT'
    N25 = 'ATAAATTTGAGAGAG' # len 15 variant for comparing energies

    repeat_nr = 5 # total number of sequences allowed to vary
    sample_nr = 26
    at_test_nr = 8 # reserve a few samples for special testing

    min20, max20, boxed_seqs = seq_parser(beg, N25, repeat_nr, sample_nr,
                                          at_test_nr)

def seq_generator(variable_nr, batch_size, DNADNAfilter, beg=False,
                  starts=False):
    """
    Yield sequences in batches (list of sequences of length batch_size)

    stricht will add the 'beg'. If 'starts', add the new seq to each start
    """

    buff_count = 0
    batch = []

    repReg = re.compile('G{5,}|A{5,}|T{5,}|C{5,}')
    nucleotides = set(['G', 'A', 'T', 'C'])

    # a trick so that you don't have to change the code a lot to include leading
    # sequences
    if not starts:
        starts = ['']

    for seq in itertools.product(nucleotides, repeat=variable_nr):

        for pre_seq in starts:
            # If having supplied a start ('AG' for example)
            if beg:

                sequence = beg + ''.join(seq)

                # don't let the first three nucs be the same
                if (sequence[1] == sequence[2] == sequence[3]):
                    continue

                # Don't let the 2,3,4 pair be identical either
                if (sequence[2] == sequence[3] == sequence[4]):
                    continue

            # if not beg, you should just add the pre_seq
            else:
                # second round -> add the 'rest'
                # pre_seq is empty unless start_seqs have been given as input
                sequence = pre_seq + ''.join(seq)

                # If DNADNAfilter, screen for DNA-DNA energy; not less than -18
                # for all 15 nt.
                if DNADNAfilter and len(sequence) == 15:
                    if Ec.DNA_DNAenergy(sequence) < -18:
                        continue

            # count the nucleotide frequencies; 
            slen = float(len(sequence))
            nfreqs = [sequence.count(n)/slen for n in nucleotides]
            # none should be above 60%
            if max(nfreqs) > 0.6:
                continue

            # don't allow repeats of more than 4
            if repReg.search(sequence):
                continue

            # add sequences to the batch until you reach the set batch size
            if buff_count < batch_size:
                batch.append(sequence)
                buff_count += 1

            else:
                yield batch
                batch = []
                buff_count = 0


def seq_parser(beg, N25, repeat_nr, sample_nr, at_test_nr):
    """
    """

    nucleotides = set(['G', 'A', 'T', 'C'])

    free_sample = sample_nr - at_test_nr

    end = N25[2 + repeat_nr:]

    energies = []
    seqs = []

    import time

    t1 = time.time()

    repReg = re.compile('G{5,}|A{5,}|T{5,}|C{5,}')

    for seq in itertools.product(nucleotides, repeat=repeat_nr):

        # don't let the first two or two and three nucs be the same
        if (seq[0] == seq[1] == 'T') or (seq[0] == seq[1] == seq[2]):
            continue

        sequence = beg + ''.join(seq) + end

        # don't allow repeats of more than 4
        if repReg.search(sequence):
            continue

        energies.append(Ec.super_f(sequence))

        seqs.append(sequence)

    print time.time() -t1

    #plt.hist(energies, bins=70)

    minEn, maxEn = min(energies), max(energies)

    # the boundary for the top/min 10% boxes
    # you'll need 20% otherwise I don't think you'll find good candidates
    minLim = minEn - minEn*0.2
    maxLim = maxEn + maxEn*0.2

    # Reverse the enrange, and leave out the most extreme values
    enRange = np.linspace(maxEn, minEn, free_sample)[::-1]

    boundaries = []
    for inx, val in enumerate(enRange[:-1]):
        boundaries.append((val, enRange[inx+1]))

    # make boxes somehow and keep a counter in the loop to tell you how far
    # you've gone
    pos = 0

    min10 = []
    max10 = []

    savers = [[] for i in range(free_sample-1)]

    gokk = sorted(zip(energies, seqs))

    for en, seq in gokk:
        if en < minLim:
            min10.append((en, seq))
        if en > maxLim:
            max10.append((en, seq))

        # change pos when you reach a boundary
        if en > boundaries[pos][1] and pos != len(savers)-1:
            pos += 1

        if boundaries[pos][0] <= en <= boundaries[pos][1]:
            savers[pos].append((en, seq))

    # what now? randomly sample to get one sequence from each 'box'

    # first, try to get some with a high AT count in the randomized region

    for seqLow in min10:
        subseq = seqLow[2:]

def naive_energies(ITSs, new_set=False):
    """
    Return the energy range of the ITSs for the naive model. If a new_set is
    provided, return the range for those values in the same order in which they
    came in. If new_set is False, return the sorted order for the ITS sequences.

    Somehow I want to know what PY value they would be associated with. To do
    that I'll have to return the its energies with PYs so I can compare.
    OKdothat.
    """

    ITSlen = 15

    if not new_set:
        # get the variables you need 
        #rna_dna = np.array([Ec.RNA_DNAenergy(s.sequence[:ITSlen]) for s in ITSs])
        dna_dna = np.array([Ec.DNA_DNAenergy(s.sequence[:ITSlen]) for s in ITSs])
        keq_delta = np.array([Ec.Delta_trans(s.sequence[:ITSlen]) for s in ITSs])
    else:
        dna_dna = np.array([Ec.DNA_DNAenergy(s[:ITSlen]) for s in new_set])
        keq_delta = np.array([Ec.Delta_trans(s[:ITSlen]) for s in new_set])

    #### the exp model with log(keq) and /RT
    parameters = (10, 0.1, 0.1)
    RT = 1.9858775*(37 + 273.15)/1000   # divide by 1000 to get kcalories
    c1, c2, c3 = parameters

    exponential = (c2*dna_dna + c3*keq_delta)/RT
    #exponential = (c2*rna_dna + c3*keq_delta)/RT
    outp = c1*np.exp(exponential)
    PYs = np.array([itr.PY for itr in ITSs])

    if new_set:
        return outp
    else:
        return zip(outp, PYs)

def optim_naive_model(its_len, PYs, rna_dna, dna_dna, keq_delta):
    """
    Simple optimizer. Assume its_len is 20.
    """

    results = []

    RT = 1.9858775*(37 + 273.15)/1000  # divide by 1000 to get kcalories

    par_range = np.linspace(-1.5, 1.5, 10)

    for params in itertools.product(par_range, repeat=3):

        c2, c3, c4 = params

        #DD =
        # you stop right here since you found out you'd been using
        # rna-dinucleotides for keq

        exponential = (c2*dna_dna + c3*keq_delta)/RT + c4*rna_dna

def naive_model(ITSs, new_set=[]):
    """
    Plot the ITS according to the naive model

        PY = c1*exp(b1*DNADNA + b2*delta_KEQ)

    """

    ITSlen = 20
    # get the variables you need 
    rna_dna = np.array([Ec.RNA_DNAenergy(s.sequence[:ITSlen]) for s in ITSs])
    dna_dna = np.array([Ec.DNA_DNAenergy(s.sequence[:ITSlen]) for s in ITSs])
    keq_delta = np.array([Ec.Delta_trans(s.sequence[:ITSlen]) for s in ITSs])
    # PYs
    PYs = np.array([itr.PY for itr in ITSs])

    # optimize parameters XXX unfinished, maybe you can delete this later.
    output, opt_param = optim_naive_model(ITSlen, PYs, rna_dna, dna_dna,
                                          keq_delta)

    #exponential = (c2*dna_dna + c3*keq_delta + c4*rna_dna)/RT
    exponential = (c2*dna_dna + c3*keq_delta)/RT + c4*rna_dna
    outp = c1*np.exp(-exponential)

    # Make a continuous energy values for plotting
    minval = min(exponential)
    maxval = max(exponential)
    plot_range = np.arange(minval, maxval, 0.01)

    plot_data = c1*np.exp(plot_range)

    # PYs
    PYs = np.array([itr.PY for itr in ITSs])
    PYs_std = [itr.PY_std for itr in ITSs]
    names = [itr.name for itr in ITSs]

    print spearmanr(PYs, outp)
    print pearsonr(PYs, outp)

    plt.ion()

    fig, ax = plt.subplots()
    ax.errorbar(c1*exponential, PYs, yerr=PYs_std, fmt='go')
    #for name, py, fit in zip(names, PYs, c1*exponential):

        ## only annotate these
        #if name not in ('N25', 'DG115a', 'DG133', 'N25/A1anti'):
            #continue

        #ax.annotate(name,
                #xy=(fit, py), xytext=(-20,20),
                #textcoords='offset points', ha='right', va='bottom',
                #bbox=dict(boxstyle='round, pad=0.5', fc='g', alpha=0.9),
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # define the x-values for plot_data
    xbeg = min(c1*exponential)
    xend = max(c1*exponential)
    x_range = np.linspace(xbeg, xend, len(plot_data))

    ax.plot(x_range, plot_data)

    ax.set_xticklabels([])
    ax.set_xlabel('Abortive propensity', size=20)
    ax.set_ylabel('Productive yield', size=20)
    ax.set_title('Fitting $PY = Ke^{-(b1\Delta G_1 + b2\Delta G_2)}$')

def fitting_models(lizt, ITSs):
    """
    You have implicitly assumed the linear relationship with unity-parameters
    for the relationship between PY and the variables you consider. What if the
    relationship is nonlinear?

    RESULT:

        Assume that the probability of aborting at each step is exp(xi) where xi
        is the rna-dna or whatev for dinucleotide i. Multiplying these give
        exp(sum_i(xi)) which is the model we're fitting

        Observation: RNA-DNA + Keq has marginal improvement by adding DNA-DNA
        alone
        Keq + DNA-DNA has large improvement by adding RNA-DNA

        You see, RNA-DNA and DNA-DNA are naturally correlated variables

        By itself, RNA-DNA correlates with PY, but not DNA-DNA.

        This points to RNA-DNA ans the causal factor and not DNA-DNA.

        However, it seems that it is a linear combination of RNA-DNA and Keq
        that best explains the values. In the standard models RNA-DNA is
        expontential and Keq is linear. This will cause loss-of-correlation in
        the ODE system. It will be solved if you make k3 linear in terms of
        RNA-DNA; but this doesn't fit well with the arrhenius equation. Maybe
        though you can make a taylor approximation to the first or second
        degree? At least you can use that to argue your case.
    """

    ITSlen = 15
    # get the variables you need 
    rna_dna = np.array([Ec.RNA_DNAenergy(s.sequence[:ITSlen]) for s in ITSs])
    dna_dna = np.array([Ec.DNA_DNAenergy(s.sequence[:ITSlen]) for s in ITSs])
    keq = np.array([Ec.Keq(s.sequence[:ITSlen]) for s in ITSs])
    k1 = np.array([Ec.K1(s.sequence[:ITSlen]) for s in ITSs])
    kMinus1 = np.array([Ec.Kminus1(s.sequence[:ITSlen]) for s in ITSs])

    added = [Ec.super_f(s.sequence[:ITSlen]) for s in ITSs]
    # your Y-values, the PYs
    PYs = np.array([itr.PY for itr in ITSs])
    PYs_std = [itr.PY_std for itr in ITSs]
    names = [itr.name for itr in ITSs]

    import scipy.optimize as optimize

    #variables = (k1, rna_dna)
    #variables = (kMinus1, rna_dna)

    #### the exp model
    #parameters = (100, -0.1, 0.5)
    #variables = (keq, rna_dna)

    #plsq = optimize.leastsq(Models.residuals_mainM, parameters,
        #args=(variables, PYs))
    #fitted_parm = plsq[0]
    #outp = Models.mainM(fitted_parm, variables)
    #fitted_vals = np.dot(fitted_parm[1:], variables)
    #c1 = fitted_parm[0]
    #plot_data = c1*np.exp(plot_range)

    #### the log(keq) model
    #parameters = (100, 0.1, 0.1)
    #variables = (keq, rna_dna)

    #plsq = optimize.leastsq(Models.residuals_logKeq, parameters, args=(variables, PYs))
    #fitted_parm = plsq[0]
    #outp = Models.logKeq(fitted_parm, variables)
    #c1, b1, b2 = fitted_parm
    #fitted_vals = b1*rna_dna - b2*np.log(keq)
    #minval = min(fitted_vals)
    #maxval = max(fitted_vals)
    #plot_range = np.arange(minval, maxval, 0.01)
    #plot_data = c1*np.exp(plot_range)
     #RESULT log(keq) model has better fit visually
     # It corresponds to a k1*Keq equilibrium constant
     # ACtualy .. the Keq^k1 fits best, and gives a k1 value of 1.97. Bening.
     # The k1*Keq gives a value of exp(28) ... not benign. Which model is more
     # likely? The exp model gives too high values for .. what about a
     # combination? A combination changes little for b3; it doesn't improve the
     # fit. A good sign.

    #### the exp model with log(keq) and /RT
    parameters = (10, -1, 1)
    #parameters = (10, -1)
    #parameters = (10,)
    variables = (rna_dna, keq)

    plsq = optimize.leastsq(Models.residuals_logRT, parameters, args=(variables, PYs))
    fitted_parm = plsq[0]
    outp = Models.logRT(fitted_parm, variables)

    #c1, c2 = fitted_parm
    #c1 = fitted_parm
    #KT = 2500
    #fitted_vals = keq/KT + np.log(keq)
    #minval = min(fitted_vals)
    #maxval = max(fitted_vals)
    #plot_range = np.arange(minval, maxval, 0.01)
    #plot_data = c1*np.exp(plot_range)

    print fitted_parm
    print spearmanr(PYs, outp)
    print pearsonr(PYs, outp)

    debug()

    plt.ion()

    fig, ax = plt.subplots()
    ax.errorbar(c1*fitted_vals, PYs, yerr=PYs_std, fmt='go')
    for name, py, fit in zip(names, PYs, c1*fitted_vals):

        # only annotate these
        if name not in ('N25', 'DG115a', 'DG133', 'N25/A1anti'):
            continue

        ax.annotate(name,
                xy=(fit, py), xytext=(-20,20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round, pad=0.5', fc='g', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # define the x-values for plot_data
    xbeg = min(c1*fitted_vals)
    xend = max(c1*fitted_vals)
    x_range = np.linspace(xbeg, xend, len(plot_data))

    ax.plot(x_range, plot_data)


    ax.set_xticklabels([])
    ax.set_xlabel('Abortive propensity', size=20)
    ax.set_ylabel('Productive yield', size=20)
    ax.set_title('Fitting $PY = Ke^{-(b1\Delta G_1 + b2\Delta G_2)}$')

    ## clumsy way of starting 3dplot but ...
    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(rna_dna, keq, PYs, zdir='z')

    debug()

def new_models(ITSs):
    """
    Define a linear ODE system that solves the movement on RNAP on the ITS.
    Optimize the system based on 3 parameters that will quantify the effect of
    RNA-DNA and Keq translocation bias.

    Your original discovery is a linear relationship between Keq and RNA-DNA.
    You'll need to recreate this in the rate equations somehow.

    The system is dX = AX

    How should the models be benchmarked? Currently, I'm running them for a
    certain amount of time and correlating the concentration at i = 20 to the
    PYs. This is not optimal, because many escape long before 20, so they never
    each 20 in scrunched conditions. Most reach 14 so this could be a trade-off.

    Are there methods that are more intuitive? The PY value is a measure of how
    much transcript is full length compared to aborted.

    Models:
    #normal_obj = scruncher(PYs, its_len, ITSs, ranges)

    Your model is as follows:
        c1*exp(-c2*rna_dna + c3*dna_dna + c4*keq)

    # rna_dna is stabilizing, whreas dna_dna and keq are destabilizing (keq is
    # for reverse translocation)

    # modeling shows that c2 goes to zero.

    """
    # Compare with the PY percentages in this notation
    PYs = np.array([itr.PY for itr in ITSs])*0.01
    # Try with the APR instead RESULT worse with APR than with PY
    #PYs = np.array([itr.APR for itr in ITSs])*0.01

    # Parameter ranges you want to test out
    #c1 = np.linspace(1, 50, 50)
    c1 = np.array([20]) # insensitive to variation here
    c2 = np.array([0]) # c2 is best evaluated to 0
    #c2 = np.linspace(0.001, 0.2, 10)
    c3 = np.linspace(0.01, 1.4, 9)
    #c3 = np.array([0])
    c4 = np.linspace(0.3, 1.4, 9)
    #c4 = np.array([0])
    c5 = np.linspace(1, 40, 7)

    par_ranges = (c1, c2, c3, c4, c5)

    its_max = 21
    its_range = range(3, its_max)

    #randomize = 5 # here 0 = False (or randomize 0 times)
    randomize = 0 # here 0 = False (or randomize 0 times)

    # Fit with 50% of ITS and apply the parameters to the remaining 50%
    #retrofit = 5
    retrofit = 0

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=randomize, retrofit=retrofit)

    # extract the specific results
    results, rand_results, retrof_results = all_results

    print('')
    print('$$')
    for val in sorted(results[15].finals):
        print val

    # ladder plot
    plt.ion()

    ylim = -0.1
    p_line = True
    fig_lad, ax_lad = print_scrunch_ladder(results, rand_results,
                                           retrof_results, randomize,
                                           par_ranges, p_line, its_max, ylim,
                                           print_params=True)
    # Save the scruch laddder
    #for fig_dir in fig_dirs:
        #for formt in ['pdf', 'eps', 'png']:

            #name = 'ITS_rand_distribution_comparison.' + formt
            #odir = os.path.join(fig_dir, formt)

            #if not os.path.isdir(odir):
                #os.makedirs(odir)

            #fig.savefig(os.path.join(odir, name), transparent=True, format=formt)
    # parameter plot (best and average for each its)
    #parameter_relationship(results, optim, randomize, par_ranges)

    # scatter plot 
    #print_scrunch_scatter(results, rand_results, optim, randomize, par_ranges,
                  #initial_bubble, PYs)

    # Maybe plot the optimal and average parameter values for each ITS-len?

    # a plot/figure where you see the distribution of RNAP for 5 selected
    # sequences over time.
    #print_rnap_distribution(results, rand_results, initial_bubble, ITSs)


def print_rnap_distribution(results, par_ranges, initial_bubble, ITSs):
    """
    For some selected ITS sequences print the RNAP concentration

    Deprecated since you have removed all references to time_series
    """
    #plt.ion()
    final_result = results[max(results.keys())]

    # for each row, pick the results 
    fmts = ['r', 'g', 'b', 'c']

    # Compare with the PY percentages in this notation
    names, PYs = zip(*[(itr.name, itr.PY) for itr in ITSs])

    # tuple with name, py, and time_series
    united =  zip(names, PYs, final_result.time_series)

    # sort by PY, largest first ...
    united.sort(key=itemgetter(1), reverse=True)

    # XXX you just checked: time series[-1][-1] is like finals
    # indices of the ones you are picking out for plotting
    # vary the limits to your preference
    indices = [int(v) for v in np.linspace(0,42,3)]

    plotme = [united[i] for i in indices]

    fig, axes = plt.subplots(3, 3, sharey=True, sharex=True)

    for col_nr, p in enumerate(reversed(plotme)):
        # each column is 1 promoter; each row is t =0, t =1/2, or t =1

        name, py, t_series = p

        for row_nr, ts_data in enumerate(t_series):

            ax = axes[row_nr, col_nr]

            ax.plot(ts_data, linewidth=2, color=fmts[row_nr])

            if row_nr == 0:
                ax.set_title(name)

            if col_nr == 0:

                if row_nr == 0:
                    ax.set_ylabel('Early')

                if row_nr == 1:
                    ax.set_ylabel('Mid')

                if row_nr == 2:
                    ax.set_ylabel('Late')

            if row_nr == 2:
                ax.set_xlabel('Nucleotide position')

            ax.set_xticklabels([])
            ax.set_yticklabels([])

def family_of_models(ITSs, p_line, global_params):
    """
    Simulate a set of pre-defined models, then print figures for all of them.
    The models are the family of all possible variations of variables in the
    original model. In the first run, just get the optimal parameters for that
    submodel. In the next run, use those optimal prameters together with random
    DNA and cross validation of those parameters.

    Return all 6 models. 8 - full and 000.
    100
    001
    010

    110
    011
    101

    How are you going to enumerate them
    DNA-DNA RNA TN

    DNA-DNA+RNA RNA+TN  DNA-DNA+TN
    """

    # Compare with the PY percentages in this notation
    PYs = np.array([itr.PY for itr in ITSs])*0.01

    # Range of its values
    max_its = 21
    its_range = range(3, max_its)

    #modelz = get_models(global_params, stepsize=10)
    modelz = get_models(global_params, stepsize=4)

    # Skip the following: m4, m5, m9, m10, m11
    for mname in ['m4', 'm5', 'm6', 'm7', 'm8']:
        modelz.pop(mname)

    # First get the optimal parameters for each model
    model_opt_params = {}
    for (model_name, m) in modelz.items():

        par_ranges = m[:-1]  # extract info
        opt_params = get_optimal_params(PYs, its_range, ITSs, par_ranges)

        model_opt_params[model_name] = (opt_params, m)

    # then solve the equation again with randomization and retrofitting
    #XXX now not using randomization and retrofitting
    randomize = 0
    retrofit = 0

    debug()
    # XXX you get the OPTIMAL parameters for the whole journey! That's why you
    # don't get what you want. You want optimization for each step -- otherwise
    # the figures are not comparable.

    all_model_results = {}
    for (model_name, (opt_params, m)) in model_opt_params.items():

        par_ranges = opt_params

        all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                     randize=randomize, retrofit=retrofit)
        # extract the specific results
        results, rand_results, retrof_results = all_results

        all_model_results[model_name] = (m, all_results)

    # print the models
    ymin = -0.7
    #fig = print_model_family(all_model_results, p_line, max_its, ymin)

    # alternative approach: plot all models in one figure!
    fig = print_model_family2(all_model_results, p_line, max_its, ymin)

    return fig

def print_model_family2(resultz, p_line, max_its, ymin):
    """
    Layout

    If 1 is DD, 2 is RD, and 3 is TR

    100 - m7
    010 - m6
    001 - m8

    110 - m3
    101 - m2
    011 - m1
    """

    models = resultz.keys()
    colors = np.array(['g', 'b', 'r', 'c', 'k', 'm'])

    labels = {'m1': "Without $\Delta G_{RNA-DNA}$",
              'm2': "Without $\Delta G_{DNA-DNA}$",
              'm3': "Without $\Delta G_{3D}$",
              'm4': "With all variables"}

    fig, ax = plt.subplots()

    # predefine tixk labels
    xticklabels = [str(integer) for integer in range(3, max_its)]
    ymin = float(format(ymin, '.1f'))
    yticklabels = [format(i, '.1f') for i in np.arange(ymin, 1.1, 0.1)]

    for model_name, colr in zip(models, colors):

        minfo, all_results = resultz[model_name]
        (par_ranges, descr) = minfo[:-1], minfo[-1]  # extract info
        results, rand_results, retrof_results = all_results

        #hedr = descr.split(':')[1][1:]

        # get its_index and corr-coeff from sorted dict
        indx, corr, pvals = zip(*[(r[0], r[1].corr_max, r[1].pvals_min)
                           for r in sorted(results.items())])

        # check for nan in corr and pval (make 0 and 1)
        corr, pvals = remove_nan(corr, pvals)

        incrX = range(indx[0], indx[-1]+1)

        if model_name == 'm4':
            ax.plot(incrX, corr, label=labels[model_name], linewidth=2, color=colr,
                    linestyle='--')
        else:
            ax.plot(incrX, corr, label=labels[model_name], linewidth=2, color=colr)


        if p_line and model_name == 'm1':
            pv, co = zip(*sorted(zip(pvals, corr)))
            f = interpolate(pv[:16], co[:16], k=1) # the first 16 should contain 0.05
            line_val = f(0.05) # 0.05 correlation mark

            #ax.axhline(y=line_val, ls='--', color='r',
                        #label='p = 0.05 threshold', linewidth=2)

            # remove the label for the pvalue
            ax.axhline(y=line_val, ls='--', color='r', linewidth=2)

        # xticks
        ax.set_xticks(range(3,21))
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(3,21)

        ax.set_xlabel("RNA length (nt)", size=10)

        for l in ax.get_xticklabels():
            l.set_fontsize(6)

        # yticks
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)
        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

        ax.set_ylim(ymin, 1)
        ax.set_ylabel("Correlation coefficient, $r$", size=10)
        ax.set_yticks(np.arange(ymin, 1.1, 0.1))
        ax.set_yticklabels(yticklabels)

        for l in ax.get_yticklabels():
            l.set_fontsize(6)

        ax.legend(loc='lower right')

    return fig

def print_model_family(resultz, p_line, max_its, ymin):
    """
    Layout

    If 1 is DD, 2 is RD, and 3 is TR

    100 - m7
    010 - m6
    001 - m8

    110 - m3
    101 - m2
    011 - m1
    """

    # Based on model name make a 2x3 grid
    model_grid = np.array([['m7', 'm6', 'm8'], ['m3', 'm2', 'm1']])

    fig, axes = plt.subplots(2,3, sharex=True, sharey=True)

    # predefine tixk labels
    xticklabels = [str(integer) for integer in range(3, max_its)]
    ymin = float(format(ymin, '.1f'))
    yticklabels = [format(i, '.1f') for i in np.arange(ymin, 1.1, 0.1)]

    for (row_nr, col_nr) in itertools.product(range(2), range(3)):

        model_name = model_grid[row_nr, col_nr]
        minfo, all_results = resultz[model_name]
        (par_ranges, descr) = minfo[:-1], minfo[-1]  # extract info
        results, rand_results, retrof_results = all_results

        hedr = descr.split('const, ')[1]

        ax = axes[row_nr, col_nr]

        ax.set_title(hedr, size=5)

        # go over both the real and random results
        # This loop covers [0][0] and [0][1]
        # don't print random and cross anymore?
        #for (ddict, name, colr) in [(results, 'real', 'b'),
                                    #(rand_results, 'random', 'g'),
                                   #(retrof_results, 'cross-validated', 'r')]:

        for (ddict, name, colr) in [(results, 'real', 'b')]:

            # don't process 'random' or 'retrofit' if not evaluated
            if False in ddict.values():
                continue

            # get its_index and corr-coeff from sorted dict
            if name == 'real':
                indx, corr, pvals = zip(*[(r[0], r[1].corr_max, r[1].pvals_min)
                                   for r in sorted(ddict.items())])

            elif name == 'random':
                indx, corr = zip(*[(r[0], r[1].corr_mean)
                                   for r in sorted(ddict.items())])

                stds = [r[1].corr_std for r in sorted(ddict.items())]

            elif name == 'cross-validated':
                indx, corr = zip(*[(r[0], r[1].corr_mean)
                                   for r in sorted(ddict.items())])

                stds = [r[1].corr_std for r in sorted(ddict.items())]

            # make x-axis
            incrX = range(indx[0], indx[-1]+1)

            # check for nan in corr and pval (make 0 and 1)
            corr, pvals = remove_nan(corr, pvals)

            # if random, plot with errorbars
            if name == 'real':
                ax.plot(incrX, corr, label=name, linewidth=2, color=colr)

                if p_line:

                    pv, co = zip(*sorted(zip(pvals, corr)))
                    f = interpolate(pv[:16], co[:16], k=1) # the first 16 should contain 0.05
                    line_val = f(0.05) # 0.05 correlation mark

                    ax.axhline(y=line_val, ls='--', color='r',
                                label='p = 0.05 threshold', linewidth=2)

                    #debug()

            elif name == 'random':
                ax.errorbar(incrX, corr, yerr=stds, label=name, linewidth=2,
                                 color=colr)

            elif name == 'cross-validated':
                ax.errorbar(incrX, corr, yerr=stds, label=name, linewidth=2,
                                 color=colr)
            # xticks
            ax.set_xticks(range(3,21))
            ax.set_xticklabels(xticklabels)
            ax.set_xlim(3,21)

            if row_nr == 1:
                ax.set_xlabel("RNA length (nt)", size=8)

            for l in ax.get_xticklabels():
                l.set_fontsize(5)

            # yticks
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)
            ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)

            ax.set_ylim(ymin, 1)
            if col_nr == 0:
                ax.set_ylabel("Correlation coefficient, $r$", size=8)
                ax.set_yticks(np.arange(ymin, 1.1, 0.1))
                ax.set_yticklabels(yticklabels)

            for l in ax.get_yticklabels():
                l.set_fontsize(5)

    return fig

def remove_nan(corr, pval):
    """
    Replace nans in corr and pval with 0 and 1. If not dont replace anything.
    """

    if True in np.isnan(corr):
        newcorr = []
        for inx, truthvalue in enumerate(np.isnan(corr)):
            if truthvalue == True:
                newcorr.append(0)
            else:
                newcorr.append(corr[inx])

        corr = newcorr

    if True in np.isnan(pval):
        newpval = []
        for inx, truthvalue in enumerate(np.isnan(pval)):
            if truthvalue == True:
                newpval.append(1)
            else:
                newpval.append(pval[inx])

        pval = newpval

    return corr, pval

def get_models(global_params, stepsize=5):
    """
    Define some models that test various hypotheseses ...

    The models are defined by their parameters
    """
    s = stepsize

    gp = global_params

    # Model 1 --
    # zero RNA-DNA, DNA-DNA and Keq variable
    m1 = (np.array([gp['K']]),
          np.array([0]),
          np.linspace(gp['dd_min'], gp['dd_max'], s),
          np.linspace(gp['eq_min'], gp['eq_max'], s),
          "M1: bubble + 3'dinucleotide")

    # Model 2 --
    # RNA-DNA and Translocation variable
    m2 = (np.array([gp['K']]),
          np.linspace(gp['rd_min'], gp['rd_max'], s),
          np.array([0]),
          np.linspace(gp['eq_min'], gp['eq_max'], s),
         "M2: hybrid + 3'dinucleotide")

    # Model 3 --
    # RNA-DNA and DNA-DNA variable
    m3 = (np.array([gp['K']]),
          np.linspace(gp['rd_min'], gp['rd_max'], s),
          np.linspace(gp['dd_min'], gp['dd_max'], s),
          np.array([0]),
         'M3: hybrid + bubble')

    # Model 4 --
    # RNA-DNA, DNA-DNA, and Translocation variable
    m4 = (np.array([gp['K']]),
          np.linspace(gp['rd_min'], gp['rd_max'], s),
          np.linspace(gp['dd_min'], gp['dd_max'], s),
          np.linspace(gp['eq_min'], gp['eq_max'], s),
         "M4: hybrid + bubble + 3'dinucleotide")

    # Model 5 --
    # RNA-DNA, DNA-DNA, and Translocation constant
    m5 = (np.linspace(2, gp['K'], s),
            np.array([0]),
            np.array([gp['dd_best']]),
            np.array([gp['eq_best']]),
         "M5: bubble and 3'dinucleotide constant")

    # Model 6 --
    # RNA-DNA variable
    m6 = (np.array([gp['K']]),
          np.linspace(gp['rd_min'], gp['rd_max'], s),
          np.array([0]),
          np.array([0]),
        'M6: hybrid')

    # Model 7 --
    #  DNA-DNA variable
    m7 = (np.array([gp['K']]),
          np.array([0]),
          np.linspace(gp['dd_min'], gp['dd_max'], s),
          np.array([0]),
         'M7: bubble')

    # Model 8 --
    # Translocation variable
    m8 = (np.array([gp['K']]),
          np.array([0]),
          np.array([0]),
          np.linspace(gp['eq_min'], gp['eq_max'], s),
         "M8: 3' dinucleotide")

    models = {'m1': m1,
              'm2': m2,
              'm3': m3,
              'm4': m4,
              'm5': m5,
              'm6': m6,
              'm7': m7,
              'm8': m8}

    return models

class Model(object):
    """
    A model ... it's characterized by the constants
    """

    def __init__(self, name, c1, c2, c3, c4, description):

        self.name = name

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

        self.description = description

def print_scrunch_scatter(results, rand_results, randomize, par_ranges,
                          PYs, PY_std, pos=None, laddax=False):
    """
    Print scatter plots at peak correlation (14 at the moment)

    RESULT this has proven to be a problem. While the correlation itself is good
    for any c1, only a specific c1 value will give similar value ranges for PYs
    and finals for a given its_len. This is to be expected, and is a trade-off
    between c1 and t -- the time of integration.

    Isn't it then marvelous that you got such a good match between PY and the
    'abortive propensity' initially? A little bit too marvelous for my flavor.
    """
    #plt.ion()

    PYs = PYs*100

    #rows = [5, 10, 15, 20]

    # use the position form the pos argument; if not given, use the position
    # wigh highest correlation
    if pos:
        maxnuc = pos
    else:
        maxnuc = list(sorted([(r.corr_max, v) for (v, r) in
                              results.items()]))[-1][-1]

    fig, ax = plt.subplots()
    if laddax is False:
        pass
    else:
        ax = laddax[1]
        ax.cla()

    # if you haven't calculated anything, return the empty figure
    if maxnuc-1 in results:
        finals = results[maxnuc-1].finals # you were comparing to PY/100
    else:
        return fig

    print("Spearman for maxnuc: {0} and PY".format(maxnuc))
    print(spearmanr(finals, PYs))
    print(pearsonr(finals, PYs))
    ax.scatter(finals, PYs, color= 'k')
    ax.errorbar(finals, PYs, yerr=PY_std, fmt=None, color= 'k')

    ax.set_ylabel("PY", size=21)
    ax.set_xlabel("SE$_{20}$", size=21)

    # awkward way of setting the tick sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
        #l.set_fontsize(6)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)
        #l.set_fontsize(6)

    xmin, xmax = min(finals), max(finals)
    xscale = (xmax-xmin)*0.1
    ax.set_xlim(xmin-xscale, xmax+xscale)

    ymin, ymax = min(PYs), max(PYs)
    yscale_low = (ymax-ymin)*0.1
    yscale_high = (ymax-ymin)*0.4
    ax.set_ylim(ymin-yscale_low, ymax+yscale_high)

    return fig

def parameter_relationship(results, optim, randomize, par_ranges):
    """
    Print the relationship between the optimal and the average parameters for
    the different ITS lengths. XXX maybe this one is better to have next to the
    ladder plot.
    """
    fig, ax = plt.subplots()

    # sorted -> in ITS increasing order
    paramz_best = [r[1].params_best for r in sorted(results.items())]
    paramz_mean = [r[1].params_mean for r in sorted(results.items())]

    #best_c3, best_c4 = zip(*[(pb['c3'], pb['c4']) for pb in paramz_best])
    #mean_c3, mean_c4 = zip(*[(pb['c3'], pb['c4']) for pb in paramz_mean])
    best_c3, best_c4 = zip(*[(pb['c2'], pb['c4']) for pb in paramz_best])
    mean_c3, mean_c4 = zip(*[(pb['c2'], pb['c4']) for pb in paramz_mean])

    # You want the relationship between them

    best_rel = np.array(best_c4)/np.array(best_c3)

    mean_rel = np.array(mean_c4)/np.array(mean_c3)

    #x_range = range(3, 21) # bug -- you don't go to 20
    x_range = range(3, 20)

    #ax.scatter(best_rel, color='b', label='Best')
    ax.plot(x_range, best_rel, color='b', label='Best')

    ax.plot(x_range, mean_rel, color='g', label='Average')
    #ax.scatter(mean_rel, color='g', label='Average')

    ax.set_xlabel('Nucleotide past +3')
    ax.set_ylabel('c4/c3 ratio')

    ax.legend(loc='lower right')

    # xticks
    xticklabels = [str(integer) for integer in range(3,21)]
    ax.set_xticks(x_range)
    ax.set_xticklabels(xticklabels)
    #ax.set_xlim(3,21)
    ax.set_xlabel("RNA length (nt)", size=20)

def print_scrunch_ladder_compare(results, rand_results, retrof_results,
                                 randomize, par_ranges, p_line, its_max, ymin,
                                 ymax, testing, description=False,
                                 print_params=True, in_axes=False, ax_nr=0):
    """
    Alternative print-scrunch.
    [0][0] is the pearson for the real and the control
    [0][1] is the parameters for the pearson fit

    You get in both the two and the three-parameter models

    In A plot the two and three models on top of each other. In B plot the
    parameter estimation of the three model.
    """
    fig, axes = plt.subplots(1,2)

    colors_params = ['r', 'c', 'm', 'k', 'g']

    all_params = ('c1', 'c2', 'c3', 'c4', 'c5')
    labels = ['K', 'RNA-DNA', 'DNA-DNA', "3' dinucleotide"]

    # only plot parameters that are variable 
    plot_params = [p for p, r in zip(all_params, par_ranges) if len(r) > 1]

    # assign colors to parameters
    par2col = dict(zip(all_params, colors_params))
    par2label = dict(zip(all_params, labels))

    # go over both the real and random results
    # This loop covers [0][0] and [0][1]
    col_corr = ['b', 'g']
    for (name, result_dict) in results.items():

        # get its_index and corr-coeff from sorted dict
        indx, corr, pvals = zip(*[(r[0], r[1].corr_max, r[1].pvals_min)
                           for r in sorted(result_dict.items())])

        # make x-axis
        incrX = range(indx[0], indx[-1]+1)

        # check for nan in corr (make it 0)
        corr, pvals = remove_nan(corr, pvals)

        axes[0].plot(incrX, corr, label=name, linewidth=3, color=col_corr.pop())

        # interpolate pvalues (x, must increase) with correlation (y) and
        # obtain the correlation for p = 0.05 to plot as a black
        if p_line and name == 'three_param_model':
            # hack to get pvals and corr coeffs sorted
            pv, co = zip(*sorted(zip(pvals, corr)))
            f = interpolate(pv, co, k=1)
            axes[0].axhline(y=f(0.05), ls='--', color='r',
                        label='p = 0.05 threshold', linewidth=3)

        # get its_index parameter values (they are index-sorted)
        if name == 'three_param_model':
            paramz_best = [r[1].params_best for r in sorted(result_dict.items())]

            # each parameter should be plotted with its best (solid) and mean
            # (striped) values (mean should have std)
            for parameter in plot_params:

                # print the best parameters
                best_par_vals = [d[parameter] for d in paramz_best]
                axes[1].plot(incrX, best_par_vals, label=par2label[parameter],
                                   linewidth=3, color=par2col[parameter])
                # mean
                #mean_par_vals = [d[parameter] for d in paramz_mean]


    xticklabels = [str(integer) for integer in range(3, its_max)]
    #Make sure ymin has only one value behind the comma
    ymin = float(format(ymin, '.1f'))
    yticklabels = [format(i ,'.1f') for i in np.arange(ymin, 1.1, 0.1)]
    #yticklabels = [str(integer) for integer in np.arange(0, 1, 0.1)]
    #yticklabels_1 = [str(integer) for integer in np.arange(-0.05, 0.5, 0.05)]

    for ax in axes.flatten():
        # legend
        ax.legend(loc='lower left')

        # xticks
        ax.set_xticks(range(3,its_max))
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(3,its_max)
        ax.set_xlabel("RNA length (nt)", size=23)

        # awkward way of setting the tick font sizes
        for l in ax.get_xticklabels():
            l.set_fontsize(15)
        for l in ax.get_yticklabels():
            l.set_fontsize(15)

    axes[0].set_ylabel("Correlation coefficient, $r$", size=23)

    axes[0].set_yticks(np.arange(ymin, 1.1, 0.1))
    #axes[0].set_yticks(np.arange(0, 1, 0.1))
    axes[0].set_yticklabels(yticklabels)
    axes[0].set_ylim(ymin, ymax)
    # you need a grid to see your awsome 0.8 + correlation coefficient
    axes[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

    axes[1].set_ylabel("Model parameter values", size=23)
    #axes[1].set_yticks(np.arange(-0.05, 0.5, 0.05))
    #axes[1].set_yticklabels(yticklabels_1)
    #axes[1].set_ylim(-0.05, 0.5)

    fig.set_figwidth(20)
    fig.set_figheight(10)

    # This last part was the title which you don't need in production
    if testing:

        for_join = []

        for let, par in zip(('a', 'b', 'c', 'd'), par_ranges):
            if len(par) == 1:
                for_join.append(let + ':{0:.2f}'.format(par[0]))
            else:
                mi, ma = (format(min(par), '.2f'), format(max(par), '.2f'))
                for_join.append(let + ':({0}--{1})'.format(mi, ma))

        if description:
            descr = ', '.join(for_join) + '\n' + description
        else:
            descr = ', '.join(for_join)

        hedr = 'Nr random samples: {0}\n\n{1}\n'.format(randomize, descr)
        fig.suptitle(hedr)


    if len(axes > 1):
        for i, label in enumerate(('A', 'B')):
            ax = axes[i]
            ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=26,
                    fontweight='bold', va='top')

    return fig, axes

def print_scrunch_ladder(results, rand_results, retrof_results, randomize,
                         par_ranges, p_line, its_max, its_range, ymin, ymax,
                         testing, description=False, print_params=True,
                         in_axes=False, ax_nr=0, inset=False):
    """
    Alternative print-scrunch.
    [0][0] is the pearson for the real and the control
    [0][1] is the parameters for the pearson fit

    insert plot makes inset of growth in correlation from one step to another
    """
    if print_params:
        fig, axes = plt.subplots(1,2)
    else:
        fig, ax = plt.subplots()
        axes = np.array([ax])

    if in_axes:
        axes = in_axes

    # assign colors to parameters
    colors_params = ['r', 'c', 'm', 'k', 'g']
    all_params = ('c1', 'c2', 'c3', 'c4', 'c5')
    labels = ['K', 'RNA-DNA', 'DNA-DNA', "3' dinucleotide"]
    # only plot parameters that are variable 

    plot_params = [p for p, r in zip(all_params, par_ranges) if len(r) > 1]

    # assign colors to parameters
    par2col = dict(zip(all_params, colors_params))
    par2label = dict(zip(all_params, labels))

    # go over both the real and random results
    # This loop covers [0][0] and [0][1]
    for (ddict, name, colr) in [(results, 'real', 'b'),
                                (rand_results, 'random', 'g'),
                               (retrof_results, 'cross-validated', 'k')]:

        # don't process 'random' or 'retrofit' if not evaluated
        if False in ddict.values():
            continue

        # get its_index and corr-coeff from sorted dict
        if name == 'real':
            indx, corr, pvals = zip(*[(r[0], r[1].corr_max, r[1].pvals_min)
                               for r in sorted(ddict.items())])


        elif name == 'random':
            indx, corr = zip(*[(r[0], r[1].corr_mean)
                               for r in sorted(ddict.items())])

            stds = [r[1].corr_std for r in sorted(ddict.items())]

        elif name == 'cross-validated':
            indx, corr = zip(*[(r[0], r[1].corr_mean)
                               for r in sorted(ddict.items())])

            stds = [r[1].corr_std for r in sorted(ddict.items())]

        # points on x-axis for plotting -- same as its_lenghts
        incrX = its_range

        if name == 'real':
            # check for nan in corr (make it 0)
            corr, pvals = remove_nan(corr, pvals)

            lab = 'correlation(PY, SE$_n$)'
            axes[ax_nr].plot(incrX, corr, label=lab, linewidth=4, color=colr)

            # interpolate pvalues (x, must increase) with correlation (y) and
            # obtain the correlation for p = 0.05 to plot as a black

            if p_line:
                # sort pvals and corr and interpolate
                pv, co = zip(*sorted(zip(pvals, corr)))
                f = interpolate(pv, co, k=1)
                axes[ax_nr].axhline(y=f(0.05), ls='--', color='r',
                                    label='p = 0.05 threshold', linewidth=3)

            # add inset plot with addition of corr at each step
            # XXX 
            #if inset:
                #pass
                #debug()

        # if random, plot with errorbars
        elif name == 'random':
            lab = 'using random sequences'
            axes[ax_nr].errorbar(incrX, corr, yerr=stds, label=lab, linewidth=3,
                             color=colr)

        elif name == 'cross-validated':
            lab = 'cross-validation'
            axes[ax_nr].errorbar(incrX, corr, yerr=stds, label=lab, linewidth=3,
                             color=colr)

        # skip if you're not printing parametes
        if print_params and name == 'real':

            paramz_best = [r[1].params_best for r in sorted(ddict.items())]
            #paramz_mean = [r[1].params_mean for r in sorted(ddict.items())]
            #paramz_std = [r[1].params_std for r in sorted(ddict.items())]

            # each parameter should be plotted with its best (solid) and mean
            # (striped) values (mean should have std)
            for parameter in plot_params:

                # print the best parameters
                best_par_vals = [d[parameter] for d in paramz_best]
                axes[ax_nr+1].plot(incrX, best_par_vals,
                                   label=par2label[parameter], linewidth=3,
                                   color=par2col[parameter])
                # mean
                #mean_par_vals = [d[parameter] for d in paramz_mean]

                # std XXX maybe it's best not to use the error bars
                # print the mean and std of the top 20 parameters
                #std_par_vals = [d[parameter] for d in paramz_std]
                #axes[ax_nr+1].errorbar(incrX, mean_par_vals, yerr=std_par_vals,
                                       #color=par2col[parameter], linestyle='--')

    xticklabels = [str(integer) for integer in range(3, its_max)]

    # only show even xticks
    xticklabels_skip = odd_even_spacer(xticklabels, oddeven='odd')

    #Make sure ymin has only one value behind the comma
    ymin = float(format(ymin, '.1f'))
    yticklabels = [format(i ,'.1f') for i in np.arange(ymin, 1.1, 0.1)]
    yticklabels_skip = odd_even_spacer(yticklabels)

    for ax in axes.flatten():
        # legend
        ax.legend(loc='lower right')

        # xticks
        ax.set_xticks(range(3,its_max))
        ax.set_xticklabels(xticklabels_skip)
        ax.set_xlim(3,its_max)
        ax.set_xlabel("RNA length (nt)", size=21)

        # awkward way of setting the tick font sizes
        for l in ax.get_xticklabels():
            l.set_fontsize(18)
        for l in ax.get_yticklabels():
            l.set_fontsize(18)
        #axes[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  #alpha=0.5)
        #axes[0].xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  #alpha=0.5)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
             alpha=0.5)
        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.5)

    axes[0].set_ylabel("Correlation coefficient, $r$", size=21)

    axes[0].set_yticks(np.arange(ymin, 1.1, 0.1))
    #axes[0].set_yticks(np.arange(0, 1, 0.1))
    axes[0].set_yticklabels(yticklabels_skip)
    axes[0].set_ylim(ymin, 1.001)
    # you need a grid to see your awsome 0.8 + correlation coefficient

    if print_params:
        axes[1].set_ylabel("Parameter value", size=23)
        #axes[1].set_yticks(np.arange(-0.05, 0.5, 0.05))
        #axes[1].set_yticklabels(yticklabels_1)
        #axes[1].set_ylim(-0.05, 0.5)

        fig.set_figwidth(20)
        fig.set_figheight(10)
    else:
        fig.set_figwidth(10)
        fig.set_figheight(10)

    # This last part was the title which you don't need in production
    # (Remove and False if you ever want to print something like this again)
    if testing and False:

        for_join = []

        for let, par in zip(('a', 'b', 'c', 'd'), par_ranges):
            if len(par) == 1:
                for_join.append(let + ':{0:.2f}'.format(par[0]))
            else:
                mi, ma = (format(min(par), '.2f'), format(max(par), '.2f'))
                for_join.append(let + ':({0}--{1})'.format(mi, ma))

        if description:
            descr = ', '.join(for_join) + '\n' + description
        else:
            descr = ', '.join(for_join)

        hedr = 'Nr random samples: {0}\n\n{1}\n'.format(randomize, descr)
        fig.suptitle(hedr)


    return fig, axes

def odd_even_spacer(vals, oddeven='even'):
    out = []
    if oddeven == 'odd':
        for val_nr, val in enumerate(vals):
            if val_nr % 2:
                out.append(val)
            else:
                out.append(' ')
    else:
        for val_nr, val in enumerate(vals):
            if not val_nr % 2:
                out.append(val)
            else:
                out.append(' ')

    return out


def scrunch_runner(PYs, its_range, ITSs, ranges, randize=0, retrofit=0,
                   normal=True, non_rnap=True, predictive=False):
    """
    Wrapper around grid_scrunch and opt_scrunch.

    randize and retrofit will randomize DNA and do cross-validation of parameter
    values. normal is on by default; it calculates the normal correlation you're
    after (CBP or [RNAP]). non_rnap will not calculate the [RNAP] concentration
    but will instead calculate the cumulative sum of Keq at the its_ranges.

    use non_rnap to NOTE use the differential equation model, but instead only
    use the sum of the equilibrium constants Keq to correlate with PY.

    use predictive if not sending in py values
    """

    # make 'int' into a 'list' containing just that int
    if type(its_range) is int:
        its_range = range(its_range, its_range +1)

    results = {}
    results_random = {}
    results_retrof = {}

    for its_len in its_range:

        state_nr = its_len - 1
        # The second nt is then state 0
        # initial vales for the states
        y0 = [1] + [0 for i in range(state_nr-1)]

        # get the 'normal' results
        if normal:
            normal_obj = grid_scruncher(PYs, its_len, ITSs, ranges, y0,
                                        state_nr, predictive=predictive,
                                        non_rnap=non_rnap)
        else:
            normal_obj = False

        # Randomize
        if randize:
            # get the results for randomized ITS versions
            random_obj = grid_scruncher(PYs, its_len, ITSs, ranges, y0,
                                        state_nr, predictive=predictive,
                                        non_rnap=non_rnap, randomize=randize)
        else:
            random_obj = False

        # Retrofit
        if retrofit:
            retrof_obj = grid_scruncher(PYs, its_len, ITSs, ranges, y0,
                                        state_nr, predictive=predictive,
                                        non_rnap=non_rnap, retrof=retrofit)
        else:
            retrof_obj = False

        results[its_len] = normal_obj
        results_retrof[its_len] = retrof_obj
        results_random[its_len] = random_obj


    return results, results_random, results_retrof


def get_its_variables(ITSs, randomize=False, retrofit=False, PY=False):
    """
    Get ITS objects and return a dict of their energy variables: RNA-DNA,
    DNA-DNA, Keq
    """
    new_ITSs = []

    for its in ITSs:

        # if randomize, create a new object ITS object from a random sequence
        if randomize:
            rand_seq = ITSgenerator()
            its = ITS(rand_seq)

        ITSs_dict = {'rna_dna_di': np.array(its.rna_dna_di),
                     'dna_dna_di': np.array(its.dna_dna_di),
                     'keq_di': np.array(its.keq_delta_di)}

        new_ITSs.append(ITSs_dict)

    # if retrofit, divide the list in 2 with random indices. ALso split the PYs
    # in the same way cause you need the its-> PY relationship
    if retrofit:
        its_nr = len(ITSs)
        pickme = np.random.randint(its_nr, size=np.ceil(its_nr/2.0))

        ITS_fitting = [new_ITSs[i] for i in pickme]
        ITS_compare = [new_ITSs[i] for i in range(43) if i not in pickme]

        PYs_fitting = [PY[i] for i in pickme]
        PYs_compare = [PY[i] for i in range(43) if i not in pickme]

        return ITS_fitting, ITS_compare, PYs_fitting, PYs_compare

    else:

        return new_ITSs

def grid_scruncher(PYs, its_len, ITSs, ranges, y0, state_nr, randomize=0,
                   retrof=0, non_rnap=True, predictive=False):
    """
    Separate scruch calls into randomize, retrofit, or neither.
    """

    # RETROFIT
    if retrof:
        retro_results = []
        for repeat in range(retrof):

            # Choose 50% of the ITS randomly; both energies and PYs
            retrovar = get_its_variables(ITSs, retrofit=True, PY=PYs)
            #extract the stuff
            ITS_fitting, ITS_compare, PYs_fitting, PYs_compare = retrovar

            arguments_fit = (y0, its_len, state_nr, ITS_fitting, PYs_fitting,
                             non_rnap, predictive)

            fit_result = grid_scrunch(arguments_fit, ranges)

            # Skip if you don't get a result (should B OK with large ranges)
            if not fit_result:
                continue

            par_order = ('c1', 'c2', 'c3', 'c4')

            fit_ranges = [np.array([fit_result.params_best[p]])
                          for p in par_order]

            # rerun with new arguments
            arguments_compare = (y0, its_len, state_nr, ITS_compare,
                                 PYs_compare, non_rnap, predictive)

            # run the rest of the ITS with the optimal parameters
            control_result = grid_scrunch(arguments_compare, fit_ranges)

            # Skip if you don't get a result (should B OK with large ranges)
            if not control_result:
                continue

            else:
                retro_results.append(control_result)

        # average the results (keeping mean and std) and return 
        return average_rand_result(retro_results)

    # RANDOMIZE
    elif randomize:
        rand_results = []
        for repeat in range(randomize):

            # normalizing by testing random sequences. Can we fit the parameters
            # from random sequences to these ITS values?

            ITS_variables = get_its_variables(ITSs, randomize=True)

            arguments = (y0, its_len, state_nr, ITS_variables, PYs, non_rnap,
                         predictive)

            rand_result = grid_scrunch(arguments, ranges)

            # skip if you don't get anything from this randomizer
            if not rand_result:
                continue

            else:
                rand_results.append(rand_result)

        return average_rand_result(rand_results)

    # NO RETROFIT AND NO RANDOMIZE
    else:
        #go from ITS object to dict to appease the impotent pickle
        # maybe you didn't have to do that. that really sucks. now.
        ITS_variables = get_its_variables(ITSs)

        arguments = (y0, its_len, state_nr, ITS_variables, PYs, non_rnap,
                     predictive)

        return grid_scrunch(arguments, ranges)

def average_rand_result(rand_results):
    """
    Keep only the top scores for each of the results in the sample. Those top
    scores will then be used to make a new object with the average and std of
    those scores. That object is then returned. This makes these objects
    different from the 'normal' result objects, where the std and mean are
    from sub-optimal results.
    """

    new_corr, new_pvals, new_params = [], [], {'c1':[], 'c2':[],
                                               'c3':[], 'c4':[]}

    # Add the values from the X random versions

    # You don't want all 20 saved outcomes from each random version, you just
    # want the best ones
    for res_obj in rand_results:
        # check if this is a NaN object; skip it
        if res_obj.corr is np.nan:
            continue

        new_corr.append(res_obj.corr[0])
        new_pvals.append(res_obj.pvals[0])

        for k, v in res_obj.params.items():
            new_params[k].append(v[0])

    # Check if new_corr is empty (means all results were NaN)
    if new_corr == []:
        return Result()
    else:
        # make a new result object
        new_res = Result(new_corr, new_pvals, new_params, res_obj.finals)

        return new_res

def grid_scrunch(arguments, ranges):
    """
    Wrapper around the multi-core solver. Return a Result object with
    correlations and pvalues.
    """

    # all possible compbinations
    params = [p for p in itertools.product(*ranges)]

    window = int(np.floor(len(params)/4.0))
    divide = [params[i*window:(i+1)*window] for i in range(3)]
    last = params[3*window:] # make sure you get all in the last one
    divide.append(last)

    t1 = time.time()

    #make a pool of workers for multicore action
    # this is only efficient if you have a range longer than 10 or so
    rmax = sum([len(r) for r in ranges])
    #rmax = 5 #XXX uncomment for multiprocessing debugging.
    if rmax > 6:
        my_pool = multiprocessing.Pool()
        results = [my_pool.apply_async(_multi_func, (p, arguments)) for p in divide]
        my_pool.close()
        my_pool.join()
        # flatten the output
        all_results = sum([r.get() for r in results], [])
    else:
        #the non-parallell version for debugging and no multi-range calculations
        all_results = sum([_multi_func(*(p, arguments)) for p in divide], [])

    # All_results is a list of tuples on the following form:
    # ((finals, par, rp, pp))
    # Sort them on the pearson/spearman correlation pvalue, 'pp'

    # Filter and pick the 20 with smallest pval
    all_sorted = sorted(all_results, key=itemgetter(3))

    # pick top 20
    top_hits = all_sorted[:20]

    # if top_hits is empty, return NaNs (created by default)
    # this might occur of you send Py = zeros because you just want to calculate
    # the final scores
    if top_hits == []:
        return Result() # all values are NaN

    # now make separate corr, pvals, params, and finals arrays from these
    finals = np.array(top_hits[0][0]) # only get finals for the top top
    pars = [c[1] for c in top_hits]
    corr = np.array([c[2] for c in top_hits])
    pvals = np.array([c[3] for c in top_hits])

    # params must be made into a dict
    # params[c1,c2,c3,c4] = [array]
    params = {}
    for par_nr in range(1, len(ranges)+1):
        params['c{0}'.format(par_nr)] = [p[par_nr-1] for p in pars]

    # make a Result object
    result_obj = Result(corr, pvals, params, finals)

    print time.time() - t1

    return result_obj

def _multi_func(paras, arguments):
    """
    Evaluate the model for all parameter combinations. Return the final values,
    the parameters, and the correlation coefficients.
    """
    all_hits = []
    # v. dangerous. should pass arguments in a dictionary to avoid this.
    PYs = arguments[-3]
    predictive = arguments[-1]

    for par in paras:
        finals = cost_function_scruncher(par, *arguments)

        # correlation
        #rp, pp = pearsonr(PYs, finals)
        rp, pp = spearmanr(PYs, finals)

        # ignore those where the correlation is a nan
        # they get a pvalue of 0 which mezzez things up
        # ooopz, w
        if not predictive and np.isnan(rp):
            continue

        all_hits.append((finals, par, rp, pp))

    return all_hits

def get_variables(seq):
    """
    Return rna_dna, dna_dna and keq energies from sequence
    """

    __indiv = list(seq)
    __dinucs = [__indiv[c] + __indiv[c+1] for c in range(len(seq)-1)]

    # Make di-nucleotide vectors for all the energy parameters
    rna_dna = [Ec.NNRD[di] for di in __dinucs]
    dna_dna = [Ec.NNDD[di] for di in __dinucs]
    keq_delta = [Ec.delta_keq[di] for di in __dinucs]

    return rna_dna, dna_dna, keq_delta

def mini_scrunch(seqs, params, state_nr, y0, multi_range):
    """
    A new wrapper around scipy.integrate.odeint. The previous one was too
    focused on optimization and you don't need that here.

    If supplied with start seqs, you should Start seqs
    """

    RT = 1.9858775*(37 + 273.15)/1000   # divide by 1000 to get kcalories

    py_like = []

    for seq in seqs:

        # energy variables from sequence
        rna_dna, dna_dna, keq_delta = get_variables(seq)

        (a, b, c, d) = params
        its_len = state_nr + 1
        k1 = calculate_k1_difference(RT, its_len, keq_delta, dna_dna, rna_dna,
                                     a, b, c, d)
        A = equlib_matrix(k1, state_nr)

        # time in which to integrate over
        tim = 1

        # if there are 'nan' in A-matrix, return nan
        if True in np.isnan(A):
            soln = [np.nan]
        else:
            # using the x(t) = e^{A*t}*y(0) solution 
            soln = dot(scipy.linalg.expm(A*tim), y0)

        # if not multi-range, only report the its_len concentration
        if not multi_range:
            py_like.append((soln[-1], seq))

        # the first state is for pos +2. The 12th state 
        # [0] -> +2. so + 12 is [10]
        # else, report the concentration at each of the pos in multi conc
        else:
            multi_conc = dict((pos, soln[pos-2]) for pos in multi_range)
            py_like.append((soln[-1], seq, multi_conc))

    return py_like

def calculate_k1_difference(RT, its_len, keq, dna_dna, rna_dna, a,
                            b, c, d, hybrid_weight=False):
    """
    Recall that the energies are in dinucleotide form. Thus ATG -> [0.4, 0.2]

    Tip for the future; start at +1 also for nucleotides, so all arrays have the
    same length. Just have the first values as 0. Makes it much easier to
    understand the code later. The 0s are placeholders.

    !!There is no RNA-DNA energy change before +10. FFS.

    Q: what do you get in? the keq, dna_dna etc?

    k1[0] should be for d(x_3)/dt

    -> keq = (1,2)
    -> RD = 0
    -> DD = (1,3) - (1,2)

    k1[1] should be for d(x_4)/dt

    -> keq = (2,3)
    -> RD = 0
    -> DD = (1,4) - (1,3)

    ...

    k1[8] is the last where RD is zero

    k1[7] is then d(x_10)/dt -- the first where RD gets subtracted

    -> keq = (8,9)
    -> RD = (2,9) - (1,9)
    -> DD = (1,10) - (1,9)

    k1[8] is then d(x_11)/dt

    -> keq = (9,10)
    -> RD = (3,10) - (2,10)
    -> DD = (1,11) - (1,10)
    ...

    until k1[17] which is d(x_20)/dt -- or do you want to include x_21?

    """

    # initialize empty rates. -2 because you start going from +2 to +3.
    # ATGCT -> [0.3, 0.5, 0.2, 0.8]
    # -> (1,2) = 0.3
    # -> (1,3) = 0.3 + 0.5
    # -> (2,3) = (0.5)
    k1 = np.zeros(its_len-2)

    # for writing to file the Delta G values
    keq_array = []
    dnadna_array = []
    rnadna_array = []

    for i in range(0, its_len-2):

        KEQ = keq[i]
        keq_array.append(KEQ)

        #DNA_DNA = sum(dna_dna[:i+1+1]) - sum(dna_dna[:i+1]) #+1 for python
        #DNA_DNA = sum(dna_dna[:i+1]) # proving a point ...
        DNA_DNA = dna_dna[i+1] # the difference is the same as the last element
        dnadna_array.append(DNA_DNA)

        # index 8 is the same as x_11. k[8] is for x_11
        if i < 8:
            #RNA_DNA = sum(rna_dna[:i])
            RNA_DNA = 0 #
        else:
            # the 7 and the 8 are because of starting at +3, dinucleotides, and
            # python indexing 
            RNA_DNA = sum(rna_dna[i-7:i+1]) - sum(rna_dna[i-8:i+1])
        rnadna_array.append(RNA_DNA)

        expo = (b*RNA_DNA +c*DNA_DNA +d*KEQ)/RT

        rate = a*np.exp(-expo)

        # if hybrid weight .. RESULT destroys correlation
        if hybrid_weight:
            if i < 10:
                hybrid = sum(rna_dna[:i])
            elif i > 10:
                hybrid = sum(rna_dna[i-10:i])

            k1[i] = rate + hybrid*0.5

        else:
            k1[i] = rate


    return k1, (rnadna_array, dnadna_array, keq_array)

def calculate_keq(RT, beg, end, d3, dna_dna, rna_dna, a, b, c, d,
                  scrunching=True):
    """
    Recall that the energies are in dinucleotide form. Thus ATG -> [0.4, 0.2]

    !!There is no RNA-DNA energy change before +10.

    This function returns a list of keq_i values, where i in [beg, end]. The
    dna_dna, rna_dna, and d3 values range from 0 to 20.

    1234567891011
    ATGCTGACGT A

    ens = [z0(AT), z1(TG), z2(GC), z3(CT), z4(TG), z5(GA), z6(AC), z7(CG),
    z8(GT), z9(TA)]

    beg/end input is in 1-notation. Thus, i = 3 is ens[2]

    """
    keqs = np.zeros(end-beg) # array the size of the nr of keq asked for

    # go from beg-1 to end-1 because the python index is one less than the +1 notation
    # when the rna is 10 in length, it changes to 9 in post-translocated
    # so that means i = 9
    index = 0 # index for the keqs array
    for i in range(beg-1, end-1):

        KEQ = d3[i]

        if scrunching:
            DNA_DNA = dna_dna[i+1] # the difference is the same as the last element
        else:
            DNA_DNA = dna_dna[i+1] - dna_dna[i-12]

        # when transcript is 10 (index 9) post-translocated has 1 less
        # basepairing
        if i < 9:
            #RNA_DNA = sum(rna_dna[:i])
            RNA_DNA = 0 #
        else:
            # the 7 and the 8 are because of starting at +3, dinucleotides, and
            # python indexing 
            RNA_DNA = -rna_dna[i-9]

        expo = (b*RNA_DNA +c*DNA_DNA +d*KEQ)/RT

        rate = a*np.exp(-expo)

        keqs[index] = rate

        index += 1

    return keqs

def cost_function_scruncher(start_values, y0, its_len, state_nr, ITS_dicts, PYs,
                            non_rnap, predictive, const_par=False,
                            truth_table=False):
    """
    k1 = exp(-(c2*rna_dna_i + c3*dna_dna_{i+1} + c4*Keq_{i-1}) * 1/RT)

    Dna-bubble one step ahead of the rna-dna hybrid; Keq one step behind.
    When the rna-dna reaches length 9, the you must subtract the hybrid for
    each step forward. The hybrid oscillates between a 8bp and a
    9bp hybrid during translocation.
    """
    # get the tuning parameters and the rna-dna and k1, kminus1 values
    # RT is the gas constant * temperature
    RT = 1.9858775*(37 + 273.15)/1000  # divide by 1000 to get kcalories
    finals = []

    #outp_file = 'deltaG.txt'
    #outp_handle = open(outp_file, 'wb')

    #outp_file_keq = 'Keq.txt'
    #outp_handle_keq = open(outp_file_keq, 'wb')

    for its_dict in ITS_dicts:

        # must shift by minus 1 ('GATTA' example: GA, AT, TT, TA -> len 5, but 4
        # relevant dinucleotides)

        # first -1 because of python indexing. additional -1 because onlt (1,2)
        # information is needed for 3. ATG -> [(1,2), (2,3)]. Only DNA-DNA needs
        # both.
        dna_dna = its_dict['dna_dna_di'][:its_len-1]
        rna_dna = its_dict['rna_dna_di'][:its_len-2]
        keq = its_dict['keq_di'][:its_len-2]

        (a, b, c, d) = start_values

        # weigh each keq with the value of the RNA-DNA hybrid
        h_weight = False
        #h_weight = True

        # equilibrium constants at each position
        k1, entup = calculate_k1_difference(RT, its_len, keq, dna_dna, rna_dna,
                                            a, b, c, d, hybrid_weight = h_weight)

        #XXX here follow two hacks: writing out keq and DG variables. if you
        #uncomment this, you must be in control of what parameters you use as
        #input to make sure they match what you want to write to file

        # extract the individual free energy values 
        # these are different from above because they are the selected energies
        # that participate in Keq_i. See calc_k1_diff function for details.
        #(rnadna_array, dnadna_array, keq_array) = entup
        #if its_len == 20:
            #outp_handle.write('placeholder\n')
            #for i in range(len(rnadna_array)):
                #rd = str(rnadna_array[i])
                #dd = str(dnadna_array[i])
                #d3 = str(keq_array[i])
                #inx = str(i + 3)
                #line = '\t'.join([inx, dd, rd, d3])
                #outp_handle.write(line + '\n')

        # write the Keq values to file
        # this is bad practise, but I don't see how else you can do it without a
        # large re-haul of the code.
        #if its_len == 20:
            #outp_handle_keq.write('placeholder\n')
            #keqs = '\t'.join([str(keq) for keq in k1])
            #outp_handle_keq.write(keqs + '\n')

        # if not calculating [RNAP], simply return the sum of k1
        if non_rnap:
            output = sum(k1)
            #if its_len > 15:
                #debug()
        else:

            # extra free variable: the abortive rate
            A = equlib_matrix(k1, state_nr)

            # time in which to integrate over
            tim = 1

            # if there are 'nan' in A-matrix, return nan
            if True in np.isnan(A):
                solution = [np.nan]
            else:
                # using the x(t) = e^{A*t}*y(0) solution 
                solution = dot(scipy.linalg.expm(A*tim,q=1), y0)

            output = solution[-1] # [RNAP_final]

        # finals are the values which will be correlated with PY
        finals.append(output)

    #outp_handle_keq.close()
    #outp_handle.close()

    return np.array(finals)


def rnap_solver(y, t, A):
    """
    Solver.
    """

    dx = np.dot(A, y)

    return dx

def jacob_second(t, y, A):
    """
    Return the jacobian matrix

    X = [X0, X1, X2, X3]
    A = [[-k0,   0 , 0,  0],
         [k0 , -k1 , 0,  0],
         [0  ,  k1, -k2, 0],
         [0  ,  0  , k2, 0]]

    AX = [[-k0*X0         ],
          [k0*X0 - k1*X1  ],
          [k1*X1 - k2*X2  ],
          [k2*X2         ]]


    J = [[df1/dX0, df1/dX1, df1/dX2, df1/dX3],
         [df2/dX0, ...,                     ]
            .
            .
         [dfN/dx0, ...,              dfN/dX3]

    J = [[-k0,   0, 0, 0],
         [k0 , -k1, 0, 0],
         [0  ,  k1]
         ...

         Waitaminute, here J = A!


    """

    return A

def equlib_matrix_experimental(rate, state_nr, abortive_rate):
    """
    Try to incorporate an abortive effect back to state 0.

    Assume equlibrium for the reaction:
    X_f_0 <=> X_e_0 -> X_f_1

    Then you get d(Xf_1)/dt = k1[Xf_0]

    Example:

    seq = 'GATTA'

    X0 = A,
    X1 = T,
    X2 = T,
    X3 = A

    X0 -> X1 , k0
    X1 -> X2 , k1, X1 -> X0, K
    X2 -> X3 , k2, X2 -> X0, K

    X3 is assumed to be the final resting place

    X = [X0, X1, X2, X3]

    A = [[-k0,   K    ,   K    , 0],
         [k0 , -(k1+K),   0    , 0],
         [0  ,  k1    , -(k2+K), 0],
         [0  ,  0     ,  k2    , 0]]

    """
    # initialize with the first row
    rows = [  [-rate[0]] + [0 for i in range(state_nr-1)] ]

    for r_nr in range(state_nr-2):
        row = []

        for col_nr in range(state_nr):

            if col_nr == r_nr:
                row.append(rate[r_nr])

            elif col_nr == r_nr+1:
                row.append(-rate[r_nr+1] - abortive_rate)

            else:
                row.append(0)

        rows.append(row)

    #last row, special case: no backward rate
    rows.append([0 for i in range(state_nr-2)] + [ rate[-1], 0 ])

    # let state 0 get some feedback from the abortive states!
    for col_pos in range(len(rows[0])-1):
        if col_pos > 0 :
            rows[0][col_pos] += abortive_rate

    return np.array(rows)

def equlib_matrix(rate, state_nr):
    """
    Assume equlibrium for the reaction:
    Xf_0 <=> Xe_0 -> Xf_1

    Then you get d(Xf_1)/dt = k0[Xf_0]

    where k0 is the equilibrium constant (ke/kf)

    Example:

    seq = 'GATTA'

    X0 = A,
    X1 = T,
    X2 = T,
    X3 = A

    X0 -> X1 , k0
    X1 -> X2 , k1
    X2 -> X3 , k2

    X = [X0, X1, X2, X3]

    A = [[-k0,   0 , 0,  0],
         [k0 , -k1 , 0,  0],
         [0  ,  k1, -k2, 0],
         [0  ,  0  , k2, 0]]
    """

    # initialize with the first row
    rows = [  [-rate[0]] + [0 for i in range(state_nr-1)] ]

    for r_nr in range(state_nr-2):
        row = []

        for col_nr in range(state_nr):

            if col_nr == r_nr:
                row.append(rate[r_nr])

            elif col_nr == r_nr+1:
                row.append(-rate[r_nr+1])

            else:
                row.append(0)

        rows.append(row)

    rows.append([0 for i in range(state_nr-2)] + [ rate[-1], 0 ])

    return np.array(rows)


def parameter_matrix(rates, states, variables):
    """
    Construct the parameter matrix for the RNAP reaction

    Example for 3 states:

    X_f_0 <=> X_e_0 -> X_f_1
    X_f_1 <=> X_e_1 -> X_f_2
    X_f_2 <=> X_e_2 -> X_f_3

    With reaction rates
    k1_0, k2_0, k3_0
    k1_1, k2_1, k3_1
    k2_2, k2_2, K3_2

    X will be [X_f_0, X_e_0, X_f_1, X_e_1, X_f_2, X_e_2, x_f_3]

    A =
    [ [-k1_0,   k2_0        , 0    , 0             , 0    , 0              , 0],
      [ k1_0, -(k2_0 + k3_0), 0    , 0             , 0    , 0              , 0],

      [0    ,   k3_0        , -k1_1,   k2_1        , 0    , 0              , 0],
      [0    , 0             ,  k1_1, -(k2_1 + k3_1), 0    , 0              , 0],

      [0,   , 0             , 0    ,   k3_1,       , -k1_2,   k2_2         , 0],
      [0,   , 0             , 0    , 0             ,  k1_2, -(k2_2 + k3_2) , 0],

      [0,   , 0,            , 0,   , 0,            , 0,   ,   k3_2         , 0] ]

    Firs two and last row are special cases. What falls in between follows this
    pattern

    i starts in col (2i -1) = f_col (first_column)
    state i odd:
        cols before f_col = 0
        col f_col   is  k3_(i-1)
        col f_col+1 is -k1_i
        col f_col+2 is  k2_i

    state i even: starts in col 2i = f_col
        cols before 2i are 0
        col 2i   is k1_i
        col 2i+1 is -(k2_i + k3_i)

    Note that you may have to reverse the sign for k3, otherwise the reactions
    will have negative rates

    The value for 'variables' is the number of pre and pos-translocated step.
    However, you have one more 'accumulation' variable at the end. This
    corresponds to the last column of the matrix
    """

    k1, k2, k3 = rates

    rows = []

    row1 = [-k1[0], k2[0]] + [0 for i in range(variables-1)]
    row2 = [k1[0], -(k2[0] + k3[0])] + [0 for i in range(variables-1)]

    rows.append(row1)
    rows.append(row2)

    # add the middle values
    for i in range(1, states):

        # Row 1 of state_i
        f_col = 2*i -1
        row1 = []
        for col_nr in range(variables+1):

            if col_nr == f_col:
                row1.append(k3[i-1])
                #row1.append(np.exp(k3[i-1]))

            elif col_nr == f_col + 1:
                row1.append(-k1[i])

            elif col_nr == f_col + 2:
                row1.append(k2[i])

            else:
                row1.append(0)

        rows.append(row1)

        # Row 2
        f_col = 2*i
        row2 = []
        for col_nr in range(variables+1):

            if col_nr == f_col:
                row2.append(k1[i])

            elif col_nr == f_col + 1:
                row2.append(-(k2[i] + k3[i]))
                #row2.append(-(k2[i] + np.exp(k3[i])))

            else:
                row2.append(0)

        rows.append(row2)

    rowN = [0 for i in range(variables -1)] + [k3[-1], 0]

    rows.append(rowN)

    return np.array(rows)

def finals2py(finals, PYs):
    """
    Take the final output concentrations and return PY values
    """
    # make a linear model from concentrations to PYs
    slope, intercept, corr, pval, stderr = scipy.stats.linregress(finals,
                                                                  PYs)
    # make a predictive linear function
    def predicted_PY(x, slope, intercept):
        return  slope*x + intercept

    return [predicted_PY(f) for f in finals]


def candidate_its(ITSs, DNADNAfilter=True):
    """
    Based on your models you must now suggest new sequences.

    Previously you could solve all 13-variants without problem. Now, running 10,
    you're starting to run into some serious trouble. It's taking a long time.
    You won't be able to test all possible variants. I think you should test all
    possible variants for the first 8;  well, 10 took like 15 minutes. that's
    not too bad. Will be 7 on work computer.

    Update: you have changed the model a bit and need to make new sequences.
    """

    # where the output should be printed
    outfile = open('output/predicted_seqs', 'wb')

    plt.ion()
    par = get_global_params()

    PYs = np.array([itr.PY for itr in ITSs])*0.01

    # Range of its values
    max_its = 16
    its_range = range(3, max_its)

    grid_size = 10

    # initial grid
    c1 = np.array([par['K']]) # insensitive to variation here
    c2 = np.array([0]) # rna-dna to zero
    c3 = np.linspace(-par['dd_max'], par['dd_max'], grid_size)
    c4 = np.linspace(-par['eq_max'], par['eq_max'], grid_size)

    par_ranges = (c1, c2, c3, c4)

    # use the optimal parameters froom the optimization
    # get average from +11 to +15
    av_range = range(10,16)
    #av_range = range(10,13)
    opt_par = get_optimal_params(PYs, its_range, ITSs, par_ranges,
                                 opt_nucs = av_range)

    # Why haven't you saved the parameters! :S:? that will take forever to get
    # at home.
    debug()

    beg = 'AT'
    N25 = 'ATAAATTTGAGAGAGGAGTT'

    #pruning_stop = 10 # stop after this many variations; trim the dataset; continue
    pruning_stop = 10
    # 5 => 0.4, 6 => 1.9, 7 =>8.9, 8 => 32.8, 9 => 128
    # At work: 7 => 1.95,  8 => 7.2, 9 => 30.1

    sample_nr = 26

    # batch size (multiprocess in batches)
    batch_size = 300

    its_len = 15 # the ultimate goal
    #its_len = 13

    # get a dict of some candidate sets
    candidates = get_final_candidates(beg, pruning_stop, sample_nr, batch_size,
                                      opt_par, its_len, av_range, DNADNAfilter)

    for_saving = {}
    #Get how well its doing with the naive model
    for set_nr, candidate_set in candidates.items():

        predPys, seqs, ost = zip(*candidate_set)

        set_Enrange = naive_energies(ITSs, new_set=seqs)

        print spearmanr(set_Enrange, predPys)

        r, pval = spearmanr(set_Enrange, predPys)

        for_saving[set_nr] = ["spearmar: {0:.2f}".format(r)]

    # Print the G, A, T, C distribution from the 5 best.
    for set_nr, entries in candidates.items():
        seqs = [e[1] for e in entries]

        As = sum([t.count('A') for t in seqs])
        Gs = sum([t.count('G') for t in seqs])
        Ts = sum([t.count('T') for t in seqs])
        Cs = sum([t.count('C') for t in seqs])

        tot = sum([As, Gs, Ts, Cs])

        #percentages
        Ap = As*100/float(tot)
        Gp = Gs*100/float(tot)
        Tp = Ts*100/float(tot)
        Cp = Cs*100/float(tot)

        print set_nr
        pres = 'A: {0:.2f}. G: {1:.2f}. T: {2:.2f}. C: {3:.2f}'.\
                format(Ap, Gp, Tp, Cp)
        print pres

        for_saving[set_nr].append(pres)

    # print the same distribution for the native ITS up to 15
    print('')
    print('Compare this with the distribution of nucs for the ITS up to 15')
    its_seqs = [s.sequence[:15] for s in ITSs]

    As = sum([s.count('A') for s in its_seqs])
    Gs = sum([s.count('G') for s in its_seqs])
    Ts = sum([s.count('T') for s in its_seqs])
    Cs = sum([s.count('C') for s in its_seqs])

    tot = sum([As, Gs, Ts, Cs])

    #percentages
    Ap = As*100/float(tot)
    Gp = Gs*100/float(tot)
    Tp = Ts*100/float(tot)
    Cp = Cs*100/float(tot)

    print 'Its up to 15'
    print 'A: {0}. G: {1}. T: {2}. C: {3}'.format(Ap, Gp, Tp, Cp)

    # XXX Print the DNA-DNA energy distribution compared to ITS and random.
    dna_dna = {}
    for set_nr, candidate_set in candidates.items():

        ens = []
        predPys, seqs, ost = zip(*candidate_set)
        for cand_seq in seqs:
            ens.append(Ec.DNA_DNAenergy(cand_seq[:15]))

        dna_dna[set_nr] = ens

    DNA_its = [Ec.DNA_DNAenergy(s.sequence[:15]) for s in ITSs]

    save_result(outfile, ITSs, for_saving, dna_dna, candidates, opt_par, its_len)

    fig, ax = plt.subplots()

    ax.hist(DNA_its, normed=True, alpha=0.7, color = 'r')

    for (new_set, col) in zip(dna_dna.values(), ['b', 'g', 'k', 'c', 'yellow']):

        ax.hist(new_set, normed=True, alpha=0.4, color=col)

    # TODO save each set with some info: the predicted PY range, the nucleotide
    # distribution, and the correlation with the naive model.

    # RESULT You get a correlation coefficient between 0.81 and 0.88 for the
    # 'naive' and the 'diff' models full set, 15 nt. I'm not so sure about how
    # to interpret that.

    # Q: how does this look if I optimize for RNA-DNA instead?

    # What about the issue that Hsu's sequences are a lot more A/T rich than
    # expected? Yours have the A a lot more, but not the others. 
    # A: 35.3846153846. G: 26.4102564103. T: 21.0256410256. C: 17.1794871795

    # vs ITS:
    # A: 36.8992248062. G: 15.1937984496. T: 27.5968992248. C: 20.3100775194

    # By using the RNA-centric model I have:
    #A: 35.1282051282. G: 22.5641025641. T: 23.0769230769. C: 19.2307692308

    # This model is definitely pushing in the direction of the ITS

    # These sequences are not followign the normal distirbution. That's the
    # selection pressure. But is there more than that?

def print_already_ordered_samples(ITSs, outfile, already_used, params, its_len):
    """
    DG401-DG406 have been ordered. Where do they fit into the grander scheme of
    things? UPDATE: The DG401-DG406 as well as the DG426 to DG452 have been
    ordered. You'll need to check all of them.

    You will have to re-simulate these sequences and then run conc2py on them.
    Then you can fit them into some of the templates you have already produced.

    So .. you will rerun them with exactly the same input as was used to
    generate the optimal sequences. Then you will get their concentrations. Then
    you will simulate the original ITS. The concentration of the original ITS
    will give you something to compare with. You make a function from original
    ITS to PY and then apply this to your new sequences.
    """

    # make an ITS list of the ITS already ordered
    ordered_ITS = []
    for line in open(already_used, 'rb'):
        ordered_ITS.append(ITS(line.split()[1], name=line.split()[0]))

    # fake PY values; doesn't matter here
    PYs = np.array([1 for _ in range(len(ordered_ITS))])*0.01

    its_range = range(3, 21)

    all_results = scrunch_runner(PYs, its_range, ordered_ITS, params)

    final_conc = all_results[0][its_len].finals

    orderd_predPY = RNAP_2_PY(ITSs, final_conc, params=params, its_len=its_len)

    # I think it's a design flaw that you did not pass the predicted PY onto the
    # ITS objects. Then you would not need to hope that the order is preserved
    # throughout.

    names = [its.name for its in ordered_ITS]

    # write 
    for (predPY, name) in sorted(zip(orderd_predPY, names)):
        outfile.write('\t'.join([name, str(predPY)]) + '\n')


def save_result(outfile, ITSs, for_saving, dna_dna, candidates, params, its_len):
    """
    Save this output kthnx.
    1 spearmanr nucleotide_distribution
    SEQ PY DNA-DNA
    """

    # First, print the predicted PY of the sequences you have already evaluated.
    # this way you can see how they fit into the greater scheme of things

    already_used = 'output_seqs/second_batch_dg400.txt'
    print_already_ordered_samples(ITSs, outfile, already_used, params, its_len)

    for set_nr, concs_seqs in candidates.items():

        concs, seqs, oist = zip(*concs_seqs)

        sp_nuc = for_saving[set_nr]
        spearm = sp_nuc[0]
        nuc_freq = sp_nuc[1]

        # dna-en
        DNAens = dna_dna[set_nr]
        mean_en = format(np.mean(DNAens), '.2f')
        std_en = format(np.std(DNAens), '.2f')

        # set header
        hedr = ' '.join([str(set_nr), spearm, mean_en, std_en, nuc_freq])
        outfile.write(hedr + '\n')

        # predicted PY
        predPYs = RNAP_2_PY(ITSs, concs, params=params, its_len=its_len)

        for (seq, predPY, dnaen) in zip(seqs, predPYs, DNAens):

            line = '\t'.join([seq, format(predPY, '.4f'), format(dnaen, '.2f')])

            outfile.write(line + '\n')

    outfile.close()

def get_final_candidates(beg, pruning_stop, sample_nr, batch_size,
                         params, its_len, av_range, DNADNAfilter):
    """
    Select candidates in a 2-step process. In the first step, solve up until
    pruning_stop and select 1% of the candidates in each group. Then resume,
    generating all possible combinations between pruning_stop and 15.

    """

    # In the first round, calculate until pruning_stop + 2 (AT)
    state_nr = (pruning_stop+2) - 1
    y0 = [1] + [0 for i in range(state_nr-1)]
    # change variable nr to reflect this
    variable_nr = pruning_stop

    arguments = (params, state_nr, y0)

    # Make a workers' pool
    first_pool = multiprocessing.Pool()

    # get first results
    first_results = multicore_scrunch_wrap(beg, variable_nr, batch_size,
                                         arguments, first_pool, DNADNAfilter)

    # get a dict of several candidate sets
    set_nr = 0 # this parameter doesn't matter here
    #set_subsize = 100 # get 100 sequences from each of the 25 lumpings 
    first_outp = get_py_ranges(first_results, sample_nr, set_nr, set_subsize=100)

    #Extract only the sequences from the first batch
    first_seqs = [f[1] for f in first_outp]

    # Make the second batch of results
    # fex; its_len = 15; pruning_stop=10 -> 3 variable
    variable_nr = its_len - (pruning_stop+2)
    print variable_nr

    # get second results
    batch_size = 500

    state_nr = its_len - 1 # now going full its_len
    y0 = [1] + [0 for i in range(state_nr-1)]
    arguments = (params, state_nr, y0)

    second_pool = multiprocessing.Pool()
    beg=False # don't add an AT second time around

    second_results = multicore_scrunch_wrap(beg, variable_nr, batch_size,
                                            arguments, second_pool,
                                            DNADNAfilter, start_seqs=first_seqs,
                                            multi_range=av_range)

    set_nr = 20 # get 50 final outputs to choose from

    return get_py_ranges(second_results, sample_nr, set_nr, av_range=av_range)

def multicore_scrunch_wrap(beg, variable_nr, batch_size, arguments,
                           my_pool, DNADNAfilter, start_seqs=False,
                           multi_range=False):
    """
    Return concs[+15], seqs, concs[multi_range]

    or if multi-range is False

    concs[+15], seqs
    """

    results = []
    t1 = time.time()
    for batch in seq_generator(variable_nr, batch_size, DNADNAfilter, beg=beg,
                               starts=start_seqs):

        # if you have start_seqs, send them in
        args = (batch,) + arguments + (multi_range,) # join the arguments

        # non-multi version
        #result = mini_scrunch(*args)

        result = my_pool.apply_async(mini_scrunch, args)

        results.append(result)

    my_pool.close()
    my_pool.join()

    print time.time() - t1

    # flatten the output and return
    results = sum([r.get() for r in results], [])

    return results

def get_py_ranges(all_results, sample_nr, set_nr, av_range=False,
                  set_subsize=False):
    """
    Given the set of sequences, output a set of sample_nr compartments where
    sequences are sorted linearly according to their concentrations.

    """
    if av_range:
        cons, seqs, multi_cons = zip(*all_results)
    else:
        cons, seqs = zip(*all_results)

    minConc, maxConc = min(cons), max(cons)

    # Gen the py-range from smallest to largest
    conRange = np.linspace(maxConc, minConc, sample_nr+1)[::-1]

    # Create limits for the bounding boxes
    boundaries = []
    for bound_inx, val in enumerate(conRange[:-1]):
        boundaries.append((val, conRange[bound_inx+1]))

    # Save seqs into bounding boxes according to the 'boundaries'
    savers = [[] for i in range(sample_nr)]

    pos = 0 # counter for which bounding box you are in
    for entry in sorted(all_results):

        if len(entry) == 3:
            conval, seq, multi_con = entry

        if len(entry) == 2:
            conval, seq = entry

        # change pos when you reach a boundary
        if conval > boundaries[pos][1] and pos != len(savers)-1:
            pos += 1

        if boundaries[pos][0] <= conval <= boundaries[pos][1]:
            savers[pos].append(entry)

    # If this is the first step before pruning, just return a set with
    # set_subsize sequences from each boundary
    if set_subsize:
        return randomly_selected_seqs(savers, set_subsize)

    else:
        # choose a final set where no nucleotide has more than 60% presence and
        # (every nucleotide is present at least once)
        final_sets = {}

        for i in range(set_nr):
            this_set = []
            # for each bounding box
            for bbox in savers:

                # start a bit before the middle of it
                box_start = int(len(bbox)/2-10)
                # if less than zero, start at zero
                if box_start < 0:
                    box_start = 0

                # if one of the boxes is empty, it's time to pack up and leave
                if bbox == []:
                    break

                # if not, remeove the tuple from bbox and add it to final_sets
                tup = bbox.pop(box_start)
                this_set.append(tup)

            final_sets[i+1] = this_set

        return final_sets

def randomly_selected_seqs(savers, set_subsize):
    """
    From each box in savers, select 'set_subsize' nr of random sequences. Return
    them all in a list.
    """
    outp = []

    for bbox in savers:

        # if the bbox size is less than set_subsize, add all of them
        if len(bbox) < set_subsize:
            [outp.append(t) for t in bbox]

        # if not, add 'set_subsize' random ones
        else:
            for _ in range(set_subsize):
                random_index = random.randrange(len(bbox))
                outp.append(bbox.pop(random_index))

    return outp


def variable_corr():
    """
    What power do the 3 sequence-dependent variables have? Show their
    correlation. Production friendly kthnx.
    """

    RD = Ec.NNRD
    DD = Ec.NNDD
    TR = Ec.delta_keq

    dinucs = TR.keys()

    RD_en = [RD[din] for din in dinucs]
    DD_en = [DD[din] for din in dinucs]
    TR_en = [TR[din] for din in dinucs]

    fix, axes = plt.subplots(1,3)

    RD_tup = (RD_en, '$\Delta G$ RNA-DNA')
    DD_tup = (DD_en, '$\Delta G$ DNA-DNA')
    TR_tup = (TR_en, '$\Delta G$ Translocation')

    group = [TR_tup, RD_tup, DD_tup]
    #group = [TR_tup, RD_tup]

    from itertools import combinations

    for nr, (comx, comy) in enumerate(combinations(group, 2)):

        (x_en, x_label) = comx
        (y_en, y_label) = comy

        ax = axes[nr]
        #ax = axes

        ax.scatter(x_en, y_en)
        ax.set_xlabel(x_label, size=18)
        ax.set_ylabel(y_label, size=18)
        # correlation
        #corr, p = spearmanr(x_en, y_en)
        corr, p = pearsonr(x_en, y_en)
        ax.set_title('Correlation {0:.2f}, p-value {1:.3f}\n'.format(corr, p))

        # set a label
        #for label, x, y in zip(dinucs, x_en, y_en):
            #if label not in ['GC', 'TA']:
                #continue

            #ax.annotate(
                #label,
                #xy = (x, y), xytext = (20, 20),
                #textcoords = 'offset points', ha = 'right', va = 'bottom',
                #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.3),
                #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    fix.subplots_adjust(wspace=0.5)
    fix.set_figheight(5)
    fix.set_figwidth(12)

    return fix

def get_optimal_params(PYs, its_range, ITSs, par_ranges,
                       opt_nucs='max'):

    # don't do randomization or cross validation now
    randomize = 0
    retrofit = 0

    # initial run to get optimal parameters
    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=randomize, retrofit=retrofit)

    # extract the specific results
    results, rand_results, retrof_results = all_results


    # if max, juset get parameter values for the optimal run
    if opt_nucs == 'max':
        corr_and_param = [(res.corr_max, res.params_best) for
                          res in results.values()]
        # sort and get the value with highest correlation coefficient
        op = list(sorted(corr_and_param))[-1][-1]

        debug()

        # finally get the optimal parameters in order. put them in an np array for
        # proper downstream handeling.
        optimal_params = (np.array([op['c1']]),
                          np.array([op['c2']]),
                          np.array([op['c3']]),
                          np.array([op['c4']]))

    # if not, get the average optimal parameters for the range you have
    # specified
    else:
        # the the parameters for the specified pos
        parr_list = [results[pos].params_best for pos in opt_nucs]
        # get the average parameters
        c1 = np.mean([r['c1'] for r in parr_list])
        c2 = np.mean([r['c2'] for r in parr_list])
        c3 = np.mean([r['c3'] for r in parr_list])
        c4 = np.mean([r['c4'] for r in parr_list])

        optimal_params = (np.array([c1]),
                          np.array([c2]),
                          np.array([c3]),
                          np.array([c4]))

    return optimal_params

def three_param_control_A(ITSs, testing, p_line, par):
    """
    Print three parameter model with the two controls.

    Question? What parameters to pick? Try the RNA-DNA at +18 and the DNA-DNA at
    +10? Keq for the average of the two. Lol, they are at 0.5 each.

    TODO OK, 1 thing is that you want to have the controls in plot A. But I also
    want to have controls in plot B! For this plot, I need no rands. The easiest
    way will be to remove the axes of plot B, and add a new axes with the new
    figure you have made. The question remains; do I save this information? I
    believe I do.
    """

    rands = 20 # for cross-validating and random sequences

    if testing:
        rands = 4

    # Compare with the PY percentages in this notation
    PYs = np.array([itr.PY for itr in ITSs])*0.01

    # This makes sense for the random DNA, but not for cross validation.
    # why then is there cross validation error bars in plot A?
    # it only makes sense in the case that these parameters are valid also for
    # half the dataset. ... 
    c1 = np.array([par['K']]) # insensitive to variation here
    c2 = np.array([0])
    c3 = np.array([0.25])
    c4 = np.array([-0.39])

    par_ranges = (c1, c2, c3, c4)

    its_max = 21
    #if testing:
        #its_max = 16

    its_range = range(3, its_max)

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=rands, retrofit=rands)

    # extract the specific results
    results, rand_results, retrof_results = all_results

    # ladder plot
    ymin = -0.4 # correlation is always high
    ymax = 0.9 # correlation is always high
    randomize = rands # you didn't randomize
    printB = False
    fig_lad, ax_lad = print_scrunch_ladder(results, rand_results,
                                           retrof_results, randomize,
                                           par_ranges, p_line, its_max,
                                           its_range, ymin, ymax, testing,
                                           print_params=printB)

    # calculate a new plot B
    grid_size = 15
    rands = 20

    if testing:
        grid_size = 6
        rands = 4

    # Parameter ranges you want to test out
    c1 = np.array([par['K']]) # insensitive to variation here
    #c2 = np.array([0]) # insensitive to variation here
    c2 = np.linspace(0, 1, grid_size)
    c3 = np.linspace(0, 1, grid_size)
    c4 = np.linspace(-1, 0, grid_size)

    par_ranges = (c1, c2, c3, c4)

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=0, retrofit=rands)

    # Add the new ax-2 to ax_ladder if printB is true. If not, just print the
    # mean and std of the mean 20-run cross-validations
    new_ax_two(ax_lad, all_results, its_max, par_ranges, printB)

    fig_lad.set_size_inches(7,7)

    return fig_lad, False

def new_ax_two(ax_lad, all_results, its_max, par_ranges, printB):
    """
    Add a plot with error-bars on parameter values. You modify an incoming
    object (ax_lad) so you don't need to return anything.
    """

    results, rand_results, retrof_results = all_results

    incrX = range(3, its_max)

    # assign colors to parameters
    colors_params = ['r', 'c', 'm', 'k', 'g']
    all_params = ('c1', 'c2', 'c3', 'c4', 'c5')
    labels = ['K', 'RNA-DNA', 'DNA-DNA', "3' dinucleotide"]
    # only plot parameters that are variable 

    plot_params = [p for p, r in zip(all_params, par_ranges) if len(r) > 1]

    # assign colors to parameters
    par2col = dict(zip(all_params, colors_params))
    par2label = dict(zip(all_params, labels))

    # extract the optimal parameters
    paramz_best = [r[1].params_best for r in sorted(results.items())]

    # extract the mean and std of the best cross-validated runs
    # what happens when you retrofit and randomize?
    # the mean is the mean of the 20 or so optimal parametes for each
    # independent run -- NOT the 20 suboptimal parameters for one estimation!
    paramz_mean = [r[1].params_mean for r in sorted(retrof_results.items())]
    paramz_std = [r[1].params_std for r in sorted(retrof_results.items())]

    # each parameter should be plotted with its best (solid) and mean
    # (striped) values (mean should have std)
    if printB:
        ax_lad[1].cla()
        ax = ax_lad[1]

    for parameter in plot_params:

        #print the mean and std of the top 20 parameters
        mean_par_vals = [d[parameter] for d in paramz_mean]
        std_par_vals = [d[parameter] for d in paramz_std]

        # print the best unz too
        best_par_vals = [d[parameter] for d in paramz_best]

        #get the parameter values from pos 6 to 21
        #par_vals = [retrof_results[pos].params_best[parameter] for pos in range(6,21)]

        # c2 is changing a lot when doing 22 random ITS values ...
        if parameter == 'c2':
            mean = np.mean(mean_par_vals[9:])
            std = np.std(mean_par_vals[9:])
        else:
            mean = np.mean(mean_par_vals)
            std = np.std(mean_par_vals)

        print('{0}: {1:.2f} +/- {2:.2f}'.format(parameter, mean, std))

        if not printB:
            continue

        # make the error bar plot
        ax.errorbar(incrX, mean_par_vals, yerr=std_par_vals,
                    color=par2col[parameter], linestyle='--')

        ax.plot(incrX, best_par_vals, label=par2label[parameter],
                     linewidth=2, color=par2col[parameter])


    # if you are not going to do any printing, just return
    if not printB:
        return

    # legend
    ax.legend(loc='lower right')

    # xticks
    xticklabels = [str(integer) for integer in range(3, its_max)]
    ax.set_xticks(range(3,its_max))
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(3,its_max)
    ax.set_xlabel("RNA length $n$ used to calculate SE$_n$", size=23)

    # awkward way of setting the tick font sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(15)
    for l in ax.get_yticklabels():
        l.set_fontsize(15)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
         alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

    # Add a B
    ax.text(0.03, 0.97, 'B', transform=ax.transAxes, fontsize=26,
            fontweight='bold', va='top')



def three_param_AB(ITSs, testing, p_line, par, scalad=False):
    """
    Print three parameter model with parameter estimation values.

    expo = (c2*RNA_DNA +c3*DNA_DNA +c4*KEQ)/RT

    Wow, awzm. Your new results show no role for the RNA-DNA hybrid. Also you've
    got 90% peak correlation at +11! :) Peaks at +14 instead.
    """

    # grid size
    grid_size = 15
    rands = 0 # for cross-validating and random sequences

    if testing:
        grid_size = 15

    # Compare with the PY percentages in this notation
    PYs = np.array([itr.PY for itr in ITSs])*0.01
    #PYs = np.array([-np.log(itr.APR) for itr in ITSs])*0.01 # use APR instead
    # or log (Fl/Ab), will give you the positive trend.
    PY_std = np.array([itr.PY_std for itr in ITSs])

    # Parameter ranges you want to test out
    # free it up
    c1 = np.array([par['K']]) # insensitive to variation here

    #c2 = np.linspace(0, par['rd_max'], grid_size)
    #c3 = np.linspace(-par['dd_max'], 0, grid_size)
    #c4 = np.linspace(0, par['eq_max'], grid_size)

    #c2 = np.linspace(-par['rd_max'], par['rd_max'], grid_size)
    #c2 = np.linspace(0, 1, grid_size)
    #c2 = np.linspace(0, 0, 1)
    #c3 = np.linspace(0, 1, grid_size)
    #c4 = np.linspace(-1, 0, grid_size)

    c2 = np.array([0.07]) # 
    c3 = np.array([0.26]) #
    c4 = np.array([-0.40]) #

    par_ranges = (c1, c2, c3, c4)

    its_max = 21
    #if testing:
        #its_max = 15

    its_range = range(2, its_max)
    #its_range = range(2, 15)
    #its_range = [15]
    #its_range = [10, 15, 20]
    #its_range = [10, 12, 15, 18, 20]

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=rands, retrofit=rands)

    #a hack for just returning nothing so that you can modify the deltaG.txt file
    #return 1, 2

    # extract the specific results
    results, rand_results, retrof_results = all_results

    # Make a regression model for SE_n and rna-dna
    # RESULT no effect seen
    #linear_model(ITSs, results, PYs)

    # XXX pringing params
    # print the mean and std of the estimated parameters
    #for param in ['c1', 'c2', 'c3', 'c4']:
        ##get the parameter values from pos 6 to 21
        #par_vals = [results[pos].params_best[param] for pos in range(6,its_max)]
        #if param == 'c2':
            #mean = np.mean(par_vals[8:])
            #std = np.std(par_vals[8:])
        #else:
            #mean = np.mean(par_vals)
            #std = np.std(par_vals)

        #print('{0}: {1:.2f} +/- {2:.2f}'.format(param, mean, std))

    # ladder plot
    ymin = -0.2 # correlation is always high
    ymax = 1.0 # correlation is always high
    randomize = rands # you didn't randomize
    inset_plot = False
    fig_lad, ax_lad = print_scrunch_ladder(results, rand_results,
                                           retrof_results, randomize,
                                           par_ranges, p_line, its_max,
                                           its_range, ymin,
                                           ymax, testing, print_params=True,
                                           inset=inset_plot)
    maxnuc = its_max
    if not scalad:
        # where to make the scatter plot
        # Without scatter + ladder
        fig_sct = print_scrunch_scatter(results, rand_results, randomize,
                                        par_ranges, PYs, PY_std, pos=maxnuc)

        return fig_lad, fig_sct
    else:
        # modify fig_lad to have the scatterplot in position B
        print_scrunch_scatter(results, rand_results, randomize, par_ranges, PYs,
                              PY_std, pos=maxnuc, laddax=ax_lad)

        # Add A and B if two plots
        for i, label in enumerate(('A', 'B')):
            ax = fig_lad.axes[i]
            ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=26,
                    fontweight='bold', va='top')

        return fig_lad

def linear_model(ITSs, results, PYs):
    """
    Linear regression on rna-dna hybrid strength and equilibrium fu.

    ---- THis didn't seem to improve your score. Seems it's not independent
    after all ... :S
    """
    import statsmodels.api as sm

    dgrna = [cumulated_rna_hybrid_energy(i.rna_dna_di) for i in ITSs]

    # compare with nucleotide 18
    ppn = results[18].finals

    X = np.column_stack((dgrna, ppn))
    #X = np.column_stack((ppn))

    X = sm.add_constant(X)
    #X = sm.add_constant(X)

    model = sm.OLS(PYs*100, X)

    results = model.fit()

    print results.summary()

    debug()


def RNAP_2_PY(ITSs, concentrations, params=(20, 0, 0.022, 0.24), its_len=15):
    """
    Return PY values instead of concentrations.

    NB! Make sure your concentrations were made using the same parameters and
    the same ITS length as are used to calculate the model here.
    """

    par_ranges = [np.array([p]) for p in params]

    PYs = np.array([itr.PY for itr in ITSs])*0.01

    its_range = range(3, 21)

    randomize = 0 # here 0 = False (or randomize 0 times)
    retrofit = 0

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=randomize, retrofit=retrofit)

    # extract the specific results
    results, rand_results, retrof_results = all_results

    finals = results[its_len].finals

    # make a linear model
    slope, intercept, corr, pval, stderr = scipy.stats.linregress(finals, PYs)

    # make a predictive function
    def predicted_PY(x, slope, intercept):
        return  slope*x + intercept

    # return the predicted PY values
    pred_PY = [predicted_PY(x, slope, intercept) for x in concentrations]

    return pred_PY

def predicted_vs_measured(ITSs):
    """
    Return the correlation between predicted and measured PYs

    You ran, I think, with these parameters.
    params = (15, 0, 0.022, 0.24)

    """

    # add a polynomial or sigmoid fit
    fit_function = True

    # 2) Calculate the PY scores for them.
    # TODO update these parameters
    params = (1, 0, 0.25, -0.39)
    par_ranges = [np.array([p]) for p in params]

    its_range = range(3, 21)

    randomize = 0 # here 0 = False (or randomize 0 times)
    retrofit = 0

    # 1) Get the sequences you sent which Hsu tested
    seq_file = 'output_seqs/tested_seqs.txt'
    testseqs = [l.split() for l in open(seq_file, 'rb')]

    # 2) Make ITS objects from the tested seqs
    test_obj = [ITS(seq + 'GAGTT', name) for name, seq in testseqs]

    for o in test_obj:
        print o.sequence

    # 3) Run the testseqs and get the output
    fake_py = [0 for _ in range(len(test_obj))]
    #fake_py = np.array([itr.PY for itr in ITSs])*0.01

    #test_results = scrunch_runner(fake_py, its_range, ITSs, par_ranges,
    test_results = scrunch_runner(fake_py, its_range, test_obj, par_ranges,
                                  randize=randomize, retrofit=retrofit,
                                  predictive=True)

    results, rand_results, retrof_results = test_results

    # add the SE_15 (or [RNAP])
    for its, outp in zip(test_obj, results[15].finals):
        its.score = outp

    # make a dictionary of promoter -> outp
    pred_PY = dict((its.name, (its.score, its.sequence)) for its in test_obj)

    # Get the PYs from the experiamant!
    #pred_file = 'prediction_experiment/my_summary_first_quant.csv'
    pred_file = 'prediction_experiment/Second_quantification/summary_second_quant.csv'
    tested_PY = get_new_exprimetal_py(pred_file)

    predicted, tested = zip(*[(pred_PY[pr][0], tested_PY[pr].PY) for pr in pred_PY])
    # get standard deviation
    tested_std = [tested_PY[pr].PY_std for pr in pred_PY]

    # write all of this to an output file
    write_py_SE15(tested_PY, pred_PY)

    fig, ax = plt.subplots()

    ax.scatter(predicted, tested)
    plt.errorbar(predicted, tested, yerr=tested_std, fmt=None)

    ########### Set figure and axis properties ############
    ax.set_xlabel('SE$_{15}$', size=21)
    ax.set_ylabel('PY', size=21)

    xmin, xmax = min(predicted), max(predicted)
    xscale = (xmax-xmin)*0.1
    ax.set_xlim(xmin-xscale, xmax+xscale)

    ymin, ymax = min(tested), max(tested)
    yscale_low = (ymax-ymin)*0.2
    yscale_high = (ymax-ymin)*0.3
    ax.set_ylim(ymin-yscale_low, ymax+yscale_high)

    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)
    #######################

    #spearm = spearmanr(predicted, tested)
    #pears = pearsonr(predicted, tested)
    #ax.set_title('r = {0:.2f}, p-value = {1:.2e}'.format(*spearm))

    if fit_function:
        add_fitted_function(ax, predicted, tested)

    return fig

def write_py_SE15(tested_PY, pred_PY):

    outp = open('output_seqs/seq_PY_SE15.csv', 'wb')

    for dgxx, (se15, sequence) in pred_PY.items():
        i = tested_PY[dgxx]
        seq = sequence[:15]

        outp.write('\t'.join([dgxx, seq, str(se15), str(i.PY)]) + '\n')

    outp.close()

def add_fitted_function(ax, predicted, tested):
    """
    Add a sigmoid fit to the plot
    """
    #from scipy.odr import odrpack as odr
    #from scipy.odr import models

    def sigmoid_function(B, x):
        """
        B is the parameters and x is the SE_15
        """

        #return B[0]/(1+np.exp(-x))
        #return B[0]/(B[1]+np.exp(-x))
        return B[0]/(B[1]+B[2]*np.exp(-x))

    def sigm_error(B, x, y):
        """
        """
        return y - sigmoid_function(B, x)

    B0 = [1, 1, 1]
    x = np.array(predicted)
    y = np.array(tested)

    # Normal least squares
    fit = optimize.leastsq(sigm_error, B0, args=(x, y))
    outp_args = fit[0]
    fit_vals = [sigmoid_function(outp_args, x_val) for x_val in x]
    # sort by increasing x-value
    (sort_x, sort_fit) = zip(*sorted(zip(x, fit_vals)))

    # odd_ least squares
    #my_model = odr.Model(sigmoid_function)
    #my_data = odr.Data(x,y)
    #my_odr = odr.ODR(my_data, my_model, beta0=B0)
    ## fit type 2 for least squares
    #my_odr.set_job(fit_type=2)
    #fit = my_odr.run()
    # XXX I didn't get a nice fit this way, but keep code just in case

    ax.plot(sort_x, sort_fit, linewidth=2)

def get_new_exprimetal_py(pred_file):
    """
    Read the experimental data and get the predicted values
    Return them as ITS objects
    the file is:
        Header
        Promoter PY1 PY2 PY3 ...
    """

    newITS = {}
    dummy_seq = 'GATTACA'

    for line in open(pred_file, 'rb'):
        # skip first line
        if line.startswith('Promoter'):
            continue
        contents = line.split()

        # extract promoter data
        promoter = contents[0] # promoter name
        PYs = [float(c) for c in contents[1:]] # py values

        mean_PY = np.mean(PYs)
        std_PY = np.std(PYs)

        # make an ITS object
        itsObj = ITS(dummy_seq, name=promoter, PY=mean_PY, PY_std=std_PY)

        newITS[promoter] = itsObj

    return newITS


def get_global_params():
    # for SE_n
    global_params = {'K': 1,
                     'dd_best': 0.25, # approximately
                     'rd_best': 0.01, # approximately
                     'eq_best': -0.39, # approximately
                     'dd_min': 0,
                     'dd_max': 1,
                     'rd_min': 0,
                     'rd_max': 1,
                     'eq_min': -1,
                     'eq_max': 0}
    # for [RNAP]
    #global_params = {'K': 1,
                     #'dd_best': 0.2, # approximately
                     #'rd_best': 0.2, # approximately
                     #'eq_best': 0.6, # approximately
                     #'dd_min': 0.001,
                     #'dd_max': 1.4,
                     #'rd_min': 0.001, # this one goes pretty much to zero
                     #'rd_max': 1.4,
                     #'eq_min': 0.1,
                     #'eq_max': 1.7}

    return global_params

def equilibrium_vs_py(ITSs, testing, global_params):
    """
    Plot the relationship between the accumulated value of the equilibrium
    constant and the PY value.

    Define cumulative propensity of backtracking (CPB)
    Sum(Keq_i)

    Alternatively call it the abortive propensity

    Hey, Jude. Can you pass an option to the functinos you've already written
    that's called CPB. If True, you would calculate not the [RNAP_i] but the
    CPB. Should be possibleh. Even enable it by default.

    """
    PYs = [its.PY for its in ITSs]

    par = global_params
    grid_size = 10

    c1 = np.array([par['K']]) # insensitive to variation here
    #c2 = np.linspace(par['rd_min'], par['dd_max'], grid_size)
    c2 = np.array([0]) # DNA-DNA
    c3 = np.linspace(-par['dd_max'], -par['dd_min'], grid_size) # RNA-DNA
    c4 = np.linspace(par['eq_min'], par['eq_max'], grid_size) # 3D

    par_ranges = (c1, c2, c3, c4)

    #stages = range(21)
    its_lens = [5, 10, 15, 20]

    # The RT constant
    RT = 1.9858775*(37 + 273.15)/1000 # divide by 1000 to get kcalories

    # store the top results for each its_len
    results = dict((ilen, []) for ilen in its_lens)

    for its_len in its_lens:
        # go over each parameter combination
        for param_comb in itertools.product(*par_ranges):
            # calculate the CPB for all ITS for this parameter combination
            (a, b, c, d) = param_comb

            for its in ITSs:
                # calculate the 3d, dd, and rd (the -1/2 just works)
                keq = its.keq_delta_di[:its_len-2]
                dna_dna = its.dna_dna_di[:its_len-1]
                dna_dna = its.dna_dna_di[:its_len-2]
                ddlen = len(dna_dna)
                dna_dna = dna_dna[:10] + [0 for i in range(ddlen-10)]
                rna_dna = its.rna_dna_di[:its_len-2]
                rdlen = len(rna_dna)
                rna_dna = [0 for i in range(10)] + rna_dna[-(rdlen-10):]

                # the accumulated k1 up to its_len
                k1 = calculate_k1_difference(RT, its_len, keq, dna_dna, rna_dna,
                                             a, b, c, d)

                its.k1 = sum(k1)
                #its.k1 = sum(-np.array(keq) + np.array(dna_dna) - np.array(rna_dna))
                #its.k1 = sum(-np.array(keq) + np.array(dna_dna))/RT

            # extract the PY and k1 values
            PYs, k1s = zip(*[(its.PY, its.k1) for its in ITSs])
            corr, pval = spearmanr(PYs, k1s)

            results[its_len].append((pval, corr))

    for its_len, outp in results.items():
        print its_len
        best_res = sorted(outp)
        debug()

    fig, ax = plt.subplots()
    ax.scatter(k1s, PYs)

    ax.set_xlabel('Sum of translocation equilibrium constants')
    ax.set_ylabel('PY')

    return fig

def paper_figures(ITSs):
    """
    The figures you want to include in the paper.

    """
    testing = True  # if testing, run everything fast
    #testing = False

    # show in the figure name that this is a testrun
    if testing:
        append = '_testing'
    else:
        append = ''

    # collect all figures for saving at the end
    figs = []

    # add a pvalue-line to the ladder plots
    p_line = True

    # global parameters
    global_params = get_global_params()

    # Figure 0 -> Keq equilibrium and PY values
    # UPDATE: now doing this with previous approach: just passing argument
    #equilib_name = 'equilibrium_vs_py'
    #equilib_fig = equilibrium_vs_py(ITSs, testing, global_params)
    #figs.append((equilib_fig, equilib_name))

    ### Figure 1 and 2 -> Three-parameter model with parameter estimation but no
    ###cross-reference. Return either one or two figures.
    #ladder_name = 'three_param_model_AB' + append
    #scatter_name = 'three_param_14_scatter' + append
    #scatterladder = True
    ##scatterladder = False
    #fig_back = three_param_AB(ITSs, testing, p_line, global_params,
                              #scalad=scatterladder)

    #if scatterladder:
        #fig_ladder = fig_back
    #else:
        #fig_ladder, fig_scatter = fig_back
        #figs.append((fig_scatter, scatter_name))

    #fig_ladder.set_size_inches(17,7)
    #figs.append((fig_ladder, ladder_name))

    #Figure 2.5 -> Three-parameter model with controls 
    #ladder_nog_name = 'three_param_control_AB' + append
    #fig_nog_ladder, fig_nog_scatter = three_param_control_A(ITSs, testing,
                                                            #p_line,
                                                            #global_params)
    #figs.append((fig_nog_ladder, ladder_nog_name))

    ### Figure 2.7 -> Compare two and three parameter models
    compare_name = 'compare_two_three_AB' + append
    fig_controversy = compare_two_three(ITSs, testing, p_line, global_params)
    figs.append((fig_controversy, compare_name))

    ### Figure 3 -> The RNA DNA hybrid as a positive stabilizing force
    #rna_stable_name = 'RNA_stabilizing' + append
    #fig_rna_stable = rna_stable(ITSs, testing, p_line, global_params)
    #figs.append((fig_rna_stable, rna_stable_name))

    ##Figure 4 -> Predicted VS actual PY
    #evaluation_name = 'Predicted_vs_measured' + append
    #fig_evaluation = predicted_vs_measured(ITSs)
    #figs.append((fig_evaluation, evaluation_name))

    ### Figure 5 -> Selection pressures
    #predicted_name = 'Selection_pressure' + append
    #fig_predicted = selection_pressure(ITSs)
    #figs.append((fig_predicted, predicted_name))

    ### Figure 5.5 -> Selection pressures ITS-style
    #predicted_name = 'ITS_centrict_selection' + append
    #fig_predicted = ITS_centric_selection(ITSs)
    #figs.append((fig_predicted, predicted_name))

    #### Figure 6 -> Model family
    #family_name = 'Model_family' + append
    #fig_family = family_of_models(ITSs, p_line, global_params)
    #figs.append((fig_family, family_name))

    #### Figure 8 -> Correlation between DNA variables
    #variable_name = 'Variable_correlation' + append
    #fig_variable = variable_corr()
    #figs.append((fig_variable, variable_name))

    #### Figure 11 -> x-mers correlation; which x-mers are most important?
    #x_mer_name = 'x_mers' + append
    #fig_x_mer = xmer(ITSs, testing, p_line, global_params)
    #figs.append((fig_x_mer, x_mer_name))
    #RESULT your figure shows more or less what you wanted -- but should you
    #include it?

    ### Figure 12 -> the good old abortive probability vs keq
    # New idea: maybe not abortive probability but some other measure? And: only
    # use the values between + 5 and +10. Will be 5*43 measures anyway.

    ### Can you make a simple possion process where the probability to backtrack
    #occurs so and so often? backtracking is not assumed to be fast; full length
    #product shuold not take more than 3 seconds to be made. 

    # Save the figures
    for (fig, name) in figs:
        for fig_dir in fig_dirs:
            for formt in ['pdf', 'eps', 'png']:

                odir = os.path.join(fig_dir, formt)

                if not os.path.isdir(odir):
                    os.makedirs(odir)

                savename = name + '.' + formt

                fig.savefig(os.path.join(odir, savename), transparent=True,
                            format=formt)

def rna_stable(ITSs, testing, p_line, global_params):
    """
    Simply the accumulated DGRNA values
    """

    PYs = [i.PY for i in ITSs]

    dgrna = []

    for i in ITSs:
        dgrna.append(cumulated_rna_hybrid_energy(i.rna_dna_di))

    print spearmanr(PYs, dgrna)
    print pearsonr(PYs, dgrna)

    fig, ax = plt.subplots()

    ax.scatter(dgrna, PYs)
    ax.set_xlabel('PY')
    ax.set_xlabel('Cumulated \Delta G_{RNA-DNA}')

    return fig

def cumulated_rna_hybrid_energy(rna_dna_en):
    """
    """
    en = []
    for i in range(18):
        # index 8 is the same as x_11. k[8] is for x_11
        if i < 8:
            RNA_DNA = sum(rna_dna_en[:i])
        else:
            # the 7 and the 8 are because of starting at +3, dinucleotides, and
            # python indexing 
            RNA_DNA = sum(rna_dna_en[i-8:i])

        en.append(RNA_DNA)

    return sum(en)

def xmer(ITSs, testing, p_line, global_params):
    """
    Get the Keq for x-mers and correlate with PY.

    12345678
    ATGCTGAC

    [z0(AT), z1(TG), z2(GC), z3(CT), z4(TG), z5(GA), z6(AC)]
    """

    # 1 a function for 
    RT = 1.9858775*(37 + 273.15)/1000   # divide by 1000 to get kcalories
    a, b, c, d = 1, 0, 0.25, -0.39

    # the keq, dnadna and rna_dna values must have the [0] at x-mer start and
    # its_len for x-mer length. Example. 4-mer starting at +3
    # keq = delta_keq[2:]
    # its_len = 4
    # But that's a problem because its_len is assumed to be the length from 1 to
    # n. so if n is < 10, the rna-dna will not be calculated correctly. now,
    # that doesn't matter so much, but still. can you solve this by passing a
    # parameter which
    # 
    # the function below

    PYs = [i.PY for i in ITSs]

    x_mers = range(6,9)

    #lstyles = ['-', '--', '-.', ':', '.', ',', 'o', 'x', '*']

    fig, ax = plt.subplots()

    for x in x_mers:
        x_energies = []
        for beg in range(1,21-x):
            end = beg + x
            # stop when your x-mer has reached the end
            if end > 20:
                continue

            # a column for the (beg, end) energy for each ITS -- this is the one
            # you correlate with PY
            its_energies = []

            for its in ITSs:

                d3 = its.keq_delta_di
                dna_dna = its.dna_dna_di
                rna_dna = its.rna_dna_di

                keqs = calculate_keq(RT, beg, end, d3, dna_dna, rna_dna, a, b,
                                     c, d, scrunching=True)

                its_energies.append(sum(keqs))

            x_energies.append(its_energies)

        # now you can make a plot for this x_mer
        corr = [spearmanr(PYs, ens)[0] for ens in x_energies]

        #fig, ax = plt.subplots()
        #ax.bar(x_ax, corr)

        x_ax = range(2, len(corr)+2)
        ax.plot(x_ax, corr, label=str(x))

    ax.set_xlabel("Start for sum of equilibrium constants")
    ax.set_ylabel("Correlation coefficient")

    ax.set_title("X-mer: {0}".format(x))

    ax.set_xticks(range(2,21))
    ax.legend()

    return fig

def compare_two_three(ITSs, testing, p_line, par):
    """
    Adding each variable sequentially

    What does this mean?
    exp(dna-dna): GC-rich -> low rate. AT-rich -> high rate

    exp(-rna-dna): GC-rich -> high rate. AT-rich -> low rate

    exp(-keq) NEW: Pre-translocated -> low rate, post-translocated -> high rate

    In the old model Keq has a positive sign

    exp(keq) OLD: pre-translocated -> low rate, post-translocated -> high rate

    This actually makes perfect sense, now it's just the correlation that is
    strange.
    """

    # no cross validation needed
    control = 0

    # Compare with the PY percentages in this notation
    PYs = np.array([itr.PY for itr in ITSs])*0.01

    its_max = 21
    its_range = range(3, its_max)

    grid_size = 10

    if testing:
        grid_size = 5

    # initial grid
    c1 = np.array([par['K']]) # insensitive to variation here
    c2 = np.linspace(0, 1, grid_size)
    c3 = np.linspace(0, 1, grid_size)
    c4 = np.linspace(-1, 0, grid_size)

    # define the ranges for the two minimal models
    zero = np.array([0])
    one_param = (c1, zero, zero, c4)
    two_param = (c1, zero, c3, c4)
    three_param = (c1, c2, c3, c4)

    # use the optimal parameters from the optimization
    # XXX at the moment you will use the 
    #two = get_optimal_params(PYs, its_range, ITSs, two_param, optim, t)

    #Building up slowly
    name1 = '$\Delta G_{3D}$'
    name2 = '$\Delta G_{3D} + \Delta G_{DNA-DNA}$'
    name3 = '$\Delta G_{3D} + \Delta G_{DNA-DNA}$ + $\Delta G_{RNA-DNA}$'
    #three = get_optimal_params(PYs, its_range, ITSs, three_param, optim, t)

    collection = [(one_param, name1),
                  (two_param, name2),
                  (three_param, name3)]

    # collect the results
    resulter = {}

    # fix this to use the average best score; not the best score at each point.
    # XXX that will make the graphs look a bit better all round.

    for (par_ranges, name) in collection:

        all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                     randize=control, retrofit=control)

        # extract the specific results
        results, rand_results, retrof_results = all_results

        params = []

        # average parameters
        for param in ['c1', 'c2', 'c3', 'c4']:
            #get the parameter values from pos 6 to 21
            par_vals = [results[pos].params_best[param] for pos in range(6,its_max)]
            if param == 'c2':
                mean = np.mean(par_vals[8:])
                std = np.std(par_vals[8:])
            else:
                mean = np.mean(par_vals)
                std = np.std(par_vals)

            params.append(np.array([mean]))

            print('{0}: {1:.2f} +/- {2:.2f}'.format(param, mean, std))

        par_ranges = tuple(params)

        # run again, now with averaged parameters
        all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                     randize=control, retrofit=control)
        # extract the specific results
        results, rand_results, retrof_results = all_results

        # store result with name
        resulter[name] = results

    ymin = -0.2 # correlation is always high
    ymax = 1.0 # correlation is always high

    #fig, axes = plt.subplots(1,2)
    fig, ax = plt.subplots()

    # plot them 
    colors = ['b', 'g', 'c', 'k', 'r']
    for name, results in resulter.items():
        color = colors.pop()
        compare_plot(ax, name, results, color, its_max, p_line, ymin, ymax)

    fig.set_size_inches(7,7)

    return fig

def compare_plot(ax, name, results, colr, its_max, p_line, ymin, ymax):
    """
    Plot all the ladder plots on top of each other.
    """

    # get its_index andecorr-coeff from sorted dict
    indx, corr, pvals = zip(*[(r[0], r[1].corr_max, r[1].pvals_min)
                       for r in sorted(results.items())])

    # make x-axis
    incrX = range(indx[0], indx[-1]+1)

    # check for nan in corr (make it 0)
    corr, pvals = remove_nan(corr, pvals)
    ax.plot(incrX, corr, label=name, linewidth=4, color=colr)

    # interpolate pvalues (x, must increase) with correlation (y) and
    # obtain the correlation for p = 0.05 to plot as a black
    if p_line and name == 'Original':
        # hack to get pvals and corr coeffs sorted
        pv, co = zip(*sorted(zip(pvals, corr)))
        f = interpolate(pv, co, k=1)
        ax.axhline(y=f(0.05), ls='--', color='r', linewidth=3)

    xticklabels = [str(integer) for integer in range(3, its_max)]
    #Make sure ymin has only one value behind the comma
    ymin = float(format(ymin, '.1f'))
    yticklabels = [format(i ,'.1f') for i in np.arange(ymin, 1.1, 0.1)]

    # legend
    ax.legend(loc='lower right')

    # xticks
    ax.set_xticks(range(3,its_max))
    #ax.set_xticklabels(xticklabels)
    ax.set_xticklabels(odd_even_spacer(xticklabels, oddeven='odd'))
    ax.set_xlim(3,its_max)
    ax.set_xlabel("RNA length (nt)", size=21)

    # awkward way of setting the tick font sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)

    ax.set_ylabel("Correlation coefficient, $r$", size=21)

    ax.set_yticks(np.arange(ymin, 1.1, 0.1))
    #ax.set_yticklabels(yticklabels)
    ax.set_yticklabels(odd_even_spacer(yticklabels))
    ax.set_ylim(ymin, 1.0001)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

    #for i, label in enumerate(('A', 'B')):
        #if i == ax_nr:
            #ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=26,
                    #fontweight='bold', va='top')


def ITS_centric_selection(ITSs):
    """
    The nucleotide distribution in the ITSs is not uniform. Calculate the
    energies for each dinucleotide from +1 to +20 for the 43 ITS and for all the
    E coli ITS.

    The result is that you don't really see any kind of selection in the 43
    compared to WT e coli. Maybe that needs to go from your paper. Check the
    previous distributions again this time by removing the three first?
    """

    # Get the DNA-DNA, translocation, and RNA-RNA energies for the first 10
    # nucleotides. Then obtain the distribution from completely random DNA.
    # Do it for the first 10 and then for the second 10

    # Mandatory AT start
    DD = []
    RD = []
    TR = []
    # Sum the energies of each dinucleotide for all 43 ITS variants
    # DD[0] will be the sum of DNA-DNA energies of the first dinucleotide
    for i in range(0,20):
        dna_ens = [Ec.DNA_DNAenergy(its.sequence[i:i+2]) for its in ITSs]
        rdna_ens = [Ec.RNA_DNAenergy(its.sequence[i:i+2]) for its in ITSs]
        tr_ens = [Ec.Delta_trans(its.sequence[i:i+2]) for its in ITSs]

        DD.append(sum(dna_ens)/len(dna_ens))
        RD.append(sum(rdna_ens)/len(rdna_ens))
        TR.append(sum(tr_ens)/len(tr_ens))


    # Do the same but for the native e coli ITS
    DD_native = []
    RD_native = []
    TR_native = []

    its_native = get_ecoli_ITS(sigma70=True, one_promoter=False, min50UTR=True,
                              ITS_len=20)
    for i in range(0,20):
        dna_ens = [Ec.DNA_DNAenergy(its[i:i+2]) for its in its_native]
        rdna_ens = [Ec.RNA_DNAenergy(its[i:i+2]) for its in its_native]
        tr_ens = [Ec.Delta_trans(its[i:i+2]) for its in its_native]

        DD_native.append(sum(dna_ens)/len(dna_ens))
        RD_native.append(sum(rdna_ens)/len(rdna_ens))
        TR_native.append(sum(tr_ens)/len(tr_ens))

    fig, axes = plt.subplots(nrows=3, ncols=1)

    # prepare the energies in a logical way
    its_e = [DD, RD, TR]
    ecoli_e = [DD_native, RD_native, TR_native]

    ylabels = ['DNA-DNA', 'RNA-DNA', 'Translocation']

    for row_nr in range(3):

        ax = axes[row_nr]

        the_43_y = np.abs(its_e[row_nr]) # absolute value
        wt_y = np.abs(ecoli_e[row_nr])

        ylabel = ylabels[row_nr]

        nr_elements = len(wt_y)

        step = 1.5
        # position of the_43
        the_43_x = np.arange(0.5, step*(nr_elements), step)
        wt_x = np.arange(1, step*(nr_elements)+0.5, step)

        ax.bar(the_43_x, the_43_y, width=0.5, color='g')
        ax.bar(wt_x, wt_y, width=0.5, color='b')

        ax.set_ylabel(ylabel)

        ax.set_xticklabels(range(1,21))

    return fig

def selection_pressure(ITSs):
    """
    The nucleotide distribution in the ITSs is not uniform. What has caused the
    selection pressure? Hsu's bet is on the DNA-DNA. According to the
    literature, modern random PCA (after mid 90s) is no longer biased toward any
    nucleotide distribution. Whatever selection has occured is due to the
    selection steps Hsu has made.

    Good results! Question: the ITS with 'normal' DNA-DNA folding, do they make
    up for it with translocation energies? For the first 15, choose those
    DNA-DNA sequences with less than -17 DNA-DNA and calculate their
    translocation values. Plot this in the translocation histogram.

    OK, now, what do you want to show? I think all plots are relevant. Compared
    to random, because this is the potential ITS library. Compared to ITS
    because this is the 'background bias'. Then make a 3x2 and let the bossed
    decide
    """

    # Get the DNA-DNA, translocation, and RNA-RNA energies for the first 10
    # nucleotides. Then obtain the distribution from completely random DNA.
    # Do it for the first 10 and then for the second 10

    # Mandatory AT start
    DD15 = [Ec.DNA_DNAenergy(i.sequence[3:15]) for i in ITSs]
    RR15 = [Ec.RNA_DNAenergy(i.sequence[3:15]) for i in ITSs]
    TR15 = [Ec.Delta_trans(i.sequence[3:15]) for i in ITSs]

    # Generate 100000 random DNA of length 10 and calculate the above energies
    # WITH AT START
    rDD15 = []
    rRR15 = []
    rTR15 = []

    eDD15 = []
    eRR15 = []
    eTR15 = []

    #its_native = get_ecoli_ITS(sigma70=True, one_promoter=True, min50UTR=True)
    its_native = get_ecoli_ITS(sigma70=True, one_promoter=False, min50UTR=True)

    for its in its_native:
        eDD15.append(Ec.DNA_DNAenergy(its[3:]))
        eRR15.append(Ec.RNA_DNAenergy(its[3:]))
        eTR15.append(Ec.Delta_trans(its[3:]))

    for random_dna in ITS_generator(10000, length=12, ATstart=False):
        rDD15.append(Ec.DNA_DNAenergy(random_dna))
        rRR15.append(Ec.RNA_DNAenergy(random_dna))
        rTR15.append(Ec.Delta_trans(random_dna))

    fig, axes = plt.subplots(nrows=3, ncols=2)

    # prepare the energies in a logical way
    its_e = [DD15, RR15, TR15]
    rand_e = [rDD15, rRR15, rTR15]
    ecoli_e = [eDD15, eRR15, eTR15]

    ylabels = ['DNA-DNA', 'RNA-DNA', 'Translocation']

    for row_nr in range(3):
        for col_nr in range(2):

            ax = axes[row_nr, col_nr]

            ax.hist(its_e[row_nr], label='DG100 ITS',
                    alpha=0.8, color='r', normed=True, bins=10)

            if col_nr == 0:
                ax.hist(rand_e[row_nr], bins=20, alpha=0.3, color='g',
                        normed=True)
                        #normed=True, label='Random DNA')

            if col_nr == 1:
                ax.hist(ecoli_e[row_nr], bins=20, alpha=0.3, color='g',
                        normed=True)
                        #normed=True, label='Native E coli ITS')

            if col_nr == 0:
                ax.set_ylabel(ylabels[row_nr])

            ax.set_yticklabels([])

            if row_nr == 2:
                ax.set_xlabel('Energy')

            if row_nr == 0 and col_nr == 0:
                ax.set_title('Random DNA')

            if row_nr == 0 and col_nr == 1:
                ax.set_title('Native E coli ITS')

            if row_nr == 0 and col_nr == 0:
                ax.legend(loc = 'upper left')

    # set xticks. Get the max/min for this row; make a linspace of 5
    # values from the min/max. put that on both rows. 
    # pass through, ge
    for row_nr in range(3):
        c1 = axes[row_nr, 0]
        c2 = axes[row_nr, 1]

        minr = min(c1.get_xlim()[0], c2.get_xlim()[0])
        maxr = max(c1.get_xlim()[1], c2.get_xlim()[1])

        xrang = np.linspace(minr, maxr, 5)
        xvals = [int(r) for r in xrang]

        for ax in [c1, c2]:
            ax.set_xlim(minr, maxr)
            ax.set_xticks(xrang)
            ax.set_xticklabels(xvals)

    fig.subplots_adjust(wspace=None, hspace=0.4)

    return fig

    # Dow low DNA-DNA have  high translocation? Yes.

    ## Low DNA-DNA
    #limit = -18
    #lowDD = [i for i in ITSs if Ec.DNA_DNAenergy(i.sequence[:15]) < limit]

    #mini_trans = [Ec.Delta_trans(i.sequence[:15]) for i in lowDD]
    #print len(mini_trans)

    #TR10_mean, TR10_std = np.mean(TR10), np.std(TR10)
    #mini_mean, mini_std = np.mean(mini_trans), np.std(mini_trans)

    #fiG, aX = plt.subplots()

    #aX.bar([0.5, 1.5], [-TR10_mean, -mini_mean], alpha=0.3, width=0.5)
    #aX.errorbar([0.75, 1.75], [-TR10_mean, -mini_mean], yerr = [TR10_std, mini_std],
                #fmt=None)

    #aX.set_xlim(0, 2.5)
    #aX.set_xticks([0.75, 1.75])
    #aX.set_xticklabels(['All ITS ({0})'.format(len(TR10)),
                        #'ITS with DNA energy < {0} ({1})'.format(limit, 
                                                                 #len(lowDD))])
    #aX.set_ylabel('Translocation energy')

    ## do a welchs test
    #n1 = len(TR10)
    #mean1 = TR10_mean
    #sem1 = TR10_std/np.sqrt(n1)

    #n2 = len(lowDD)
    #mean2 = mini_mean
    #sem2 = mini_std/np.sqrt(n2)

    #alpha = 0.05

    #welchs_approximate_ttest(n1, mean1, sem1, n2, mean2, sem2, alpha)

    # XXX you can do the same with bar-plots. It shows that the low-DNA-DNA
    # variants compensate by having very high translocation values! Wonderful!


    ### XXX do the same with the last 10, but now choose completely random DNA
    ### last 8 last 6?
    ### No mandatory start
    #DD20 = [Ec.DNA_DNAenergy(i.sequence[-10:]) for i in ITSs]
    #RR20 = [Ec.RNA_DNAenergy(i.sequence[-10:]) for i in ITSs]
    #TR20 = [Ec.Delta_trans(i.sequence[-10:]) for i in ITSs]

    ## Generate 100000 random DNA of length 10 and calculate the above energies
    ## WITHOUT AT START
    #rDD20 = []
    #rRR20 = []
    #rTR20 = []

    #for random_dna in ITS_generator(10000, length=10, ATstart=False):
        #rDD20.append(Ec.DNA_DNAenergy(random_dna))
        #rRR20.append(Ec.RNA_DNAenergy(random_dna))
        #rTR20.append(Ec.Delta_trans(random_dna))

    #fig2, [ax12, ax22, ax32] = plt.subplots(3)

    #ax12.hist(DD20, label='DG100 Last 10 nt of ITS', alpha=0.7, color='r',
              #normed=True)
    #ax12.hist(rDD20, bins=40, alpha=0.3, color='g', normed=True,
              #label='Random DNA starting with AT length 10')
    #ax12.set_title('DNA-DNA: ITS and random compared')

    #ax22.hist(RR20, label='DG100 Last 10 nt of ITS', alpha=0.7, color='r',
              #normed=True)
    #ax22.hist(rRR20, bins=40, alpha=0.3, color='g', normed=True,
              #label='Random DNA starting with AT length 10')
    #ax22.set_title('RNA-DNA: ITS and random compared')

    #ax32.hist(TR20, label='DG100 Last 10 nt of ITS', alpha=0.7, color='r',
              #normed=True)
    #ax32.hist(rTR20, bins=40, alpha=0.3, color='g', normed=True,
              #label='Random DNA starting with AT length 10')
    #ax32.set_title('Translocation: ITS and random compared')

    #debug()

def s_shaped(ITSs):
    """
    Generate sequences to test if DNA-DNA energy has an s-shaped effect

    Looks like your TR should stay around -7.5 (-8), and your DNA should vary
    from -15 to -23. At around -18 you should see the change. Get at least 3
    seqs below - 18, and 3 seqs above - 18.
    -23, -21, -19, -17, -15, -13, would be perfect. I don't know if it's
    possible though. We should see a model-mismatch starting at around - 18,
    because no such sequence should have survived, at least not at around -23.

    Can we estimate curves for DNA-DNA and translocation? DNA-DNA being S-shaped
    and translocation being more linear? Can you try to do that? Hey, that's not
    really how it looks when you plot them one by one, but that could be because
    of mutual effects. How can you separate those effects? With PCA? Seems a bit
    redundant with only two variables. There is a DNA-DNA at -19, recall, with a
    PY of 7. It has the lowest TR though. That's the kind of stuff you should
    normalize by. Is that what you get with a linear regression? No ... Well
    yes; make a linear regression; with DNA-DNA from 0 to 5; 5 to 10; 10 to 15,
    and the same with TR. You would just see the same again. The interesting
    thing would be if your model or the linear regression would predict that if
    you hold TR fixed at -7.5, you'd lose PY after a certain level, but with
    DNA-DNA fixed at say -15, you can allow for a larger variation in TR.

    However, I think you'll need to show that these sequences give a different
    value than expected in the model. The model should predict a non-zero value
    for them, but in all likelihood they will be zeroed. Maybe reduce the value
    from -8 to -7? Or -7.5. You'll need to write an ad hoc function from model
    outptut to PY. That's good anyway. You should fit that to something. It's
    nonlinear because model output has a smaller range than PY. Maybe linear
    regression here? The relationship is linear after all. Then you can invert
    them model and go from PY -> X and X -> PY the way you like it. Too bad you
    don't have a way of speeding up the results.

    Maybe you should dissalow not less than -18 but less than -20? Then maybe
    you can still see some oddities around there.

    Hey ... I've just had me a look, and for the lowest DNA-DNA, there's also a
    large variation in TR -> seems to be just as non-linear in this direction.
    However, there as no selection pressure on this parameter. Annoying.

    Could this have anything to do with the fact that the screening happened in
    vivo? Is it possible that ... I don't see how that fits together.

    You see that within the dna-dna population (after screening) there is a
    correlation where dna-dna energy decreases with increasing TR. Is it
    combinatorially more difficult to screen out tr than dna-dna?
    """

    # What's the PY of the < -17 DNA-DNA ITSs?
    # What's the PY of the > -7

    #gg = [s.PY for s in ITSs if Ec.DNA_DNAenergy(s.sequence[:15]) < -17]
    #print gg

    #tap = []

    #dforplot = []

    #for s in ITSs:
        #tr = Ec.Delta_trans(s.sequence[:15])
        #dd = Ec.DNA_DNAenergy(s.sequence[:15])

        ##if tr > -9:
        #if tr < -4:
        ##if dd > -15:
        ##if dd < -15:
            #tap.append((tr, dd, s.PY))
            ##print 'TR:', tr, 'DNA:', dd, 'PY:', s.PY

    #for pp in sorted(tap, key=itemgetter(0,1)):
        #dforplot.append(pp[1])
        #print('TR: {0:.1f}, DNA: {1:.1f}. PY: {2:.2f}'.format(*pp))

    #plt.ion()
    #plt.plot(dforplot)

    # Make a bunch of random DNA sequences and get them in the -7.8 to -8.2
    # range for TR; from those, sort them by DNA-DNA energy low to high.
    # Hopefully you'll get something nice out of it.

    round1 = []

    # RESULT it's easy to generate this kind of sequences 
    # XXX however, now you've proved that DNA-DNA works like S-shaped, but how
    # do you show that the effect of TR is more linear? By keeping DNA-DNA fixed
    # and show that you can vary TR over a larger range with a linear increase?
    # But by this time, you don't have any left to test your model. 26 -12 = 14.
    # Maaaaaybe it's enough
    # A way to show the linear relationship could be to divide the sequences in
    # two: low and high DNA-DNA and low and high TR
    # The high and low TR should both show the nonlinearity of DNA-DNA (measured
    # how?, with a linear model? The low DNA-DNA should give a larger
    # coefficient to DNA-DNA, while the high-DNA-DNA should give a smaller
    # coefficient to DNA-DNA. The high and low TR should give a similar
    # coefficient to both DNA-DNA and TR in both groups. The problem is that
    # DNA-DNA is so tightly bundled that you are unlikely to see any effect in
    # the low-group -> They won't be very discernable.

    # Include the same checks as for the other model.
    #repReg = re.compile('G{5,}|A{5,}|T{5,}|C{5,}')
    #nucleotides = set(['G', 'A', 'T', 'C'])

    #for seq in ITS_generator(100000, length=15):
        #if -8.1 < Ec.Delta_trans(seq) < -7.9:

            ## don't let the first three nucs be the same
            #if (seq[1] == seq[2] == seq[3]):
                #continue

            ## don't allow repeats of more than 4
            #if repReg.search(seq):
                #continue

            #nfreqs = [seq.count(n)/15.0 for n in nucleotides]
            ## none should be above 60%
            #if max(nfreqs) > 0.6:
                #continue

            #round1.append((Ec.DNA_DNAenergy(seq), seq))

    #ham = sorted(round1, key=itemgetter(0))

    # pick 6 from max to min

    # XXX try to make linear regression with statsmodels for the DNA-DNA and KT
    # XXX result, no significance for DNA-DNA when cutting group in half.
    # XXX abandon this project? I still propse that DNA-DNA has got this effect.
    # Maybe you can propose six sequences and 20 normal. 20 should be enough to
    # see a trend. You must add this routine to your proposaling.

    # for up to 15. Use weighted least squares for heteroscledascial errors.
    # your model should be a y = ax1 + bx2 + c, whre x1 and x2 is the
    # exponential of your energies. If that works satisfactorially, repeat the
    # seame for the low/high dna/dna. Compare the coefficients.
    #g1 = [s for s in ITSs if Ec.DNA_DNAenergy(s.sequence[:15]) <= -15]
    #g2 = [s for s in ITSs if Ec.DNA_DNAenergy(s.sequence[:15]) > -15]

    #for group in [g1, g2]:

        #PY = np.array([s.PY for s in group])
        #DD = np.array([Ec.DNA_DNAenergy(s.sequence[:15]) for s in group])
        #TR = np.array([Ec.Delta_trans(s.sequence[:15]) for s in group])

        ## This model works to make it linear. You have to fit the coefficients of
        ## 0.1 and 0.2. Can you obtain the same by taking the ln of PY? 

        ## If that wor
        ## get data

        #import scikits.statsmodels.api as sm

        #X = sm.add_constant(np.column_stack((DD, TR)))
        #y = np.log(PY)

        ### run the regression
        #results = sm.WLS(y, X).fit()

        ### look at the results
        #results.summary()

        #c1, c2, c3 = results.params

        #print results.summary2

    ##simplemod = c1*DD + c2*TR + c3

    ##plt.ion()
    ##plt.scatter(simplemod, np.log(PY))

    ### investrigate the seqs you sent to hsu
    # XXX result: it looks OK. It's not too bad.
    testseqs = [l.split()[1] for l in open('seqs_for_testing.txt', 'rb')]
    for t in testseqs:
        tr = Ec.Delta_trans(t)
        dd = Ec.DNA_DNAenergy(t)
        print 'TR: {0:.1f}, DNA: {1:.1f}'.format(tr, dd)


def get_ecoli_ITS(sigma70=True, one_promoter=True, min50UTR=True, ITS_len=15):
    """
    Return the ITS sequences of E coli.

    if sigma70 -> only get sigma 70 promoters

    if one_promoter -> only get ITS from genes with 1 promoter

    if min50UTR -> only get ITS from genes with a 5utr of minimum 50 nt

    This is currently not yet implemented. You could do it like this: find a
    list of all the translation initiation sites, and for each of your
    transcription intiation site, just search each of them for the distance. I
    think I've done something like that.
    """

    # e coli sequence
    ecoli = 'sequence_data/ecoli/ecoli_K12_MG1655'
    ecoli_seq = ''.join((line.strip() for line in open(ecoli, 'rb')))

    sigprom_dir = 'sequence_data/ecoli/sigma_promoters'
    sig_paths = glob(sigprom_dir+'/*')

    sig_dict = {}
    for path in sig_paths:
        fdir, fname = os.path.split(path)
        sig_dict[fname[8:15]] = path


    # save ITS sequences here
    seqs = []

    for sigma, sigpath in sig_dict.items():

        # see if restricted to sigma70
        if sigma != 'Sigma70' and sigma70:
            continue

        for row_pos, line in enumerate(open(sigpath, 'rb')):

            (pr_id, name, strand, pos, sig_fac, seq, evidence) = line.split('\t')
            if seq == '':
                continue

            # get only those with 1 promoter <-> 1 gene
            if not name.endswith('p') and one_promoter:
                continue

            # use the position to access the e coli genome
            pos = int(pos)

            if strand == 'forward':
                seqs.append(ecoli_seq[pos-1:pos+ITS_len])

            elif strand == 'reverse':
                seqs.append(reverseComplement(ecoli_seq[pos-ITS_len:pos]))

    return seqs

def DG400_verifier(ITSs):
    """
    Read the predicted set and run through the model. Note deviations from
    linearity and PY values. If not too different, keep'em.
    """

    # 2) Calculate the PY scores for the origianl PY to get a linear score
    params = (20, 0, 0.3, 0.8)
    par_ranges = [np.array([p]) for p in params]

    PYs = np.array([itr.PY for itr in ITSs])*0.01
    PYerr = np.array([itr.PY_std for itr in ITSs])*0.01


    its_range = range(3, 21)
    randomize = 0 # here 0 = False (or randomize 0 times)
    retrofit = 0

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges,
                                 randize=randomize, retrofit=retrofit)

    # extract the specific results
    results, rand_results, retrof_results = all_results

    finals = results[15].finals

    # make a linear model
    slope, intercept, corr, pval, stderr = scipy.stats.linregress(finals, PYs)

    # make a predictive function
    def predicted_PY(x, slope, intercept):
        return  slope*x + intercept

    minX, maxX = min(finals), max(finals)

    minY, maxY = slope*minX + intercept, slope*maxX + intercept

    plt.scatter(finals, PYs)
    plt.errorbar(finals, PYs, yerr=PYerr, fmt=None)
    plt.plot([minX, maxX], [minY, maxY])

    # 1) Get the sequences you sent to Hsu
    seq_file = 'sequence_data/seqs_for_testing.txt'
    testseqs = [l.split() for l in open(seq_file, 'rb')]

    # 2) Run the testseqs and get the output
    test_obj = [ITS(seq + 'GAGTT', name) for name, seq in testseqs]

    fake_py = [0 for _ in range(len(test_obj))]

    test_results = scrunch_runner(fake_py, its_range, test_obj, par_ranges,
                                  optim, t, randize=randomize, retrofit=retrofit)

    results, rand_results, retrof_results = test_results

    pred_PY = [predicted_PY(f, slope, intercept) for f in results[14].finals]

    min_pred, max_pred = min(pred_PY), max(pred_PY)

    lin_range = np.linspace(min_pred, max_pred, num=len(pred_PY))

    print('\nCorrelation: r:{0:.3f}, p:{1:.3f} '.format(*pearsonr(lin_range,
                                                                pred_PY)))
    print('\nHAI\n')
    for py in pred_PY:
        print py

def write_test(ITSs):
    """
    Just test your method for re-testing the test0rs ;)
    """

    outfile = open('cheeseblog.txt', 'wb')

    ordered_samples = 'output_seqs/seqs_for_testing.txt'

    params = ()
    c1 = np.array([20]) # insensitive to variation here
    c2 = np.array([0])
    c3 = np.array([0.3])
    c4 = np.array([0.6])

    params = (c1, c2, c3, c4)

    its_len = 15

    # calculate the predicted PY and write to file
    print_already_ordered_samples(ITSs, outfile, ordered_samples, params, its_len)
    # close file
    outfile.close()

def up_hill_vs_elongation(ITSs):
    """
    Show that there is a cumulative effect here compared to elongation
    """
    fig, [ax, ax2, ax3] = plt.subplots(1,3)

    first_its = ITSs[0]

    # add up hill contribuition
    added_up_hill = [sum(first_its.dna_dna_di[1:i]) for i in range(2,21)]

    # add pre-sequence for the elongation model
    pre_tss_seq = 'AGCATGCATGCTAGATG'
    pre_en = [Ec.NNDD[pre_tss_seq[i:i+2]] for i in range(len(pre_tss_seq)-2)]
    join_en = pre_en + first_its.dna_dna_di

    elongation = [sum(join_en[i+1:i+15]) - sum(join_en[i:i+14]) for i in
                  range(2,21)]

    # add elongation contributiomn
    added_elongation = [sum(elongation[1:i]) for i in range(2,21)]

    cumul_en = [sum(first_its.dna_dna_di[:i]) for i in range(1,20)]

    # add cumul contribution
    added_cumul = [sum(cumul_en[:i]) for i in range(1,20)]

    its_X = range(2,21)

    # Plot the up-hill versus cumulative vs elongation added up
    ax2.plot(its_X, added_up_hill, label='"Up hill" model, added', color='g', lw=3)
    ax2.plot(its_X, added_elongation, label='"Transcription elongation", added',
            color='b', lw=3)
    ax2.plot(its_X, added_cumul, label='"Cumulative", added', color='r', lw=3)

    ax2.set_xlabel("Position of the ITS", size=20)
    ax2.set_ylabel("$\Delta G_{DNA-DNA}$ added up", size=20)

    ax2.legend(loc='lower left')

    ax2.set_ylim(-120,10)

    # Plot the up-hill versus cumulative vs elongation at each point
    model_en = first_its.dna_dna_di[1:20]
    cumul_en = [sum(first_its.dna_dna_di[:i]) for i in range(1,20)]

    ax.plot(its_X, model_en, label='"Up hill", each position', color='g', lw=3)
    ax.plot(its_X, elongation, label='"Transcription elongation", each position',
            color='b', lw=3)
    ax.plot(its_X, cumul_en, label='"Cumulative", each position', color='r', lw=3)

    ax.legend(loc='lower left')
    ax.set_xlabel("Position of the ITS", size=20)
    ax.set_ylabel("$\Delta G_{DNA-DNA}$ at each position", size=20)

    # Plot the % of increase at each point
    norm_model = [100*abs((abs(model_en[i+1]) - abs(model_en[i]))/abs(model_en[i]))
                  for i in range(18)]

    # Add the from -11 thing
    norm_cumul = [100*abs((abs(cumul_en[i+1]) - abs(cumul_en[i]))/abs(cumul_en[i]))
                  for i in range(18)]

    its_X = range(3,21)

    ax3.plot(its_X, norm_model, label='"Up hill"', color='g', lw=3)
    ax3.plot(its_X, norm_cumul, label='"Cumulative"', color='r', lw=3)

    ax3.set_xlabel("Position of the ITS", size=20)
    ax3.set_ylabel("Percentage change of $\Delta G_{DNA-DNA}$ at each position",
                   size=20)

    ax3.legend(loc='upper left')

    for axx in [ax, ax2, ax3]:

        axx.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.8)
        axx.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.8)

    return fig

def uphill_vs_cumulative(ITSs):
    """
    Compare the current, 'up hill' model versus the cumulative model.

    Can you show that the information contained in both is identical? Normalize
    both values to 1 and you will see the same ups and downs. A problem with
    cumulative is that the % of change decreases. This means that each
    nucleotide has comparatively less and less to say for the matter.
    """

    #testing = False
    testing = True
    p_line = True
    global_params = get_global_params()

    fig_ladder, fig_scatter = three_param_AB(ITSs, testing, p_line, global_params)

def no_dnadna(ITSs):
    """
    There is no correlation between DNADNA energy at +10, +15, and +20 and PY.
    """

    fig, axes = plt.subplots(1,3)

    PYs = [its.PY for its in ITSs]

    for plt_nr, pos in enumerate([10, 15, 20]):

        DNA_en = [sum(its.dna_dna_di[:pos]) for its in ITSs]

        ax = axes[plt_nr]

        ax.scatter(DNA_en, PYs)

        ax.set_xlabel('Cumulative DNA-DNA energy up to +{0}'.format(pos))

        corr, p = pearsonr(DNA_en, PYs)
        ax.set_title('Correlation {0:.2f}, p-value {1:.3f}\n'.format(corr, p))

    axes[0].set_ylabel('PY')

    return fig


def third_report_figures(ITSs):
    """
    Make the 4-5 plots which you think argue your case.
    """

    the_figs = []

    #1) Cumulative increase in DNA-DNA energy compared to elongation
    # The cumulative term is not used in each step -- but the initiation complex
    # has suffered an uphill battle against -17 dG at +15 
    #name = 'three_kings'
    #fig = up_hill_vs_elongation(ITSs)
    #fig.set_size_inches(20,10)
    #the_figs.append((name, fig))

    #2) Model output comparing the two methods of calculating DNA-DNA energy 
    # Upper plot is 'current, 'up hill'' model, lower plot is cumulative energy.
    # Maybe show in a table the DNADNA contribution at each step?
    # XXX you did this manually.
    #name = 'uphill_vs_cumulative'
    #fig = uphill_vs_cumulative(ITSs)
    #the_figs.append((name, fig))

    #3) Absence between correlation of DNA-DNA at +10, +15, and +20 for the PY
    name = 'no_dna_correlation'
    fig = no_dnadna(ITSs)
    the_figs.append((name, fig))
    fig.set_size_inches(20,10)

    formt = 'pdf'
    odir = '/home/jorgsk/phdproject/The-Tome/my_papers/'\
            'rna-dna-paper/third_report/figures'

    for name, fig in the_figs:

        if not os.path.isdir(odir):
            os.makedirs(odir)

        savename = '.'.join([name, formt])

        fig.savefig(os.path.join(odir, savename), transparent=True,
                    format=formt)

def ratio_issue(ITSs):
    """
    Compare old to new PY ratio
    """

    common_promoters = ['N25', 'DG115a', 'DG133', 'N25/A1anti']

    newPY_file = 'prediction_experiment/summary.txt'

    newFileObj = open(newPY_file, 'rb')

    # read the entries from the file
    promoters = newFileObj.next().split()[1:]
    totalRNA = newFileObj.next()
    abortiveRNA = newFileObj.next()
    fulllenRNA = newFileObj.next()
    abortiveYield = newFileObj.next()
    productiveYield = newFileObj.next().split()[1:]
    apRatio = newFileObj.next()

    newITS = {}

    for promoter, PY in zip(promoters, productiveYield):
        if promoter not in newITS:
            newITS[promoter] = float(PY)
        else:
            newITS[promoter] = (newITS[promoter] + float(PY))/2

    ratios = []
    for promoter in common_promoters:
        new = newITS[promoter]
        old = [its.PY for its in ITSs if its.name==promoter][0]

        ratio = new/old
        ratios.append(ratio)

        print(promoter)
        print("New/old ratio: {0}".format(ratio))
        print('\n')

    print('\n')
    print('Mean and std of ratios: {0}, {1}'.format(np.mean(ratios),
                                                    np.std(ratios)))
def compare_quantitations():
    """
    Compare round 1 and round 2 of quantitations
    """
    pred_file1 = 'prediction_experiment/my_summary_first_quant.csv'
    pred_file2 = 'prediction_experiment/Second_quantification/summary_second_quant.csv'

    pys = []
    errs = []
    for pfile in [pred_file1, pred_file2]:
        tested_PY = get_new_exprimetal_py(pfile)
        pys.append([its.PY for its in tested_PY.values()])
        errs.append([its.PY_std for its in tested_PY.values()])

    plt.scatter(pys[0], pys[1])
    plt.xlabel('First quantitation (PY)')
    plt.ylabel('Second quantitation (PY)')
    plt.plot(range(30), range(30))

    plt.errorbar(pys[0], pys[1], xerr=errs[0], yerr=errs[1], fmt=None)

    corrs = spearmanr(pys[0], pys[1])

    header = 'r = {0:.2f}, p = {1:.2e}'.format(corrs[0], corrs[1])
    plt.title(header)

def outp_write(ITSs):
    """
    """
    infile = 'deltaG.txt'
    outfile = 'DG.txt'

    out_handle = open(outfile, 'wb')

    for line in open(infile, 'rb'):

        if line.startswith('placeholder'):
            its_obj = ITSs.pop(0)
            name, py = its_obj.name, its_obj.PY

            new_line = 'ITS_variant:{0}\nPY:{1}'.format(name, py)

            out_handle.write(new_line + '\n')
        else:
            out_handle.write(line)

    out_handle.close()

def keq_outp_write(ITSs):
    """
    """
    infile = 'Keq.txt'
    outfile = 'Keq_data_and_PY.txt'

    out_handle = open(outfile, 'wb')

    for line in open(infile, 'rb'):

        if line.startswith('placeholder'):
            its_obj = ITSs.pop(0)
            name, py = its_obj.name, its_obj.PY

            new_line = 'ITS_variant:{0}\nPY:{1}'.format(name, py)

            out_handle.write(new_line + '\n')
        else:
            out_handle.write(line)

    out_handle.close()

def abortive_initiation_fromwhere(ITSs):
    """
    Three different models of where abortive initiation occurs from:

        1) Pre
        2) Pre and Post
        3) Post

    These models are tested against 5 hypothetical sequences which have the
    following sum of Keq:

    Very high
    high
    medium
    low
    very low

    These sums can be obtained in two ways: varying k1 or k2 while keeping the
    other constant. Thus I get 6 output sets.

    What is Keq? In my calculations it varies from 0.49 to 2.72. Incidentally
    this is within the same range as obtained by Hein. et al, which reflects
    that it's mostly the dinucleotide making its impact.

    I believe that reverse-translocation plays a much larger role during
    scrunching than during normal RNAP transcription elongation. Mean is 1.5.

    1) Get something simple working.

    You need to easily define the abortive probabilities
    [0.1 for i in range(10)]

    The k1 and k2 (one constant, the other between some high/low to achieve
    mean values between 0.5 and 2.5).

    You should vary the Keq values and see how it varies the variability. Think
    hard about your benchmarks.

    The problem is the following parameters:

        backtracking rate
        forward rate (k3)
        time of simulation (t)

    when you vary them, your conclusions may vary. further, those parameters are
    not constant at all; they probably depend on scrunching. but what you
    actually wanted to find out is if there is some kind of mechanistic problem
    with having abortive initiation happening from the post-state.

    RESULT you have not varied them, and overall the backtracking from pre has
    got the best results, hands down.

    Now you need to finish the three models. Then you need to summarize your
    arguments. You spent 3 whole days doing this.
    """

    # name, py, and keq of ITS
    its_info = keq_reader()

    # the length of 'full length RNA' -- assuming promoter escape happens once
    # RNA has reached this length
    rna_max = 12

    # if rna_max is 12, there are 10 k3 nucleotide incorporation reactions
    # Before each nucleotide incorporation state there are _2_ post/pre states.
    # Thus, there are len(k3)*2 post/pre states. Then there are additionally 2
    # FL + A states.
    state_nr = (rna_max-2)*2 + 2

    model_types = ['Pre', 'Pre_and_Post', 'Post']
    #variator_coefficients = ['k1']
    variator_coefficients = ['k2'] # k2 gives smaller values but same pattern

    y0 = [1] + [0 for _ in range(state_nr-1)]

    # the three nettlesome parameters
    b_rate = 0.1 # vary from 0.05 to 0.5
    t = 15 # vary from 1 to 100
    k3_coeff = 1 # vary from 0.5 to 1.5

    #b_rates = (0.5-0.05)*np.random.rand(100) + 0.05
    #times = (100-1)*np.random.rand(100) + 1
    #k3_coeffs = (1.5 - 0.5)*np.random.rand(10) + 0.5

    b_rates = (0.5-0.05)*np.random.rand(50) + 0.05
    times = (100-1)*np.random.rand(50) + 1
    k3_coeffs = (1.5 - 0.5)*np.random.rand(5) + 0.5
    #k3_coeffs = [1]

    t1 = time.time()

    # create the parameter space
    parameter_space = []

    for b_rate in b_rates:
        for t in times:
            for k3_coeff in k3_coeffs:

                parameter_space.append((b_rate, t, k3_coeff))


    # divide the parameter space into 3
    nr_cpu = 4
    ss = int(len(parameter_space)/float(nr_cpu)) # subspace size
    subspaces = [parameter_space[i*ss:i*ss+ss] for i in range(nr_cpu)]

    # make a multiprocessing pool
    my_pool = multiprocessing.Pool(nr_cpu)

    results = []
    # 
    for param_subspace in subspaces:
        arguments = (param_subspace, model_types, rna_max, variator_coefficients,
                     its_info, state_nr, y0)

        results.append(my_pool.apply_async(model_launcher, (arguments)))

        #results = model_launcher(*arguments)

    my_pool.close()
    my_pool.join()

    # gather the output
    all_results = [r.get() for r in results]

    # separate fold from std
    outp_folds, outp_stds = zip(*all_results)

    print time.time() - t1

    # Print the output
    fig1, ax1 = plt.subplots()

    # for the mean
    mean_fold = {}
    mean_std = {}

    # for keeping all folds and stds
    all_folds = {}
    all_std = {}

    for mtype in model_types:
        # sum up all the cpu_nr different dictionaries and flatten to 1 list
        mtype_folds = sum([subfold[mtype] for subfold in outp_folds], [])
        ax1.plot(mtype_folds, label = mtype)

        mean_fold[mtype] = np.mean(mtype_folds)
        all_folds[mtype] = np.array(mtype_folds)

    ax1.set_title('PY fold (max/min) for the different structures')
    ax1.legend()

    fig2, ax2 = plt.subplots()

    for mtype in model_types:
        mtype_stds = sum([substd[mtype] for substd in outp_stds], [])
        ax2.plot(mtype_stds, label = mtype)

        mean_std[mtype] = np.mean(mtype_stds)

        all_std[mtype] = np.array(mtype_stds)

    ax2.set_title('Standard deviation of PY for the different structures')
    ax2.legend()

    print('Variating: {0}'.format(variator_coefficients))

    for mtype in model_types:
        print ('Mean max/min fold difference of PY values for {0}: {1:.2f}'\
               .format(mtype, mean_fold[mtype]))

    for mtype in model_types:
        print ('Mean std of PY values for {0}: {1:.2f}'\
               .format(mtype, mean_std[mtype]))

    # Print how often Pre is higher than pre_and_post and post
    pre_gr_than_post_fold = all_folds['Pre'] > all_folds['Post']
    pgtpf_pcnt = sum([1 for i in pre_gr_than_post_fold if i ==
                      True])/len(all_folds['Pre'])

    print('Percentage folds with Pre > Post: {0:.2f}'.format(pgtpf_pcnt))

    pre_gr_than_prepost_fold = all_folds['Pre'] > all_folds['Pre_and_Post']
    pgtppf_pcnt = sum([1 for i in pre_gr_than_prepost_fold if i ==
                      True])/len(all_folds['Pre'])
    print('Percentage folds with Pre > Pre_and_Post: {0:.2f}'.format(pgtppf_pcnt))

    pre_gr_than_post_std = all_std['Pre'] > all_std['Post']

    pgtps_pcnt = sum([1 for i in pre_gr_than_post_std if i ==
                      True])/len(all_folds['Pre'])

    print('Percentage std with Pre > Post: {0:.2f}'.format(pgtps_pcnt))

    pre_gr_than_prepost_std = all_std['Pre'] > all_std['Pre_and_Post']
    pgtpps_pcnt = sum([1 for i in pre_gr_than_prepost_std if i ==
                      True])/len(all_folds['Pre'])

    print('Percentage std with Pre > Pre_and_Post: {0:.2f}'.format(pgtpps_pcnt))

    print('')



def model_launcher(parameter_spaces, model_types, rna_max, variator_coefficient,
                  its_info, state_nr, y0):

    # I will vary the three parameters and store the fold and std output
    outp_fold = dict((m_type, []) for m_type in model_types)
    outp_std = dict((m_type, []) for m_type in model_types)


    for (b_rate, t, k3_coeff) in parameter_spaces:

        #  "rate" of backtracking, separate for prE and posT translational states
        aT_original = [b_rate for _ in range(rna_max-2)]
        aE_original = [b_rate + b_rate*0.5 for _ in range(rna_max-2)]
        #aE_original = [b_rate for _ in range(rna_max-2)]

        # "rate" of nucleotide incorporation
        k3 = [k3_coeff for _ in range(rna_max-2)]

        its2py = model_evaluator(model_types, aT_original, aE_original,
                                 t, variator_coefficient, its_info,
                                 state_nr, k3, y0)

        # Benchmark in X ways:
            # 1) the correlation with the actual PY values
            # 2) The min max ratio compared to actual PY
            # 3) the distance vector with the actual py (parameter sensitive)

        # get the names and the PY in the order of those names
        its_names = its_info.keys()
        PY = [float(its_info[name][0]) for name in its_names]

        for mod_type, variator_dict in its2py.items():
            for variator, its_pydict in variator_dict.items():

                # extract PY from model
                new_PY = [its_pydict[name]*100 for name in its_names]

                # correlation
                cor, pval = pearsonr(PY, new_PY)

                minval = min(new_PY)
                maxval = max(new_PY)
                fold = maxval/minval
                std = np.std(new_PY)

                outp_fold[mod_type].append(fold)
                outp_std[mod_type].append(std)

    return (outp_fold, outp_std)


def model_evaluator(model_types, aT_original, aE_original, t,
                    variator_coefficient, its_info, state_nr, k3, y0):

    its2py = AutoVivification()

    for model_type in model_types:

        if model_type == 'Pre':
            aT = [0 for _ in aT_original] # no aborive from PosT
            aE = aE_original

        elif model_type == 'Pre_and_Post':
            aT = aT_original
            aE = aE_original

        elif model_type == 'Post':
            aE = [0 for _ in aE_original] # no abortive from PrE
            aT = aT_original

        for variator in variator_coefficient:

            # Make a matrix for each of the ITS variants; solve the model; and
            # get the PY output from the two last states
            for its_name, (PY, keqs) in its_info.items():

                # evaluate k1/k2 based on keq = k1/k2
                # if variator is k1, then k2 is = 1 and k1 varies
                k1, k2 = rates_from_keq(keqs, variator)

                # keq matrix
                A = keq_matrix(state_nr, aT, aE, k1, k2, k3)

                # using the x(t) = e^{A*t}*y(0) solution 
                soln = dot(scipy.linalg.expm(A*t,q=3), y0)

                # get abortive and full length
                fl, ab = soln[-2:]

                its2py[model_type][variator][its_name] = (fl)/(fl + ab)

    return its2py




def rates_from_keq(keqs, variator):
    """
    Evaluate k1/k2 based on keq = k1/k2
    if variator is k1, then k2 is = 1 and k1 varies
    """

    if variator == 'k1':
        k2 = [1 for _ in range(len(keqs))]
        k1 = [float(keq) for keq in keqs] # dividing keqs by 1 is keqs

    if variator == 'k2':
        k1 = [1 for _ in range(len(keqs))]
        k2 = [1/float(keq) for keq in keqs]

    return k1, k2

def keq_matrix(state_nr, aT, aE, k1, k2, k3):
    """
    Return matrix of equilibrium process depending on model type

    Sequence = GACTA

    there are (len(seq)*2 - 2) post/pre states + 2 F/A states

    Pre1  Post1 Pre2  Post2
     # <->  # -> # <-> # -> F
     |      |    |     |
     A      A    A     A

    F = full length
    A = abortive

    Vector = [Pre1, Post1, Pre2, Post2, F, A]

    Matrix =
    [

    [-(k11 + aE1), k21,        0, 0, 0, 0, 0, 0],
    [k11, - (k21 + k31 + aT1), 0, 0, 0, 0, 0, 0],

    [0, k31, - (k12 + aE2),     k22, 0, 0, 0, 0],
    [0, 0, k12, -(k22 + k32 + aT2),  0, 0, 0, 0],

    [0, 0, 0, k32, - (k13 + aE3),    k23,  0, 0], # This is extrapolated
    [0, 0, 0, 0, k13, -(k23 + k33 + aT3),  0, 0], #

    [0, 0, 0,                   0, 0, k33, 0, 0],
    [aE1, aT1, aE2, aT2, aE3, aT3,         0, 0]

    ]

    ['-(k11 + aE1)',       'k21', 0, 0, 0, 0, 0, 0],
    ['k11', '-(k21 + k31 + aT1)', 0, 0, 0, 0, 0, 0],

    [0, 'k31',  '-(k12 + aE2)', 'k22', 0, 0, 0, 0],
    [0, 0, 'k12', '-(k22 + k32 + aT2)', 0, 0, 0, 0],

    [0, 0, 0, 'k32',   '-(k13 + aE3)', 'k23', 0, 0],
    [0, 0, 0, 0, 'k13', '-(k23 + k33 + aT3)', 0, 0],

    [0, 0, 0, 0, 0, 'k33', 0, 0],
    ['aE1', 'aT1', 'aE2', 'aT2', 'aE3', 'aT3', 0, 0]]

    For
    """

    matrix = []

    # XXX testing! Write out the above matrix. Make mock values
    #k1 = ['k1'+str(i) for i in range(1,4)]
    #k2 = ['k2'+str(i) for i in range(1,4)]
    #k3 = ['k3'+str(i) for i in range(1,4)]
    #aE = ['aE'+str(i) for i in range(1,4)]
    #aT = ['aT'+str(i) for i in range(1,4)]

    # make the first rows since it is different. Also make the second row; it is
    # not different, but since you do two rows at a time it is convenient
    pre1 =  [-(k1[0] + aE[0]), k2[0]]         + [0 for _ in range(state_nr-2)]
    post1 = [k1[0], -(k2[0] + k3[0] + aT[0])] + [0 for _ in range(state_nr-2)]

    # XXX for testing
    #pre1 =  ['-('+k1[0] + ' + ' + aE[0]+')', k2[0]]         + [0 for _ in range(state_nr-2)]
    #post1 = [k1[0], '-('+k2[0] + ' + ' + k3[0] + ' + ' + aT[0] +')'] + [0 for _ in range(state_nr-2)]

    matrix.append(pre1)
    matrix.append(post1)

    # there are len(k3)*2 post + pre states. You make post+ pre for each
    # iteration. Skip the first since you've already made it -> start with 1
    # instead of 0
    for pp_nr in range(1, len(k3)):

        # fill up the row for pre-translocated
        pre = []
        post = []

        # count row from 1 and up; easier arithmetic
        for col_nr in range(1,state_nr+1):

            # zero before post_pre nr and state_nr are the same
            if col_nr < pp_nr*2:
                pre.append(0)
                post.append(0)

            elif col_nr == pp_nr*2:
                pre.append(k3[pp_nr-1])
                post.append(0)

            elif col_nr == pp_nr*2+1:
                pre.append(-(k1[pp_nr] + aE[pp_nr]))
                #pre.append('-('+k1[pp_nr] + ' + ' + aE[pp_nr] + ')') # XXX
                post.append(k1[pp_nr])

            elif col_nr == pp_nr*2+2:
                pre.append(k2[pp_nr])
                post.append(-(k2[pp_nr] + k3[pp_nr] + aT[pp_nr]))
                #post.append('-('+k2[pp_nr] + ' + ' + k3[pp_nr] + ' + ' + aT[pp_nr] + ')')
                # XXX

            # zero after this
            else:
                pre.append(0)
                post.append(0)

        matrix.append(pre)
        matrix.append(post)

    #Add the two last rows
    fl_row = [0 for _ in range(state_nr-3)] + [k3[-1], 0, 0]
    matrix.append(fl_row)

    ab_row = []
    for (aE_val, aT_val) in zip(aE, aT):
        ab_row.append(aE_val)
        ab_row.append(aT_val)

    # two last zeros
    ab_row.append(0)
    ab_row.append(0)

    matrix.append(ab_row)

    # return a np.array 
    return np.array(matrix)


def keq_reader():
    """
    read the keq_data.csv file
    """

    infile = "Keq_data_and_PY.txt"

    its_info = {}

    #ITS_variant:DG163a
    #PY:3.4
    #0.662700068357

    inf_handle = open(infile, 'rb')

    line1 = inf_handle.next()

    its_name = line1.split(':')[1].rstrip()

    info = []

    # parse the file ...
    for line in inf_handle:
        if not line.startswith('ITS_variant'):
            if line.startswith('PY'):
                PY = line.split(':')[1].rstrip()
                info.append(PY)
            else:
                keqs = line.split()
                info.append(keqs)
        else:
            its_info[its_name] = info

            its_name = line.split(':')[1].rstrip()
            info = []

    pys = [float(g[0]) for g in its_info.values()]

    keq_sum = [sum([float(i) for i in g[1]]) for g in its_info.values()]

    print spearmanr(pys, keq_sum)

    return its_info


def main():
    ITSs = ReadAndFixData() # read raw data

    #abortive_initiation_fromwhere(ITSs)

    #compare_quantitations()

    #third_report_figures(ITSs)

    #debug()

    #genome_wide()
    #new_genome()
    #new_ladder(lizt)
    #new_scatter(lizt, ITSs)

    # XXX Hsu's PYnew experiments have higher PY values
    # Compare the ratio in the new to the old
    #ratio_issue(ITSs)

    paper_figures(ITSs)

    # write the its and py to your delta g output file. it's a horrible hack
    # that will come back to haunt you some day. you add its name and py.
    #outp_write(ITSs)
    #keq_outp_write(ITSs) #XXX equally horrible, a keq-write hack

    #selection_pressure(ITSs)

    # XXX the new ODE models
    #new_models(ITSs)
    # XXX model with abortive does not improve results, but run on a larger grid
    # to confirm.

    # XXX Generate candidate sequences! : )
    #candidate_its(ITSs, DNADNAfilter=False)

    # test method for writing already ordered sequences
    #write_test(ITSs)

    # XXX Check the DG400 series with the new model
    # You did this before; how did you do it?
    #DG400_verifier(ITSs)

    #s_shaped(ITSs)

    # old one
    #new_designer(lizt)

    # naive mini
    #naive_model(ITSs)

    # XXX parameter-correlations
    # Kew-RD, Keq-DD, RD-DD
    #variable_corr()

    # the naive exp-fitting models
    #fitting_models(lizt, ITSs)

    #frequency_change()
    # NOTE the poly(A) and poly(T) tracts could be regulatory elements of
    # transcriptional slippage; this has been shown for several promoters

    # RESULT a weak signal when filtering by greA
    #greA_filter()

    # Now plotting the damn expression colors on the PCA plots
    #my_pca()
    #hsu_pca(lizt)
    # NOTE it's not clear what I should do now. There is this odd TTT repeating
    # going on. You should relate to super_en. In super_en the AA and TT have
    # totally opposite meanings. TTTT and AAAA are both associated with lower
    # rpkm. TTT tracts have been found at promoter proximal regions before. It's
    # not clear what they do.

    # They found promoter proximal stalling at these promoters
    # tnaA, cspA, cspD, rplK, rpsA and rpsU as well as lacZ. How do these stand
    # out in the crowd? They dont

    # TODO you still need to define strong promoters and see if there's a
    # tradeoff there. For strong promoters I would expect a shift in mean energy
    # compared to the rest. UPDATE: promoter strength is hard to define: plus,
    # Hsu gave an example of a strong E coli promoter with very little abortive
    # transcript. She says it's because escape is not the rate limiting step for
    # this promoter, it's open bubble formation. The bubble is so strong (GCs
    # between -10 and +2) so that the bubble collapses easily.
    #new_promoter_strength()

    # You have ribosomal genes + DPKM values for e coli genes
    # Is there any covariance between DPKM and ribosomal gene for the energy
    # terms you've found? I suggest using only 1-promoter genes

    #new_AP(ITSs)
    # RESULT: variance of AP correlates with variance of energy
    # Does that mean that variance of AP correlates with PY? Yes, but
    # weakly and negatively negatively.

    # RESULTS energy[:msat]/msat gives .8 correlation


class Sequence(object):
    """
    Add a sequence; calculate energy parameters and nucleotide composition.
    Shuffle sequence and do the same.
    """

    # Hidden variables you only want to initiate once
    # nucs
    _nucs = ('G', 'A', 'T', 'C')

    # dinucs
    _dinucs = tuple(set([''.join([_n1, _n2]) for _n1 in _nucs for _n2 in _nucs]))

    # repeats
    _repeats = tuple((''.join([n for val in range(i)]) for n in _nucs
                     for i in range(3,6)))

    # how many times do you want to shuffle the original sequence?
    _shuffle_freq = 50

    def __init__(self, sequence, name='noname', shuffle=True):

        self.seq = sequence
        self.name = name
        self.ID = name

        # get the dinucleotide pairs
        self._dinucPairs = self._dinucer(sequence)

        # frequencies of nucleotides, dinucleotides, and repeats
        self.nuc_freq = self._lookup_frequency(self._nucs)

        self.dinuc_freq = self._lookup_frequency(self._dinucs,
                                                 provided_seq=self._dinucPairs)
        self.purine_count = sequence.count('A') + sequence.count('G')

        # how to solve the repeats? :S how to get all repeats??
        # RESULT repeats are well handled by str.count since it considers
        # non-overlapping repeats
        self.repeat_freq = self._lookup_frequency(self._repeats)

        # the four energy terms
        self.super_en = Ec.super_f(sequence)
        self.keq_en = Ec.Keq(sequence)
        self.RNADNA_en = Ec.RNA_DNAenergy(sequence)
        self.DNADNA_en = Ec.DNA_DNAenergy(sequence)

        if shuffle:
            # make 20 randomly shuffled versions of the sequence
            self._randomSeq = [''.join(random.sample(sequence, len(sequence))) for _
                             in range(self._shuffle_freq)]

            # shuffled frequencies: the average of the lookup frequency for each random sequence
            self.shuf_nuc_freq = self._dict_average(
                [self._lookup_frequency(self._nucs, provided_seq=rs)
                 for rs in self._randomSeq])

            # need to provide dinucletoide list instead of sequence string
            self.shuf_dinuc_freq = self._dict_average(
                [self._lookup_frequency(self._dinucs, provided_seq=self._dinucer(rs))
                 for rs in self._randomSeq])

            self.shuf_repeat_freq = self._dict_average(
                [self._lookup_frequency(self._repeats, provided_seq=rs)
                 for rs in self._randomSeq])

    def _dinucer(self, sequence):
        """
        Return dicnueltode pairs for the sequeence
        """
        self._sal = list(sequence)
        return [self._sal[cnt] + self._sal[cnt+1] for cnt in range(len(self._sal)-1)]

    def _lookup_frequency(self, lookup, provided_seq=False):
        """
        Count the occurences of lookup keys in either self.sequence or a
        provided sequence / list for counting from
        """

        if not provided_seq:
            return dict((din, self.seq.count(din)) for din in lookup)
        else:
            return dict((din, provided_seq.count(din)) for din in lookup)


    def _dict_average(self, in_dicts):
        """
        Return the average of all counts of keys in the incoming dicts
        """

        _adder = {}
        for indict in in_dicts:
            for k, v in indict.items():
                if k in _adder:
                    _adder[k] += v
                else:
                    _adder[k] = v

        return dict((k, v/float(self._shuffle_freq)) for k, v in _adder.items())


# Background

# Abortive iniitiaton, found everywhere 
# Previous attempts; correlation with purines; attempt to correlate with RNA-DNA
# and DNA-DNA values using a thermodynamic model. Here we show that the best
# correlators are kinetic parameters for pyrophosphorylation
# How does this change the angle?

# 1) Mechanism for ITS-dependent abortive initiation is not known.
# 2) It has been suggested that the RNA-DNA hybrid is not involved in abortive
# initiation. Others have suggested that the DNA-DNA hybrid could play a part.
# It is known that promoters have higher DNA-DNA energies than random DNA.
# 3) Previous model has used DNA-DNA and RNA-DNA parameters, inspired by other
# models of translation elongation.
# 4) We show that the di-nucleotide dependent pyrophosphatase correlates
# strongly with duration of abortive cycling. We also show that for the
# previously suggested DNA-DNA there is no correlation, in agreement with recent
# results. For the RNA-DNA parameter there is some correlation, but weaker than
# for pyrophosphatase. This must be discussed.

# What did you conclude about the sum? Maybe ignore them.

# This explains the ITS-sequence dependence of abortive cycling. A model is
# where abortion of transcription initiation happens stochastically; the more
# likely pausing occurs in early transcription, the more likely it is that
# transcription will be aborted.
# idea: put all +/- promoters in a matrix and calculate the di-nucleotide/
# RNA-DNA / DNA-DNA energies for these values.
# Include the same but for random positions in the genome

############ Small test to correlate rna-dna and dna-dna energies ############
#dna = Ec.NNDD
#rna = Ec.NNRD

#fix, ax = plt.subplots()
#dna = Ec.resistant_fraction
##dna = Ec.Keq_EC8_EC9
#rna = Ec.NNRD
##rna = Ec.resistant_fraction

#dna_en = []
#rna_en = []

#for term in rna.keys():
    #if term in dna and term in rna and term != 'Initiation':
        #dna_en.append(dna[term])
        #rna_en.append(rna[term])

#ax.scatter(dna_en, rna_en)
#ax.plot([-3, 0], [-3, 0])
#ax.show()
############ XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ############

if __name__ == '__main__':
    main()

# IDEA: If there is a selection pressure for RNA-DNA energy at the ITS, it
# should be visible in promoters all over. Check annotated ITS versus random
# sequence, and also against jumbled ITS sequence
# RESULT random 15-mers have higher RNA-DNA binding energy than promoter ITS.
# Naturally, the AT-count determines this. Is this entirely explained from the
# 5' folding energy? Or can we say that this is another contributing factor to
# having high AT content at the ITS of promoters? What if we look at promoters
# with long UTRS (+50), does the correlation then still hold?
# RESULT variance of the RNA-DNA energy increases for randomly jumbled sequence,
# both for ITS and control. What could be the cause of this? Maybe I need to use
# promoters where the UTR is at least XX nucleotides, because this value could
# be biased by codons, where the NT-order is highly non-random.
# TODO find long 5UTRs. It's a pain in the ass to find good annotated regions. I
# think you have to intersect the CDS in the gtf file with the transcription
# start sites from the sigma files. Then you'll get the 5'UTRs. Chose only those
# that are long. But that can wait.

# IDEA: compare the energies of Hsu's set with random sequences using a QQ plot.
# QQ plots are a graphical way of comparing two distributions. Another way would
# be to count the number of G, A, T, and C in Hsu's data. If random, they should
# each occur at 1/4 each. RESULT see HsuRandomTester()

# TODO Look at those two e coli genes regulated by slippage when U concentration
# is too high. They hypothesize that it is the RNA-DNA energy of the first 4-5
# nucleotides that determines if stuttering happens or not.

# TODO For paper. Include 'critique' of the kinetic model stuff of Kippel and
# the chinese. Essence: looking just at the kinetic parameters in a kinetic
# model do not predict well the PY values or the abortive probabilities.

# IDEA For paper: Calculate correlations for AP with DNA/DNA bonds as well. For
# yourself: plot the AP-RNA/DNA correlation scores method 2. There's something
# funny about them. RESULT: I don't trust the AP so much any more.

# IDEA: In SimpleCorr, also add expected energy after msat, (compare that to
# dividing by msat!!), and then look for correlations with increasing length of
# ITS (0,4), (0,5) to see if you explain more data with more information.
# RESULT: More DNA information -> more explanation, until the end. Have
# not yet compared with random, but the increase should be highly improbable.
# RESULT: have compared to random, and it occurs rarely. Assuming independent
# occurence of near-monotonic increase of probability and achieving a corr >
# maxcorr, I multiply these and find e-6 probability of having this result.

# IDEA: go back to sliding window analysis and note DOWN what you find!
# RESULT: The sliding window looks at maxlen intervals and for those intervals
# the correlation is worse than for full lengths.

# IDEA Use the 10: sequences for the correlation ladder too (a reverse ladder).
# Do the first 10 or the last 10 explain more????
# RESULT:
# First 10 explain more than last 10
# (0.33176412097819036, 0.039083941355087082) for 1-10
# (0.22631578947368422, 0.16592729588956373) for 10-20

# IDEA investigate 2007 promoters with +1 to +10 variation
# RESULT: strange correlation. High at first, then lower toward +10.

# TODO Since 50% of abortive transcripts come from 2nt products (hsu 2009), and
# the 2 first nt are identical, shouldn't there be something you can look at
# here? First you need a way of normalizing the phospholevels of the 2nt with
# respect to initiation frequency or something else. Then try to correlate with
# the first 3 nucleotides (or 4!) (because you can't with the first 2!) Or
# alternatively see if there is no variation in 2nt products when normalized by
# some factor. TODO really investigate this. What causes the difference in the
# 2-and 3nt bands? This could be crucial.
# RESULT: 2012 comment, the 2-band could just be a source of inaccuracy in the
# experiments

# TODO -10-like sequences in the exposed non-template
# strand? Might hamper expression. Sigma intervention!!
# QUESTION: why non-template (coding) strand? Can sigma rebind both strands?
# This should be clear from crystallographical images.

# The exact physics of the DNA bubble:
# Begins at -11,+2/3. Scrunching theory implies that DNA is pulled in, so that the
# DNA bubble remains open from -11 to however far it reaches before promoter
# escape. From the 'vehicle of transcription' Borukhov paper.
# Seems bubble is 1 nt before nucleotide addition (structure paper)

# RESULTS: GOOD news for RNA-DNA interaction, more correlations than expected
# from random? Close to. Need to do a proper statistical analysis though. BAD news for
# DNA-DNA interaction. No correlation found what so ever.

# RESULT: from combo+sliding window: RNA + DNA no correlation; RNA - DNA some
# correlation, but probably nothing better than random.

# RESULT When I summed up the minimal energy I found a 0.436 correlation
# coefficient for RNA alone and only 0.425 for RNA + DNA. This is a a good
# result I think :)

# RESULT: the 0.436 seems highly significant! :) using EstomatorOfWrongness() I
# obtained 0.0059 frequency of random results with 0.436 or higher! That means
# 0.006 which is about 6 per thousand

# RESULT: when looking at how correlation changes with increasing summed up sequence
# length, it seems that our data set has a non-random increase in correlation
# strength with increasing sequence length. 

# RESULT: although the bigger productive yield values have a higher standard
# deviation, there is no correlation between productive yield and relative
# standard deviation, which is a stochastic variable with average of about 30%
# of the mean. (From Workhouse.relativeStandardDeviation

# RESULT: N25 and N25/A1 are the two big outliers. N25/A1 always and N25 when
# correcting for msat. N25 with its 11nt msat messes up msat calculations.
# Still when doing +1 to +15, N25 is still an outlier.

# IDEA: Instead of making 8nt steps, why not calculate the (0,msat) value RNA/DNA
# bond. Maybe it's just as well.
# RESULT: Almost as good correlation as with the Fancy Way!! OMG Good result. In
# the paper, begin with this result, then go to the more advanced stuff.

# RESULT of data analysis: Rank correlations in data set
# PY and RPY:  0.98
# PY and RIF: -0.85
# PY and APR: -0.99
# PY and MSAT: 0.32
# PY and R   : 0.77

# Life lesson: when doing optimization or parameter estimation or curve fitting,
# normalize the data so the poor solvers don't explode...
