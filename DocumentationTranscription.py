""" Calculate correlations between RNA/DNA and DNA/DNA binding energies and PY. """

# NOTE this file is a simplified version of 'Transcription.py' to generate data for the Hsu-paper

# Python modules
from __future__ import division
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.interpolate
import operator
from matplotlib import rc
# My own modules
import Testfit
import Workhouse
import Filereader
import Energycalc
import Orderrank
from glob import glob

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

# Figure directory for rna_dna analysis
# The path to the directory the script is located in
here = os.path.dirname(os.path.realpath(__file__))
fig_dir1 = os.path.join(here, 'figures')
fig_dir2 = '/home/jorgsk/phdproject/The-Tome/my_papers/rna-dna-paper/figures'

fig_dirs = (fig_dir1, fig_dir2)

# Correlator correlates the DNA-RNA/DNA energies of the subsequences of the ITS
# sequences with the data points of those sequences. ran = 'yes' will generate
# random sequences instead of using the sequence from seqdata. kind can be RNA,
# DNA or combo, where combo combines RNA and DNA energy calculations. sumit =
# 'yes' will sum up the energy values for each ITS. sumit == 'no' wil do a
# sliding window analysis. The sliding window may be important for correlating
# with Abortive Probabilities.

def welchs_approximate_ttest(n1, mean1, sem1, n2, mean2, sem2, alpha):
    """
    Got this from the scipy mailinglist, guy named Angus
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

def Correlator(seqdata, ran='no', sumit='yes', msatyes='yes', kind='RNA', p=0.05, plot='no'):
    """ Investigate correlations between DNA/DNA, RNA/DNA, and productive yield
    (PY) values of ITS-data.

    ran == 'yes' to compare the PY values to random DNA instead of experiment DNA.

    kind == 'RNA', 'DNA', or 'combo', to analyze only RNA/DNA, DNA/DNA, or a
    combination of these factors.

    sumit == 'yes' to correlate with the sum of energies instead of absolute
    energies. For example, for RNA/DNA the energy term at 18 will be the sum of
    all energy terms up to 18, instead of the (10,18) energy term obtained with
    sumit=='no'.

    p is the threshold for reporting correlation corefficients.

    msatyes == 'yes': do not return 0 in enrows for RNA/DNA bonds that are
    unphysical according to msat. For example, 'N25' will have 0 for (10,18).
    Instead return the expected energy of a sequence of that length. """

    maxlen = 8  # max length of RNA/DNA hybrid
    # Interchange 'rows' and 'columns' in seqdata
    rowdata = [[row[val] for row in seqdata] for val in range(len(seqdata[0]))]
    # rowdata = [sequence name, sequence, PY, PYstd..]
    PYstdv = rowdata[3]
    msat = rowdata[-2]
    #
    # Calculate energies of subsets of the ITS-sequences. row[-2] is the msat;
    # NOTE encolsRNA adds Expected energy value after msat is reached!
    encolsRNA = [Energycalc.PhysicalRNA(row[1], row[-2], msatyes, maxlen) for row in seqdata]
    encolsDNA = [Energycalc.PhysicalDNA(row[1]) for row in seqdata]

    if ran == 'yes':
        ran = ITSgenerator_local(len(seqdata))
        # Adding msat to sequences to mimick the real data
        ran = [[ran[num], msat[num]] for num in range(len(ran))]
        encolsRNA = [Energycalc.PhysicalRNA(seqs[0], seqs[1], msatyes, maxlen) for seqs in ran]
        encolsDNA = [Energycalc.PhysicalDNA(seqs[0]) for seqs in ran]
    # Interchange 'rows' and 'columns' in encols and add DNA range after energies
    # enrows is the energy for each stretch [0,2], [0,3],...[11,20] for each
    # seq. At the end of each 'enrow' is the list [0,3] to show this.
    enrowsRNA = [[[row[val][0] for row in encolsRNA], row[val][1]] for val in
              range(len(encolsRNA[0]))]
    enrowsDNA = [[[row[val][0] for row in encolsDNA], row[val][1]] for val in
              range(len(encolsDNA[0]))]

    if kind == 'RNA':
        enrows = enrowsRNA
    elif kind == 'DNA':
        enrows = enrowsDNA
    elif kind == 'combo':
        #   (-11,2)+(0,1), (-11,3)+(0,2),..., (-11,9)+(0,8), (-11,10)+(1,9),...
        # = (0,13)+(0,1), (0,14)+(0,2),..., (0,20)+(0,8), (0,21)+(1,9)
        # Convert the energy values to numpy arrays to _add_ them together
        # Hippel science 1998
        enrows = []
        for indx in range(len(enrowsRNA)):
            zum = np.array(enrowsRNA[indx][0]) + np.array(enrowsDNA[indx][0])
            zumranges = [enrowsRNA[indx][1], enrowsDNA[indx][1]]
            enrows.append([zum.tolist(), zumranges])

    # Including only the relevant data in rowdata: PY(2) and MSAT(6).
    experidata = [rowdata[2]]
    plot_returns = [[], [], []]
    if sumit == 'yes':
        # Removing indices and summing energy terms for each ITS.
        enMatrix = np.array([row[0] for row in enrows])
        enrows = [[sum(enMatrix[:cnt]).tolist(), cnt] for cnt in range(1,20)]
        # TODO which to plot? -1,-2,... etc
        normalizer = (-1)*min(enrows[-1][0]) #divide by min energy
        # energy, PY, seqnames are for plots and error report.
        energy = np.array(enrows[-1][0])
        energy_norm = np.array(enrows[-1][0])/normalizer
        PY = experidata[0]
        plot_returns = [energy_norm, PY, PYstdv]
        if plot == 'yes':
            Testfit.Expfit(energy_norm, PY, PYstdv) #stdv is a "main-list"
    reslist = [] # where statistically significant correlations are stored
    for datapoint in range(len(experidata)):
        for subseqnr in range(len(enrows)):
            # Separating the energies from their indeces
            statistics = scipy.stats.spearmanr(experidata[datapoint], enrows[subseqnr][0])
            pval = statistics[1]
            if pval <= p:
                corr = statistics[0]
                reslist.append([corr, pval, datapoint, enrows[subseqnr][1]])

    return reslist, plot_returns

def CorrelatorPlotter(lizt, pline='yes'):
    """ Plot the results from Correlation(). Lists are from [0,3] to [0,20] real
    world"""
    # Full lists. Plotdata = [energy, PY, PYstdv]
    title = 'Correlation of energy terms with PY for a kinetic model'
    sumit = 'yes'
    msatyes = 'no'
    ran = 'no'
    RNAnoMsat, plotNo = Correlator(lizt, ran, sumit, msatyes, kind='RNA', p=1)
    # Plot Not taking Msat into account
    #Workhouse.StdPlotter(plotNo[0], plotNo[1], plotNo[2], 'RNA/DNA '
                         #'energy', 'Productive yield', title)
    msatyes = 'yes'
    RNAyesMsat, plotYes = Correlator(lizt, ran, sumit, msatyes, kind='RNA', p=1)
    # Plot taking Msat into account 
    #Workhouse.StdPlotter(plotYes[0], plotYes[1], plotYes[2], 'RNA/DNA '
                         #'energy', 'Productive yield', title)
    DNA, DNAplot = Correlator(lizt, ran, sumit, msatyes, kind='DNA', p=1)
    both, bothplot = Correlator(lizt, ran, sumit, msatyes, kind='combo',p=1)
    # Just the correlation coefficients (and the pvals of one of them for interp)
    RNAnoM = [row[0] for row in RNAnoMsat[1:]]
    RNAyesM = [row[0] for row in RNAyesMsat]

    DNA = [row[0] for row in DNA[1:]]
    both = [row[0] for row in both[1:]]

    xticklabels = [str(integer) for integer in range(3,21)]
    yticklabels = [str(integer) for integer in np.arange(0, 0.8, 0.1)]
    incrX = range(3,21)

    fig, ax = plt.subplots()

    ax.set_xticks(range(3,21))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Nucleotide from TTS", size=26)
    ax.set_ylabel("Correlation coefficient, $r$", size=26)

    ax.plot(incrX, RNAnoM,'b', label='RNA/DNA', linewidth=2)
    ax.plot(incrX, DNA,'g', label='DNA/DNA', linewidth=2)
    ax.plot(incrX, both,'c', label='RNA/DNA + DNA/DNA', linewidth=2)
    if pline == 'yes':
        pval = PvalDeterminer(RNAnoMsat)
        ax.axhline(y=pval, color='r', label='$p$ = 0.05 threshold', linewidth=2)
    ax.legend(loc='upper left')

    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.set_yticklabels(yticklabels)

    #ax.set_title(title, size=26)

    # awkward way of setting the tick sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)

    fig.set_figwidth(8)
    fig.set_figheight(9)

    ax.legend()
    debug()

    for fig_dir in fig_dirs:
        for formt in ['pdf', 'eps', 'png']:

            name = 'physical_ladder.' + formt
            odir = os.path.join(fig_dir, formt)

            if not os.path.isdir(odir):
                os.makedirs(odir)

            fig.savefig(os.path.join(odir, name), transparent=True, format=formt)

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
        seqs = ITSgenerator_local(nrseq)
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

            tempRNA = Energycalc.RNA_DNAenergy(rnaRan)
            tempDNA = Energycalc.PhysicalDNA(dnaRan)[-1][0]

            incrRNA[0].append(tempRNA)
            incrDNA[0].append(tempDNA)

            # you are beyond msat -> calculate average energy
            if top > msat[index]:
                diffLen = top-msat[index]
                #RNA
                baseEnRNA = Energycalc.RNA_DNAenergy(sequence[:int(msat[index])])
                diffEnRNA = Energycalc.RNA_DNAexpected(diffLen+1)
                incrRNA[1].append(baseEnRNA + diffEnRNA)

                #DNA
                baseEnDNA = Energycalc.DNA_DNAenergy(sequence[:int(msat[index])])
                diffEnDNA = Energycalc.DNA_DNAexpected(diffLen+1)
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
        RNADNA.append(scipy.stats.spearmanr(incrEnsRNADNA[index], PY))

    #RNA
    incrWithExp20RNA = []
    incrWithoExp20RNA = []
    for index in range(len(incrEnsRNA[0])):
        incrWithoExp20RNA.append(scipy.stats.spearmanr(incrEnsRNA[0][index], PY))
        incrWithExp20RNA.append(scipy.stats.spearmanr(incrEnsRNA[1][index], PY))

    #DNA
    incrWithExp20DNA = []
    incrWithoExp20DNA = []
    for index in range(len(incrEnsDNA[0])):
        incrWithoExp20DNA.append(scipy.stats.spearmanr(incrEnsDNA[0][index], PY))
        incrWithExp20DNA.append(scipy.stats.spearmanr(incrEnsDNA[1][index], PY))

    arne = [incrWithExp20RNA, incrWithExp20DNA, incrWithoExp20RNA, incrWithoExp20DNA]

    output = {}
    output['RNA_{0} msat-corrected'.format(maxlen)] = arne[0]
    output['DNA_{0} msat-corrected'.format(maxlen)] = arne[1]

    output['RNA_{0} uncorrected'.format(maxlen)] = arne[2]
    output['DNA_{0} uncorrected'.format(maxlen)] = arne[3]

    output['RNADNA_{0} uncorrected'.format(maxlen)] = RNADNA

    return output

def ladPlot(lizt, reverse='no', pline='yes'):
    """ Printing the probability ladder for the ITS data. When nt = 5 on the
    x-axis the correlation coefficient of the corresponding y-value is the
    correlation coefficient of the binding energies of the ITS[0:5] sequences with PY.
    nt = 20 is the full length ITS binding energy-PY correlation coefficient.  """
    maxlen = 20
    arne = SimpleCorr(lizt, rev=reverse, maxlen=maxlen)
    # The first element is the energy of the first 3 nucleotides
    start = 3
    end = maxlen+1
    incrX = range(start, end)

    WithoExp20RNA = arne['RNA_{0} uncorrected'.format(maxlen)]
    WithoExp20DNA = arne['DNA_{0} uncorrected'.format(maxlen)]
    WithoExp20RNADNA = arne['RNADNA_{0} uncorrected'.format(maxlen)]

    toplot1 = WithoExp20RNA #Always RNA smth
    toplot2 = WithoExp20DNA
    toplot3 = WithoExp20RNADNA

    corr1 = [tup[0] for tup in toplot1]
    corr2 = [tup[0] for tup in toplot2]
    corr3 = [tup[0] for tup in toplot3]

    # Making a figure the pythonic way
    if reverse == 'yes':
        ticklabels = [str(integer)+'-20' for integer in range(1,21-2)]

    xticklabels = [str(integer) for integer in range(3,21)]
    yticklabels = [str(integer) for integer in np.arange(0, 0.8, 0.1)]

    fig, ax = plt.subplots()

    ax.set_xticks(range(3,21))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Nucleotide from transcription start", size=26)
    ax.set_ylabel("Correlation coefficient, $r$", size=26)
    ax.plot(incrX, corr1, 'b', label='RNA-DNA', linewidth=2)
    ax.plot(incrX, corr2, 'g', label='DNA-DNA', linewidth=2)
    ax.plot(incrX, corr3, 'c', label='RNA-DNA + DNA-DNA', linewidth=2)

    if pline == 'yes':
        pval = PvalDeterminer(toplot1)
        ax.axhline(y=pval, color='r', label='p = 0.05 threshold', linewidth=2)

    ax.legend(loc='upper left')
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.set_yticklabels(yticklabels)

    # awkward way of setting the tick font sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)

    fig.set_figwidth(9)
    fig.set_figheight(10)

    for fig_dir in fig_dirs:
        for formt in ['pdf', 'eps', 'png']:
            name = 'simplified_ladder.' + formt

            odir = os.path.join(fig_dir, formt)

            if not os.path.isdir(odir):
                os.makedirs(odir)

            fig.savefig(os.path.join(odir, name), transparent=True, format=formt)

def ITSgenerator_local(nrseq):
    """Generate list of nrseq RNA random sequences """
    gatc = list('GATC')
    return ['AT'+ ''.join([random.choice(gatc) for dummy1 in range(18)]) for dummy2 in
            range(nrseq)]

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
        ranITS = ITSgenerator_local(nrseq)
        # Calculate the energy
        enRNA = [Energycalc.PhysicalRNA(seq, msat, msatyes) for seq in ranITS]
        RNAen = [[[row[val][0] for row in enRNA], row[val][1]] for val in
                  range(len(enRNA[0]))]
        corrz = [scipy.stats.spearmanr(enRNArow[0],py) for enRNArow in RNAen]
        onlyCorr = np.array([tup[0] for tup in corrz])
        # Rank the correlation ladder
        ranLa = Orderrank.Order(onlyCorr[:13])
        Zcore = sum(abs(ranLa-perf))
        bigr = [1 for value in onlyCorr if value >= maxcor]
        noter[0].append(Zcore)
        noter[1].append(sum(bigr))
    return noter

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
            sequence = ITSgenerator_local(1)[0][:upto]
            if rand == 'biased':
                seq_iterator = ITSgenerator.RanGen((0.15,0.36,0.29,0.19))
                sequence = seq_iterator.next()[:upto]
            purTable[nr] = sequence.count('G') + sequence.count('A')
            enTable[nr] = Energycalc.RNA_DNAenergy(sequence)
        cor, pval = scipy.stats.spearmanr(purTable, enTable)
        cor_vals.append(cor)
        p_vals.append(pval)

    pval_mean, pval_sigma = scipy.stats.norm.fit(p_vals)
    cor_mean, cor_sigma = scipy.stats.norm.fit(cor_vals)
    #return 'Mean (corr, pval): ', cor_mean, pval_mean, ' -- Sigma: ', cor_sigma, pval_sigma
    return 'Mean corr : ', cor_mean, ' -- Sigma: ', cor_sigma

def EstimatorOfWrongness(lizt):
    """ Find correlation coefficients of random sequences until each new random
    success changes probability of getting more than 0.436 at random with less
    than 5%. """
    Realdata = Correlator(lizt, kind='RNA', sumit='yes')
    realp = Realdata[0][1] # p-value of sum correlation coefficient of Hsu data
    randseq = []
    count = 0
    converging = False
    deviationold = 2 # high enough not to cause problems for the if test below 
    while not converging:
        count += 1 # add 1 to count for each in silico experiment
        v = Correlator(lizt, ran='yes', kind='RNA', sumit='yes', p=realp)
        if v != []:
            print v
            randseq.append(1)
            deviation = len(randseq)/count
            print deviation, count
            print abs((deviationold - deviation)/deviationold)
            # the above will append data if p-val lower or equal than realp of Hsu data
            if abs((deviationold - deviation)/deviationold) <= 0.01:
                converging = True
                # When the change from one mean to the next is less than 1%
            else: deviationold = deviation
    return count, deviation

def JustHowWrong(n, lizt):
    """ Calculate how good the experimental sequences are compared to random """
    HsuSeq = Correlator(lizt)
    hsulen = len(HsuSeq)
    randseq = []
    for dummy in range(n):
        randseq.append(Correlator(lizt, ran='yes'))
    randlens = []
    bigunz = []
    for row in randseq:
        randlens.append(len(row))
        if len(row) >= hsulen:
            bigunz.append(len(row))
    avrglen = sum(randlens)/n
    if len(bigunz)>0:
        bigfreq = len(bigunz), sum(bigunz)/len(bigunz)
        return n, avrglen, hsulen, bigfreq, randlens
    return n, avrglen, hsulen, [], [] # randlens

# TODO Should also add the correlation coefficient. How many exceed bla bla bla for
# example.
def PlotError(lizt, n=20):
    """ Plot how good the experimental sequences are compared to random """
    statz = JustHowWrong(n, lizt)
    lenz = statz[4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, _bins, _patches = ax.hist(lenz, 50, normed=1, facecolor='green', alpha=0.75)
    ax.set_xlabel('"Significant" sequences')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 60)
    return statz

def StripSet(awaynr, lizt, ITSs):
    """ Remove sets of sequences from the dataset, depending on awaynr """
    seqset = [[], ['N25/A1'],['N25/A1','N25'],['N25/A1','N25','N25anti'],['N25/A1','N25','N25anti','N25/A1anti']]
    strip_names = seqset[awaynr]
    allnames = [row[0] for row in lizt] # Getting the names to get their index
    popthese = sorted([allnames.index(na) for na in strip_names]) # Getting the index
    popthese = reversed(popthese) # reverse-sorted iterator for safe removal!
    # remove from lizt
    for popz in popthese:
        del(lizt[popz])
    # remove from ITSs
    newITSs = []
    for dummy in range(len(ITSs)):
        if ITSs[dummy].name in strip_names:
            continue
        else:
            newITSs.append(ITSs[dummy])
    return lizt, newITSs

class ITS(object):
    """ Storing the ITSs in a class for better handling. """
    def __init__(self, name, sequence, PY, PY_std):
        # Set the initial important data.
        self.name = name
        self.sequence = sequence
        self.PY = PY
        self.PY_std = PY_std
        self.rna_dna1_10 = Energycalc.RNA_DNAenergy(self.sequence[:10])
        self.rna_dna1_15 = Energycalc.RNA_DNAenergy(self.sequence[:15])
        self.rna_dna1_20 = Energycalc.RNA_DNAenergy(self.sequence)
        self.dna_dna1_10 = Energycalc.DNA_DNAenergy(self.sequence[:10])
        self.dna_dna1_15 = Energycalc.DNA_DNAenergy(self.sequence[:15])
        self.dna_dna1_20 = Energycalc.DNA_DNAenergy(self.sequence)
        # Redlisted sequences 
        self.redlist = []
        self.redlisted = False

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
    rho15, p15 = scipy.stats.spearmanr(purine_levs, rna_dna15)
    rho20, p20 = scipy.stats.spearmanr(purine_levs, rna_dna20)

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

def NewEnergyAnalyzer(ITSs):
    """ Analyze the range of energies in Hsu's sequences -- this is important
    for the next wave of experiments. You need to know at what energies the
    transition from low PY to high PY exists. You need to know this mainly for
    the 1_15 energies, but it will also be interesting to look at the 1_10
    energies. """
    PYs = []
    one_15s = []
    one_10s = []
    one_20s = []
    for itr in ITSs:
        PYs.append(itr.PY)
        one_15s.append(itr.rna_dna1_15)
        one_10s.append(itr.rna_dna1_10)
        one_20s.append(itr.rna_dna1_20)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title("PY vs 1 to 15 rna-dna energies", size=10)
    ax1.set_xlabel("1 to 15 energies", size=20)
    ax1.set_ylabel("PY", size=20)
    ax1.scatter(one_15s, PYs)
    ax2 = fig.add_subplot(212)
    ax2.set_title("PY vs 1 to 10 rna-dna energies", size=10)
    ax2.set_xlabel("1 to 10 energies", size=20)
    ax2.set_ylabel("PY", size=20)
    ax2.scatter(one_10s, PYs)
    # RESULT the 1 to 15 energies go from -19 to -8 with the peak PY occuring
    # around -12. Ideally your energies should go from -18 to -8 in 1 to 15
    # the 1 to 10 energies go from -12 (one at -14) to -4

def PurineLadder(ITSs):
    """ Create a ladder of correlations between purine vs PY and energy."""
    pur_ladd_corr = []
    # calculate the purine ladders
    for itr in ITSs:
        seq = itr.sequence
        pur_ladder = [seq[:stop].count('A') + seq[:stop].count('G') for stop in
               range(2,20)]
        a_ladder = [seq[:stop].count('G') for stop in range(2,20)]
        energies = [Energycalc.RNA_DNAenergy(seq[:stp]) for stp in range(2,20)]
        itr.energy_ladder = energies
        itr.pur_ladder = pur_ladder
        itr.a_ladder = a_ladder
    # put the ladders in an array so that each column can be obtained
    pur_ladd = np.array([itr.pur_ladder for itr in ITSs])
    a_ladd = np.array([itr.a_ladder for itr in ITSs])
    PYs = np.array([itr.PY for itr in ITSs])
    energies = np.array([itr.energy_ladder for itr in ITSs])
    pur_corr = [scipy.stats.spearmanr(pur_ladd[:,row], PYs) for row in range(18)]
    a_corr = [scipy.stats.spearmanr(a_ladd[:,row], PYs) for row in range(18)]
    en_corr = [scipy.stats.spearmanr(a_ladd[:,row], energies[:,row]) for row in range(18)]
    # The correlation between purines and the Hsu R-D energies (quoted as 0.457
    # for up to 20)
    pur_en_corr = [scipy.stats.spearmanr(pur_ladd[:,row], energies[:,row]) for row in range(18)]

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
    en15s = [Energycalc.RNA_DNAenergy(itr.sequence[:15]) for itr in ITSs]
    pur15s = [itr.sequence[:15].count('A') + itr.sequence[:15].count('G') for itr in ITSs]
    fck_corr = scipy.stats.spearmanr(en15s, pur15s)

def ReadAndFixData():
    """ Read Hsu paper-data and Hsu normalized data. """

    # labels of Hsu data
    _labels = ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst','APR','APRst','MSAT','R']

    # Selecting the dataset you want to use
    #
    #lazt = Filereader.PYHsu(hsu1) # Unmodified Hsu data
    #lazt = Filereader.PYHsu(hsu2) # Normalized Hsu data by scaling
    lazt = Filereader.PYHsu(hsu3) # Normalized Hsu data by omitting experiment3
    #lazt = Filereader.PYHsu(hsu4) # 2007 data, 10 sequences

    # Selecting the columns I want from Hsu. Removing all st except PYst.
    # list=[Name 0, Sequence 1, PY 2, PYstd 3, RPY 4, RIFT 5, APR 6, MSAT 7, R 8]
    lizt = [[row[0], row[1], row[2], row[3], row[4], row[6], row[8], row[10],
             row[11]] for row in lazt]
    # Making a list of instances of the ITS class. Each instance is an itr.
    # Storing Name, sequence, and PY.
    ITSs = []
    for row in lizt:
        ITSs.append(ITS(row[0], row[1], row[2], row[3]))

    return lizt, ITSs

def RandEnCheck(nr=1000):
    ranens = []
    for dummy in range(nr):
        ranseq = ITSgenerator_local(1)[0][:9]
        ranens.append(Energycalc.RNA_DNAenergy(ranseq))
    meanz = np.mean(ranens)
    return meanz

def ScatterPlotter(lizt, ITSs, model='physical', stds='no'):
    """ Scatterplots for the paper. TODO make everything thicker and bigger so
    small versions of the figure will look good. """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("RNA-DNA energy ($\Delta G$)", size=28)
    ax.set_ylabel("Productive yield ", size=28)

    if model == 'physical':
        RD, RDplot = Correlator(lizt, ran='no', sumit='yes', msatyes='yes', kind='RNA', p=1)
        one_20s = RDplot[0] # energy is normalized for the physical model
        PYs = RDplot[1]
        PYs_std = RDplot[2]
        name = 'physical_scatter_PY_vs_RNADNA'

    if model == 'simplified':
        one_20s = [itr.rna_dna1_20 for itr in ITSs] # simply the 1-20 energy
        PYs = [itr.PY for itr in ITSs]
        PYs_std = [itr.PY_std for itr in ITSs]
        name = 'simplified_scatter_PY_vs_RNADNA'

    if stds == 'yes':
        ax.errorbar(one_20s,PYs,yerr=PYs_std,fmt='ro')
    else:
        ax.scatter(one_20s, PYs)

    # awkward way of setting the tick sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)

    if model == 'simplified':
        ax.set_xlim(-25.5,-11)
    if model == 'physical':
        ax.set_xlim(-1.02,-0.5)

    ax.set_ylim(0,11)
    fig.set_figwidth(9)
    fig.set_figheight(10)

    for fig_dir in fig_dirs:
        for formt in ['pdf', 'eps', 'png']:
            if stds == 'yes':
                fname = name + '_stds.' + formt
            else:
                fname = name + '.' + formt

            odir = os.path.join(fig_dir, formt)

            if not os.path.isdir(odir):
                os.makedirs(odir)

            fig.savefig(os.path.join(odir, fname), transparent=True, format=formt)

def PaperResults(lizt, ITSs):
    """Produce all the results that should be included in the paper."""
    # RESULT The science kudla paper concludes that DNA-DNA is the key :) Lol
    # Also use them in the introduction as they propose the stressed
    # intermediate that controls abortive initiation! Perfect.
    #
    # #################### Physical (or full) model RESULTS #########################
    #
    # D-D + R-D correlation
    #both, bothplot = Correlator(lizt, ran='no', sumit='yes', msatyes='yes', kind='combo', p=1)
    ## RESULT: correlation 0.39, p = 0.02 for both at 20, physical model.
    ## RESULT: correlation 0.29, p = 0.08 for both at 20, physical model with
    ## subtraction of forces instead of addition as is otherwise asumed.
    #RD, RDplot = Correlator(lizt, ran='no', sumit='yes', msatyes='yes', kind='RNA', p=1)
    ## RESULT: correlation 0.49 for, p = 0.002 just RNA at 20, physical model.
    #DD, DDplot = Correlator(lizt, ran='no', sumit='yes', msatyes='yes', kind='DNA', p=1)
    ## RESULT: correlation 0.28, p = 0.09 for just DNA at 20, physical model.
    #
    # Plots for the physical model
    #CorrelatorPlotter(lizt) # the ladder plot
    #ScatterPlotter(lizt, ITSs, model='physical', stds='yes') # the scatter plot
    #ScatterPlotter(lizt, ITSs, model='physical', stds='no') # the scatter plot
    #
    # #################### Simplifided model RESULTS #########################
    #
    #simple_correlation = SimpleCorr(lizt, rev='no', maxlen=20)

    #WithExp20RNA = arne[0]
    ## RESULT correlation 0.52, p = 0.0007 at 20, RNA w/Exp en, simplified model.
    #WithExp20DNA = arne[1]
    ## RESULT correlation 0.13, p = 0.43 at 20, DNA w/Exp en, simplified model.
    #WithoExp20RNA = arne[2]
    ## RESULT correlation 0.50, p = 0.001 at 20, RNA w/o/Exp en, simplified model.
    #WithoExp20DNA = arne[3]
    ## RESULT correlation 0.20, p = 0.21 at 20, RNA w/o/Exp en, simplified model.

    # Scatter plot of 1-20 RNA-DNA and DNA-DNA
    #simple_corr_scatter(ITSs, stds='yes')
    simple_corr_scatter(ITSs, stds='no')
    # Plots for the simplified model

    #ladPlot(lizt) # ladder plot simplified

    #ScatterPlotter(lizt, ITSs, model='simplified', stds='yes') # the scatter plot

def simple_corr_scatter(ITSs, stds='yes'):
    """
    Contrast the scatterplots of the RNA-DNA energy with the DNA-DNA energy
    """

    name = '1_20_scatter_comparison_RD_DD'

    fig, axes = plt.subplots(1, 2, sharey=True)

    rna_one_20s = [its.rna_dna1_20 for its in ITSs] # simply the 1-20 energy
    dna_one_20s = [its.dna_dna1_20 for its in ITSs] # simply the 1-20 energy

    energies = [rna_one_20s, dna_one_20s]

    PYs = [itr.PY for itr in ITSs]
    PYs_std = [itr.PY_std for itr in ITSs]

    # attempting log-transform
    #PYs = [np.log2(val) for val in PYs]
    #PYs_std = [np.log2(val) for val in PYs_std]

    fmts = ['ro', 'go']

    for ax_nr, ax in enumerate(axes):
        this_en = energies[ax_nr]
        if stds == 'yes':
            ax.errorbar(this_en, PYs, yerr=PYs_std, fmt=fmts[ax_nr])
        else:
            ax.scatter(this_en, PYs, color=fmts[ax_nr][:1])

        corrs = scipy.stats.spearmanr(this_en, PYs)

        header = 'r = {0:.2f}, p = {1:.3f}'.format(corrs[0], corrs[1])
        ax.set_title(header, size=25)

        if ax_nr == 0:
            ax.set_ylabel("Productive yield ", size=23)
            ax.set_xlabel("RNA-DNA energy ($\Delta G$)", size=23)
        else:
            ax.set_xlabel("DNA-DNA energy ($\Delta G$)", size=23)

        # awkward way of setting the tick sizes
        for l in ax.get_xticklabels():
            l.set_fontsize(18)
        for l in ax.get_yticklabels():
            l.set_fontsize(18)

    fig.set_figwidth(20)
    fig.set_figheight(10)

    for formt in ['pdf', 'eps', 'png']:

        if stds == 'yes':
            fname = name + '_stds.' + formt
        else:
            fname = name + '.' + formt

        for fig_dir in fig_dirs:
            odir = os.path.join(fig_dir, formt)

            if not os.path.isdir(odir):
                os.makedirs(odir)

            fpath = os.path.join(odir, fname)

            fig.savefig(fpath, transparent=True, format=formt)


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
            energies.append(Energycalc.RNA_DNAenergy(its))
            all_energies.append(Energycalc.RNA_DNAenergy(its))

            jumbl_energies.append(Energycalc.RNA_DNAenergy(jumbl_its))

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

        randenergies.append(Energycalc.RNA_DNAenergy(randseq))

        jumbl_rs = ''.join([randseq[np.random.randint(clen)] for v in range(clen)])
        jumbl_randenergies.append(Energycalc.RNA_DNAenergy(jumbl_rs))

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

def dimer_calc(energy_function, seq):
    """
    Use the dictionary 'energy_function' to cacluate the value of seq
    """
    def repl(n):
        if n == 'T':
            return 'U'
        return n

    indiv = [repl(l) for l in list(seq)] # splitting sequence into individual letters

    # getting dinucleotides
    neigh = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]
    energy = sum([energy_function[nei] for nei in neigh])

    return energy

def NewSimpleCorr(seqdata, energy_function, maxlen=20):
    """Calculate the correlation between RNA-DNA and DNA-DNA energies with PY
    for incremental positions of the correlation window (0:3) to (0:20). Two
    versions are returned: one where sequence-average expected energies are
    added after msat and one where nothing is done for msat.
    The rev='yes' option only gives meaningful result for
    incremental without adding expected values."""

    rowdata = [[row[val] for row in seqdata] for val in range(len(seqdata[0]))]
    seqs = rowdata[1]

    PY = rowdata[2]

    msat = rowdata[-2]
    # Calculating incremental energies from 3 to 20 with and without expected
    # energies added after msat in incr[1]. incr[0] has incremental energies
    # without adding expected energies after msat. NOTE E(4nt)+E(5nt)=E(10nt)
    # causes diffLen+1

    incrEnsRNA = [[], []] # 0 is withOut exp, 1 is With exp
    start = 3 #nan for start 0,1, and 2. start =3 -> first is seq[0:3] (0,1,2)
    for index, sequence in enumerate(seqs):
        incrRNA = [[], []]

        for top in range(start, maxlen+1):
            # Setting the subsequences from which energy should be calculated
            rnaRan = sequence[0:top]

            tempRNA = dimer_calc(energy_function, rnaRan)

            incrRNA[0].append(tempRNA)

            # you are beyond msat -> calculate average energy
            if top > msat[index]:
                diffLen = top-msat[index]
                #RNA
                baseEnRNA = dimer_calc(energy_function, sequence[:int(msat[index])])
                diffEnRNA = Energycalc.RNA_DNAexpected(diffLen+1)
                incrRNA[1].append(baseEnRNA + diffEnRNA)

            else:
                incrRNA[1].append(tempRNA)
        #RNA
        incrEnsRNA[0].append(incrRNA[0])
        incrEnsRNA[1].append(incrRNA[1])

    #RNA
    incrEnsRNA[0] = np.array(incrEnsRNA[0]).transpose() #transposing
    incrEnsRNA[1] = np.array(incrEnsRNA[1]).transpose() #transposing

    # Calculating the different statistics

    #RNA
    incrWithExp20RNA = []
    incrWithoExp20RNA = []
    for index in range(len(incrEnsRNA[0])):
        incrWithoExp20RNA.append(scipy.stats.spearmanr(incrEnsRNA[0][index], PY))
        incrWithExp20RNA.append(scipy.stats.spearmanr(incrEnsRNA[1][index], PY))

    arne = [incrWithExp20RNA, incrWithoExp20RNA]

    output = {}
    output['RNA_{0} msat-corrected'.format(maxlen)] = arne[0]
    output['RNA_{0} uncorrected'.format(maxlen)] = arne[1]

    return output

def New_Correlatvalues(lizt, ITSs):
    """
    New values for dinucleotide pyrophosphate addition have been published. How
    do these data correlate with your values?

    For easiest correlation, make the ladder.
    """

    """ Printing the probability ladder for the ITS data. When nt = 5 on the
    x-axis the correlation coefficient of the corresponding y-value is the
    correlation coefficient of the binding energies of the ITS[0:5] sequences with PY.
    nt = 20 is the full length ITS binding energy-PY correlation coefficient.  """
    maxlen = 20
    pline = 'yes'

    # use all energy functions from new article
    from dinucleotide_values import resistant_fraction, k1, kminus1, Keq_EC8_EC9

    name2func = [('r_f', resistant_fraction), ('k1', k1), ('k1_mins', kminus1),
                 ('Keq', Keq_EC8_EC9)]
    name2func = [('r_f', resistant_fraction), ('k1', k1), ('k1_mins', kminus1),
                 ('Keq', Keq_EC8_EC9)]

    fig, ax = plt.subplots()
    for name, energy_func in name2func:
        arne = NewSimpleCorr(lizt, energy_func, maxlen=maxlen)
        # The first element is the energy of the first 3 nucleotides
        start = 3
        end = maxlen+1
        incrX = range(start, end)

        toplot1 = arne['RNA_{0} uncorrected'.format(maxlen)]
        corr1 = [tup[0] for tup in toplot1]

        ax.plot(incrX, corr1, label=name, linewidth=2)

    xticklabels = [str(integer) for integer in range(3,21)]
    yticklabels = [str(integer) for integer in np.arange(-1, 1, 0.1)]

    ax.set_xticks(range(3,21))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Nucleotide from transcription start", size=26)
    ax.set_ylabel("Correlation coefficient, $r$", size=26)

    if pline == 'yes':
        pval = PvalDeterminer(toplot1)
        ax.axhline(y=pval, color='r', label='p = 0.05 threshold', linewidth=2)

    ax.legend(loc='upper left')
    ax.set_yticks(np.arange(-1, 1, 0.1))
    ax.set_yticklabels(yticklabels)

    # awkward way of setting the tick font sizes
    for l in ax.get_xticklabels():
        l.set_fontsize(18)
    for l in ax.get_yticklabels():
        l.set_fontsize(18)

    fig.set_figwidth(9)
    fig.set_figheight(10)

    plt.show()

    debug()
    #for fig_dir in fig_dirs:
        #for formt in ['pdf', 'eps', 'png']:
            #name = 'simplified_ladder.' + formt

            #odir = os.path.join(fig_dir, formt)

            #if not os.path.isdir(odir):
                #os.makedirs(odir)

            #fig.savefig(os.path.join(odir, name), transparent=True, format=formt)

def main():
    lizt, ITSs = ReadAndFixData() # read raw data
    #lizt, ITSs = StripSet(2, lizt, ITSs) # strip promoters (0 for nostrip)
    #PaperResults(lizt, ITSs)

    #NewEnergyAnalyzer(ITSs) # going back to see the energy-values again
    #RedlistInvestigator(ITSs)
    #RedlistPrinter(ITSs)
    #HsuRandomTester(ITSs)
    #PurineLadder(ITSs)

    #genome_wide()

    New_Correlatvalues(lizt, ITSs)


############ Small test to correlate rna-dna and dna-dna energies ############
#dna = Energycalc.NNDD
#rna = Energycalc.NNRD

#dna_en = []
#rna_en = []

#for term in rna.keys():
    #if term in dna and term in rna and term != 'Initiation':
        #dna_en.append(dna[term])
        #rna_en.append(rna[term])

#plt.scatter(dna_en, rna_en)
#plt.plot([-3, 0], [-3, 0])
#plt.show()
#debug()
############ XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ############

if __name__ == '__main__':
    main()
    #lizt, ITSs = main()

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

# TODO test make curves for upper and lower bound. Perhaps try with other
# functions than exponential? Polynomial for example. Also: return the errors!

# TODO LadderScrutinizer: finds the probability that the almost-uniformly
# increasing pattern in the correlation coefficients is accidental. Calculates
# new sequences are runs correlation analysis on those of equal length as the
# one from the sequences of the data set. Of those, the function counts how many
# have a similarly significant increase in correlation values and divides that
# by the number of such sets with equal length (10 with p = 0.05).

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
