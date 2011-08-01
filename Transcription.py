""" Calculate correlations between RNA/DNA and DNA/DNA binding energies and PY. """

# Python modules
from __future__ import division
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.interpolate
import operator
from Bio.Seq import Seq
from matplotlib import rc
# My own modules
import Testfit
import Workhouse
import Filereader
import Energycalc
import Orderrank

rc('text', usetex=True)  # Using latex in labels in plot
rc('font', family='serif')  # Setting font family in plot text

# Locations of input data
biotech = '/Rahmi/full_sequences_standardModified'
hsu1 = '/Hsu/csvHsu'
hsu2 = '/Hsu/csvHsuNewPY'
hsu3 = '/Hsu/csvHsuOmit3'
hsu4 = '/Hsu/csvHsu2008'

# Correlator correlates the DNA-RNA/DNA energies of the subsequences of the ITR
# sequences with the data points of those sequences. ran = 'yes' will generate
# random sequences instead of using the sequence from seqdata. kind can be RNA,
# DNA or combo, where combo combines RNA and DNA energy calculations. sumit =
# 'yes' will sum up the energy values for each ITS. sumit == 'no' wil do a
# sliding window analysis. The sliding window may be important for correlating
# with Abortive Probabilities.

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

    maxlen = 9  # max length of RNA/DNA hybrid
    # Interchange 'rows' and 'columns' in seqdata
    rowdata = [[row[val] for row in seqdata] for val in range(len(seqdata[0]))]
    # rowdata = [sequence name, sequence, PY, PYstd..]
    PYstdv = rowdata[3]
    #
    # Calculate energies of subsets of the ITR-sequences. row[-2] is the msat;
    # NOTE encolsRNA adds Expected energy value after msat is reached!
    encolsRNA = [Energycalc.PhysicalRNA(row[1], row[-2], msatyes, maxlen) for row in seqdata]
    encolsDNA = [Energycalc.PhysicalDNA(row[1]) for row in seqdata]
    if ran == 'yes':
        ran = ITRgenerator(len(seqdata))
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
        energy = np.array(enrows[-1][0])/normalizer
        PY = experidata[0]
        plot_returns = [energy, PY, PYstdv]
        if plot == 'yes':
            Testfit.Expfit(energy, PY, PYstdv) #stdv is a "main-list"
    reslist = [] # where statistically significant correlations are stored
    for datapoint in range(len(experidata)):
        for subseqnr in range(len(enrows)):
            # Separating the energies from their indeces
            statistics = Workhouse.Spearman(experidata[datapoint], enrows[subseqnr][0])
            pval = statistics[1]
            if pval <= p:
                corr = statistics[0]
                reslist.append([corr, pval, datapoint, enrows[subseqnr][1]])

    return reslist, plot_returns

def CorrelatorPlotter(lizt, pline='yes'):
    """ Plot the results from Correlation(). Lists are from [0,3] to [0,20] real
    world"""
    # Full lists. Plotdata = [energy, PY, PYstdv]
    title = 'Correlation between sum of RNA-DNA bonds from 0-3 to 0'
    sumit = 'yes'
    msatyes = 'no'
    ran = 'no'
    RNAnoMsat, plotNo = Correlator(lizt, ran, sumit, msatyes, kind='RNA', p=1)
    # Plot Not taking Msat into account
    title = 'Not taking MSAT into account'
    #Workhouse.StdPlotter(plotNo[0], plotNo[1], plotNo[2], 'RNA/DNA '
                         #'energy', 'Productive yield', title)
    msatyes = 'yes'
    RNAyesMsat, plotYes = Correlator(lizt, ran, sumit, msatyes, kind='RNA', p=1)
    # Plot taking Msat into account 
    title = 'Taking MSAT into account'
    #Workhouse.StdPlotter(plotYes[0], plotYes[1], plotYes[2], 'RNA/DNA '
                         #'energy', 'Productive yield', title)
    DNA, DNAplot = Correlator(lizt, ran, sumit, msatyes, kind='DNA', p=1)
    both, bothplot = Correlator(lizt, ran, sumit, msatyes, kind='combo',p=1)
    # Just the correlation coefficients (and the pvals of one of them for interp)
    RNAnoM = [row[0] for row in RNAnoMsat]
    RNAyesM = [row[0] for row in RNAyesMsat]
    DNA = [row[0] for row in DNA]
    both = [row[0] for row in both]
    # Making a figure the pythonic way
    ticklabels = ['1-'+str(integer) for integer in range(3,21)]
    incrX = range(3,21)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(range(3,21))
    ax.set_xticklabels(ticklabels)
    ax.set_title('Correlation between RNA/DNA, DNA/DNA and RNA/DNA+DNA/DNA '
                 'binding energies and productive yield where energies are '
                 'sums of physically possible molecule bindings.', size=10)
    ax.set_xlabel("ITR nucleotide range for summation of energies ", size=20)
    ax.set_ylabel("Correlation coefficient", size=20)
    ax.plot(incrX, RNAnoM, incrX, DNA, 'c', incrX, both)
    if pline == 'yes':
        pval = PvalDeterminer(RNAnoMsat)
        ax.axhline(y=pval, color='r')
        ax.legend(('RNA/DNA', 'DNA/DNA' , 'RNA/DNA + DNA/DNA', 'p = 0.05 threshold'), loc='upper left')
    else:
        ax.legend(('RNA/DNA', 'DNA/DNA', 'RNA/DNA + DNA/DNA'), loc='upper left')
    ax.set_yticks(np.arange(0.05, 0.9, 0.1))
    plt.show()

def SimpleCorr(seqdata, ran='no', rev='no'):
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
        seqs = ITRgenerator(nrseq)
#    labels = ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst','APR','MSAT','R']
    PY = rowdata[2]
#    PY = rowdata[8] # the correlation between RNA/DNA and R must be commented
#    upon.
    msat = rowdata[-2]
    # Calculating incremental energies from 3 to 20 with and without expected
    # energies added after msat in incr[1]. incr[0] has incremental energies
    # without adding expected energies after msat. NOTE E(4nt)+E(5nt)=E(10nt)
    # causes diffLen+1
    incrEnsRNA = [[],[]] # 0 is withOut exp, 1 is With exp
    incrEnsDNA = [[],[]]
    start = 3 #nan for start 0,1, and 2. start =3 -> first is seq[0:3] (0,1,2)
    for index, sequence in enumerate(seqs):
        incrRNA = [[],[]]
        incrDNA = [[],[]]
        for top in range(start,21):
            # Setting the subsequences from which energy should be calculated
            rnaRan = sequence[0:top]
            if rev == 'yes':
                rnaRan = sequence[top-start:]
            dnaRan = sequence[0:top+1]
            tempRNA = Energycalc.RNA_DNAenergy(rnaRan)
            tempDNA = Energycalc.PhysicalDNA(dnaRan)[-1][0]
            incrRNA[0].append(tempRNA)
            incrDNA[0].append(tempDNA)
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
    incrEnsDNA[1] = np.array(incrEnsDNA[1]).transpose() #transposing
    # Calculating the different statistics
    #RNA
    incrWithExp20RNA = []
    incrWithoExp20RNA = []
    for index in range(len(incrEnsRNA[0])):
        incrWithoExp20RNA.append(Workhouse.Spearman(incrEnsRNA[0][index], PY))
        incrWithExp20RNA.append(Workhouse.Spearman(incrEnsRNA[1][index], PY))
    #DNA
    incrWithExp20DNA = []
    incrWithoExp20DNA = []
    for index in range(len(incrEnsDNA[0])):
        incrWithoExp20DNA.append(Workhouse.Spearman(incrEnsDNA[0][index], PY))
        incrWithExp20DNA.append(Workhouse.Spearman(incrEnsDNA[1][index], PY))
    #
    arne = [incrWithExp20RNA, incrWithExp20DNA, incrWithoExp20RNA, incrWithoExp20DNA]

    return arne

def ladPlot(lizt, reverse='no', pline='yes'):
    """ Printing the probability ladder for the ITS data. When nt = 5 on the
    x-axis the correlation coefficient of the corresponding y-value is the
    correlation coefficient of the binding energies of the ITS[0:5] sequences with PY.
    nt = 20 is the full length ITS binding energy-PY correlation coefficient.  """
    arne = SimpleCorr(lizt, rev=reverse)
    # The first element is the energy of the first 3 nucleotides
    start = 3
    end = len(arne[0])+3
    incrX = range(start, end)
    WithExp20RNA = arne[0]
    WithExp20DNA = arne[1]
    WithoExp20RNA = arne[2]
    WithoExp20DNA = arne[3]
    toplot1 = WithoExp20RNA #Always RNA smth
    toplot2 = WithoExp20DNA
    corr1 = [tup[0] for tup in toplot1]
    corr2 = [tup[0] for tup in toplot2]
    # Making a figure the pythonic way
    ticklabels = ['1-'+str(integer) for integer in range(3,21)]
    if reverse == 'yes':
        ticklabels = [str(integer)+'-20' for integer in range(1,21-2)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(range(3,21))
    ax.set_xticklabels(ticklabels)
    ax.set_title("Correlation between RNA/DNA and DNA/DNA binding energies and productive"
                 " yield for different ITR ranges.", size=15)
    ax.set_xlabel("ITR nucleotide range for energy calculation", size=20)
    ax.set_ylabel("Correlation coefficient", size=20)
    ax.plot(incrX, corr1, incrX, corr2)
    if pline == 'yes':
        pval = PvalDeterminer(toplot1)
        ax.axhline(y=pval, color='r')
        ax.legend(('RNA/DNA', 'DNA/DNA' ,'p = 0.05 threshold'), loc='upper left')
    else:
        ax.legend(('RNA/DNA', 'DNA/DNA'), loc='upper left')
    ax.set_yticks(np.arange(0.05, 0.9, 0.1))
    plt.show()

def ITRgenerator(nrseq=1):
    """Generate list of nrseq RNA random sequences"""
    gatc = list('GATC')
    return ['AT'+ ''.join([random.choice(gatc) for dummy1 in range(18)]) for dummy2 in
            range(nrseq)]

def LadderScrutinizer(lizt, n=10):
    """ Calculate sets of size n of random sequences, perform RNA/DNA energy
    calculation for incremental ITR sublenghts, and evaluate the "monotonicity"
    of the correlation ladder; evaluation depends on number k of allowed
    deviations from monotonicity."""
    msatyes = 'no'
    rowdata = [[row[val] for row in lizt] for val in range(len(lizt[0]))]
    msat = rowdata[-2]
    py = rowdata[2]
    # The real correlation ladder
    arne = SimpleCorr(lizt)
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
        ranITR = ITRgenerator(nrseq)
        # Calculate the energy
        enRNA = [Energycalc.PhysicalRNA(seq, msat, msatyes) for seq in ranITR]
        RNAen = [[[row[val][0] for row in enRNA], row[val][1]] for val in
                  range(len(enRNA[0]))]
        corrz = [Workhouse.Spearman(enRNArow[0],py) for enRNArow in RNAen]
        onlyCorr = np.array([tup[0] for tup in corrz])
        # Rank the correlation ladder
        ranLa = Orderrank.Order(onlyCorr[:13])
        Zcore = sum(abs(ranLa-perf))
        bigr = [1 for value in onlyCorr if value >= maxcor]
        noter[0].append(Zcore)
        noter[1].append(sum(bigr))
    return noter

def strip_set(awaynr, lazt):
    """ Remove sets of sequences from the dataset, depending on awaynr """
    seqset = [[], ['N25/A1'],['N25/A1','N25'],['N25/A1','N25','N25anti'],['N25/A1','N25','N25anti','N25/A1anti']]
    names = seqset[awaynr]
    allnames = [row[0] for row in lazt] # Getting the names to get their index
    popthese = sorted([allnames.index(na) for na in names]) # Getting the index
    popthese = reversed(popthese) # reverse-sorted iterator for safe removal!
    for popz in popthese: del(lazt[popz])
    return lazt

def PvalDeterminer(toplot):
    """ Take a list of list of [Corr, pval] and return an interpolated Corr
    value that should correspond to pval = 0.05. """
    toplot = sorted(toplot, key=operator.itemgetter(1)) # increasing pvals like needed
    pvals = [tup[1] for tup in toplot]
    corrs = [tup[0] for tup in toplot]
    f = scipy.interpolate.interp1d(pvals, corrs)
    return f(0.05)

def Purine_RNADNA():
    # How to best find the correlation between Purine content of a DNA sequence
    # and its RNA/DNA bond energy? Correlation for all possible? :P
    # Correlation converges at 0.32
    # RESULT For 10000 runs of 43 r the standard deviation was 0.135. Thus the
    # 0.44 correlation in the 43 sequences in the paper is not significantly
    # different from 0.31. This suggests that the 0.77 correlation between
    # purines and PY in the paper hints at a mechanism different than RNA/DNA
    # bond!! NOTE hints at an experimental test. Sequences with
    # same RNA/DNA binding energy-ranges but different purine levels.
    ranNr = 39
    purTable = np.zeros(ranNr)
    enTable = np.zeros(ranNr)
    for nr in range(ranNr):
        sequence = ITRgenerator(1)[0]
        purTable[nr] = sequence.count('G') + sequence.count('A')
        enTable[nr] = Energycalc.RNA_DNAenergy(sequence)
    return Workhouse.Spearman(purTable, enTable)

def MeanAndStdof43():
    repnr = 100
    vals = []
    for dummy in range(repnr):
        corr, _pval = Purine_RNADNA()
        vals.append(corr)
    mean, sigma = scipy.stats.norm.fit(vals)
#    plt.hist(vals, 100)
    plt.show()
    # My the mean value theorem -> a normal distribution. Can thus calculate if
    # a mean = 0.48 is significantly different. Given mean = 0.32, sigma = 0.14
    # (calculated with n = 10000 repeats), we get z = (0.48-0.32)/0.14 = 0.86.
    # This means that the mean 0.43 is 0.86 standard deviations away from 0.31,
    # and thus not significantly different.
    return mean, sigma

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
    """ Plot how well the experimental sequences are compared to random """
    statz = JustHowWrong(n, lizt)
    lenz = statz[4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, _bins, _patches = ax.hist(lenz, 50, normed=1, facecolor='green', alpha=0.75)
    ax.set_xlabel('"Significant" sequences')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 60)
    plt.show()
    return statz

def StripSet(awaynr, lazt):
    """ Remove sets of sequences from the dataset, depending on awaynr """
    seqset = [[], ['N25/A1'],['N25/A1','N25'],['N25/A1','N25','N25anti'],['N25/A1','N25','N25anti','N25/A1anti']]
    names = seqset[awaynr]
    allnames = [row[0] for row in lazt] # Getting the names to get their index
    popthese = sorted([allnames.index(na) for na in names]) # Getting the index
    popthese = reversed(popthese) # reverse-sorted iterator for safe removal!
    for popz in popthese:
        del(lazt[popz])
    return lazt

class ITR(object):
    """ Storing the ITRs in a class for better handling. """
    def __init__(self, name, sequence, PY):
        # Set the initial important data.
        self.name = name
        self.sequence = sequence
        self.PY = PY
        # counting from nt nr. 3 as the first 2 are constant
        self.Gs = sequence[2:].count('G')
        self.As = sequence[2:].count('A')
        self.Ts = sequence[2:].count('T')
        self.Cs = sequence[2:].count('C')
        self.rna_dna1_10 = Energycalc.RNA_DNAenergy(self.sequence[:10])
        self.rna_dna1_15 = Energycalc.RNA_DNAenergy(self.sequence[:15])
        self.rna_dna1_20 = Energycalc.RNA_DNAenergy(self.sequence[:])
        # Redlisted sequences 
        self.redlist = []
        self.redlisted = False

def HistPlot(lizt):
    """ Investigating the normality of Hsu's data for comparing with the random
    ITRs. Conclusion from scatter between PY and energies: you need energies
    down to at least -15 if trying to reproduce data :S """
    seqs = []
    pys = []
    for row in lizt:
        energy = Energycalc.RNA_DNAenergy(row[1])
        seqs.append(energy)
        pys.append(row[2]) # py
#    plt.hist(seqs, 6)
    print sorted(seqs)
    print np.mean(seqs)
    print np.std(seqs)
    plt.scatter(seqs, pys)
    plt.show()

def RedlistInvestigator(ITRs):
    """ Read minus ten elements from Filereader and add redlisted sequence
    elements to the ITR class instances."""

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
    for itr in ITRs:

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
#    plt.show()

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

def RedlistPrinter(ITRs):
    """ Give a printed output of what ITRs have what redlisted sequence
    in them at which location. """
    # What I would like to see (sorted by PY value): Name of promoter, PY value, redlisted sequences
    # according to type with location in brackets.
    _header = '{0:10} {1:5} {2:15} {3:8} {4:10} {5:10}'.format('Promoter', 'PY value', 'Sequence', 'Redlisted '
                                                          'sequence', 'Location on ITR','Redlist type')
    #print(header)
    # Sort ITRs according to PY value
    ITRs.sort(key=operator.attrgetter('PY'), reverse=True)
    for itr in ITRs:
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

def HsuRandomTester(ITRs):
    gatc = {'G':[], 'A':[], 'T':[], 'C':[]}

    bitest = scipy.stats.binom_test # shortcut to function

    for itr in ITRs:
        gatc['G'].append(itr.Gs)
        gatc['A'].append(itr.As)
        gatc['T'].append(itr.Ts)
        gatc['C'].append(itr.Cs)

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

def NewEnergyAnalyzer(ITRs):
    """ Analyze the range of energies in Hsu's sequences -- this is important
    for the next wave of experiments. You need to know at what energies the
    transition from low PY to high PY exists. You need to know this mainly for
    the 1_15 energies, but it will also be interesting to look at the 1_10
    energies. """
    PYs = []
    one_15s = []
    one_10s = []
    for itr in ITRs:
        PYs.append(itr.PY)
        one_15s.append(itr.rna_dna1_15)
        one_10s.append(itr.rna_dna1_10)
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
    plt.show()
    # RESULT the 1 to 15 energies go from -19 to -8 with the peak PY occuring
    # around -12. Ideally your energies should go from -18 to -8 in 1 to 15
    # the 1 to 10 energies go from -12 (one at -14) to -4

def ReadAndFixData():
    """ Read Hsu paper-data and Hsu normalized data. """

    # labels of Hsu data
    _labels = ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst','APR','APRst','MSAT','R']

    # Read Rahmi's dataset (it comes as as a dictionary)
    RahmiR = Filereader.Rahmi104()
    Rahmi = []
    WT_1_to_29 = 'AACAUGUACAAUAAUAAUGGAGUCAUGAA'
    WT_19_to_29 = WT_1_to_29[18:] # Part of sequence for scrutiny
    for key, value in RahmiR.iteritems():
        name = key
        sequence = value['Sequence'][:20]
        sequence = Seq(sequence).back_transcribe().tostring() # RNA -> DNA
        induced = value['Induced']
        uninduced = value['Uninduced']
        # Optional filter to get only the most comparable sequences
        # Making sure it's as long as the Hsu dataset list. Msat=20 for all.
        if value['Sequence'][18:29] == WT_19_to_29:
            Rahmi.append([name, sequence, induced, uninduced, 1, 1, 1, 1, 1, 1, 20, 20])

    # Selecting the dataset you want to use
    #
    #lazt = Rahmi #  Rahmi's data
    #lazt = Filereader.PYHsu(hsu1) # Unmodified Hsu data
    #lazt = Filereader.PYHsu(hsu2) # Normalized Hsu data by scaling
    lazt = Filereader.PYHsu(hsu3) # Normalized Hsu data by omitting experiment3
    #lazt = Filereader.PYHsu(hsu4) # 2007 data, 10 sequences

    # Selecting the columns I want from Hsu. Removing all st except PYst.
    # list=[Name 0, Sequence 1, PY 2,  PYst 3, RPY 4, RIFT 5, APR 6,  MSAT 7   ,
    #         R 8]
    lizt = [[row[0], row[1], row[2], row[3], row[4], row[6], row[8], row[10],
             row[11]] for row in lazt]
    # Making a list of instances of the ITR class. Each instance is an itr.
    # Storing Name, sequence, and PY.
    ITRs = []
    for row in lizt:
        ITRs.append(ITR(row[0], row[1], row[2]))

    return lizt, ITRs

def Testrandom():
    table = []
    for dummy in range(1000):
        table.append(ITRgenerator(1))
    1/0

def Main():
    lizt, ITRs = ReadAndFixData()
    lizt = StripSet(4, lizt) # must uncomment when not working with the standard set
    #NewEnergyAnalyzer(ITRs) # going back to see the energy-values again
    #RedlistInvestigator(ITRs)
    #RedlistPrinter(ITRs)
    #HsuRandomTester(ITRs)
    #CorrelatorPlotter(lizt)
    #ladPlot(lizt) # the DNA-DNA and RNA-DNA separate
    #skrot = LadderScrutinizer(lizt)
    Testrandom()


Main()

# TODO create a method that outputs all the results you use in the paper?

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
# ITR (0,4), (0,5) to see if you explain more data with more information.
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

# IDEA Investigate Rahmi's set with the RNA-DNA energy sums! Also check the two
# sequences he sent to Hsu. IDEA See if the combination of the RNA-DNA values
# from your investigation with the relative initiation frequency values from the
# Hsalis program can explain the induced/uninduced values better!
# RESULT: Looking at all of Rahmi's dataset I see no correlation. However,
# looking at the subset that does not have mutations from SD and onwards, I find
# some correlation, but not a strong one. But it's going in the right direction!
# I think that the 5'UTR modification affect translation rate much more than
# transcription rate. Therefore, we need to isolate these two processes! :)

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
