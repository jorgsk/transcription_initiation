""" Calculate correlations between RNA/DNA and DNA/DNA binding energies and PY. """

# NOTE this file is a simplified version of 'Transcription.py' to generate data for the Hsu-paper

# Python modules
from __future__ import division
# You need future division even in numpy
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy import optimize
from scipy.stats import spearmanr
import scipy.interpolate
import operator
from matplotlib import rc
import itertools
# My own modules
import Workhouse
import Filereader
import Energycalc as Ec
import Models
from glob import glob

import transformations as transf

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
fig_dir2 = '/home/jorgsk/phdproject/The-Tome/my_papers/rna-dna-paper/figures'

fig_dirs = (fig_dir1, fig_dir2)

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

def ITSgenerator_local(nrseq):
    """Generate list of nrseq RNA random sequences """
    gatc = list('GATC')
    return ['AT'+ ''.join([random.choice(gatc) for dummy1 in range(18)]) for dummy2 in
            range(nrseq)]

def PvalDeterminer(toplot):
    """ Take a list of list of [Corr, pval] and return an interpolated Corr
    value that should correspond to pval = 0.05. """
    toplot = sorted(toplot, key=operator.itemgetter(1)) # increasing pvals like needed
    pvals = [tup[1] for tup in toplot]
    corrs = [tup[0] for tup in toplot]
    f = scipy.interpolate.interp1d(pvals, corrs)
    return f(0.05)

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
    def __init__(self, name, sequence, PY, PY_std, msat):
        # Set the initial important data.
        self.name = name
        self.sequence = sequence
        self.PY = PY
        self.PY_std = PY_std
        self.msat = int(msat)
        self.rna_dna1_10 = Ec.RNA_DNAenergy(self.sequence[:10])
        self.rna_dna1_15 = Ec.RNA_DNAenergy(self.sequence[:15])
        self.rna_dna1_20 = Ec.RNA_DNAenergy(self.sequence)
        self.dna_dna1_10 = Ec.DNA_DNAenergy(self.sequence[:10])
        self.dna_dna1_15 = Ec.DNA_DNAenergy(self.sequence[:15])
        self.dna_dna1_20 = Ec.DNA_DNAenergy(self.sequence)
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
    #_labels = ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst','APR','APRst','MSAT','R']

    # Selecting the dataset you want to use
    #
    lazt = Filereader.PYHsu(hsu1) # Unmodified Hsu data
    #lazt = Filereader.PYHsu(hsu2) # Normalized Hsu data by scaling
    #lazt = Filereader.PYHsu(hsu3) # Normalized Hsu data by omitting experiment3
    #lazt = Filereader.PYHsu(hsu4) # 2007 data, skipping N25, N25anti
    #lazt = Filereader.PYHsu(hsu5) # 2007 data, including N25, N25anti
    #biotech = '/Rahmi/full_sequences_standardModified'

    #lazt = Filereader.Rahmi104(adapt=True)
    # you should normalize these with the RBS calculator

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
        ITSs.append(ITS(row[0], row[1], row[2], row[3], row[7]))

    return lizt, ITSs

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

    from Energycalc import reKeq, NNRD, NNDD, super_en

    name2func = [('RNA-DNA', NNRD), ('DNA-DNA', NNDD), ('Translocation', reKeq),
                ('RNA-DNA - Translocation', super_en)]
    # The r_f, k1, and K_eq correlate (r_f positively and k1 and K_eq negatively

    plt.ion()
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
    ax.set_xlabel("Nucleotide from transcription start", size=26)
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

    plt.show()

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


def new_scatter(lizt, ITSs):
    """
    Scatter plots at 5, 10, 13, 15, and 20
    """

    #stds = 'no'
    stds = 'yes'

    plt.ion()

    #rows = [5, 10, 15, 20, 'msat']
    rows = [10, 15, 20]
    fig, axes = plt.subplots(len(rows), 3, sharey=True)

    for row_nr, maxnuc in enumerate(rows):
        name = '1_{0}_scatter_comparison'.format(maxnuc)

        if maxnuc == 'msat':
            #rna_dna = [Ec.RNA_DNAenergy(s.sequence[:s.msat])/float(s.msat) for s in ITSs]
            #dna_dna = [Ec.DNA_DNAenergy(s.sequence[:s.msat])/float(s.msat) for s in ITSs]
            #keq = [Ec.Keq(s.sequence[:s.msat])/float(s.msat) for s in ITSs]
            #added = [Ec.super_f(s.sequence[:s.msat])/float(s.msat) for s in ITSs]
            rna_dna = [Ec.RNA_DNAenergy(s.sequence[:s.msat]) for s in ITSs]
            dna_dna = [Ec.DNA_DNAenergy(s.sequence[:s.msat]) for s in ITSs]
            keq = [Ec.Keq(s.sequence[:s.msat]) for s in ITSs]
            added = [Ec.super_f(s.sequence[:s.msat]) for s in ITSs]
            #PCA = [Ec.pca_f(s.sequence[:s.msat])/float(s.msat) for s in ITSs]

        else:
            rna_dna = [Ec.RNA_DNAenergy(s.sequence[:maxnuc]) for s in ITSs]
            dna_dna = [Ec.DNA_DNAenergy(s.sequence[:maxnuc]) for s in ITSs]
            keq = [Ec.Keq(s.sequence[:maxnuc]) for s in ITSs]
            added = [Ec.super_f(s.sequence[:maxnuc]) for s in ITSs]
            #added = [rna_dna[i] - keq[i] for i in range(len(keq))]
            #PCA = [Ec.pca_f(s.sequence[:maxnuc]) for s in ITSs]

        energies = [('RNA-DNA', rna_dna), ('Translocation', keq),
                    ('RNA-DNA - Translocation', added)]
        #energies = [('Abortive propensity', added)]

        PYs = [itr.PY for itr in ITSs]
        PYs_std = [itr.PY_std for itr in ITSs]

        fmts = ['ro', 'go', 'bo']

        for col_nr, (name, data) in enumerate(energies):

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
                #ax.set_ylabel("PY ({0} nt of ITS)".format(maxnuc), size=15)
                ax.set_ylabel("Productive yield of ITS variant)".format(maxnuc), size=15)
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

            # remove me
            ax.set_xticklabels(range(len(data)))

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

def new_AP(lizt, ITSs):
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
    APs = []
    props = []
    pys = []

    for ITS in ITSs:
        ap = []
        prob = []

        indiv = list(ITS.sequence[:9])
        neigh = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]

        # not always agreement between ITS.msat and abortive probs
        abortz = [a for a in abortProbs[ITS.name] if a != '-']

        # zip them
        for dinuc, AP in zip(neigh, abortz):
            ap.append(AP)
            prob.append(Ec.super_en[dinuc])
            #prob.append(Ec.Keq_EC8_EC9[dinuc])

            # compare the standard deviations
        APs.append(np.std(ap))
        props.append(np.std(prob))
        pys.append(ITS.PY)

        #r, pval = spearmanr(ap, prob)

        #fig, ax = plt.subplots()
        #ax.bar(range(2, len(ap)+2), ap, color='b', alpha=0.7)
        #ax.bar(range(2, len(prob)+2), prob, color='r', alpha=0.7)
        #ax.set_title(ITS.name)

            #print ITS.name
            #print ap
        #print spearmanr(ap, prob)



    print spearmanr(APs, props)
    print spearmanr(APs, pys)
    debug()

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

    # Is there a way to quantify when 90% of the abortive transcription is over?
    # Maybe by the imagequant? Maybe you'll find a correlation. What would that
    # prove exactly? The msat? How? At each point there is a stochastic chance
    # of falling off. MSAT means that the active site has not fallen off yet. It
    # could be that MSAT is higher for those with weak PY? But then I should
    # have seen it. Or maybe not use MSAT but use 90% of MSAT. You might then
    # find that 
    # What do I expect, biochemically? I expect that MSAT is limited by an
    # energy barrier that RNAP is unlikely to pass while still attached to
    # promoter.
    # This suggests that it is not the "free energy" held up in scrunching, but
    # 
    # The model is that RNAP has at any point a chance of aborting, as long as
    # it's attached to SIGMA. Mayby thermodynamic fluctuations play a part,
    # making the process stochastic in nature. The determining factor then
    # becomes the ease at which RNAP synthisizes the first XX nucleotides. Then
    # sigma release is also stochastic. Sigma is pulling backward and RNAP is
    # pulling forward. The probability of each one winning is the ease at which
    # RNAP translocates the first 10-15 nucleotiedes. But pulling? RNAP is
    # standing still at the promoter. However, if the DNA-DNA buildup is not the
    # case, what is? I should re-read that article. That's for T7 though. What
    # is the stressor? Why escape? Is this important only for strong promoters?
    # If RNAP continues straight from initiation to escape, what is escape? Why
    # does sigma fall off? Or why does sigma dissasociate from promoter
    # contacts? That's the most relevant question.

    # They say that rnap moves by random fluctuations that occasionally becomes
    # blocked backwards by nucleotide incorporations. The free energy of nt
    # incorporation should also somehow be a driving force, even if it's not
    # well explained.

    # Design sequences which are both AT rich but have opposite effects on
    # transcription initiation, showing that it's not just the AT-richness, but
    # the specific order of the AT-richness
    # What about that -10 like element? Is that a particularly weak one? Would
    # be interesting if it were the weakest possible association of these
    # values. It's TATAAT
    #
    # The 10 pause site had more variation in RNA-DNA than in super_en, so I'm
    # ruling it out as a super_en effect.

    # RNAP is using force from nucleotide incorporation to propel itself?
    # Conclusion of 1998 Science.

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

    import re
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

    # XXX a quandary; Keq gives the best correlation but makes the least sense
    # Maybe the best solution is the differential equations with r =
    # r_1*exp(r2*DeltaG(RNA-DNA))

    #variables = (k1, rna_dna)
    #variables = (kMinus1, rna_dna)

    #### the exp model
    #parameters = (100, -0.1, 0.5)
    #variables = (keq, rna_dna)

    #plsq = optimize.leastsq(Models.residuals_mainM, parameters, args=(variables, PYs))
    #fitted_parm = plsq[0]
    #outp = Models.mainM(fitted_parm, variables)
    #fitted_vals = np.dot(fitted_parm[1:], variables)
    #c1 = fitted_parm[0]
    #plot_data = c1*np.exp(plot_range)

    #### the log(keq) model
    parameters = (100, 0.1, 0.1, 0.1)
    variables = (keq, rna_dna)

    plsq = optimize.leastsq(Models.residuals_logKeq, parameters, args=(variables, PYs))
    fitted_parm = plsq[0]
    outp = Models.logKeq(fitted_parm, variables)
    c1, b1, b2, b3 = fitted_parm
    fitted_vals = b1*rna_dna - b2*np.log(b3*keq)
    minval = min(fitted_vals)
    maxval = max(fitted_vals)
    plot_range = np.arange(minval, maxval, 0.01)
    plot_data = c1*np.exp(plot_range)
     #RESULT log(keq) model has better fit visually
     # It corresponds to a k1*Keq equilibrium constant
     # ACtualy .. the Keq^k1 fits best, and gives a k1 value of 1.97. Bening.
     # The k1*Keq gives a value of exp(28) ... not benign. Which model is more
     # likely? The exp model gives too high values for .. what about a
     # combination? A combination changes little for b3; it doesn't improve the
     # fit. A good sign.

    print fitted_parm
    print spearmanr(PYs, outp)
    print scipy.stats.pearsonr(PYs, outp)

    # How should I fit it? The predicted PY against the actual PY?
    # Or the actual PY against the -0.14 * keq + 0.15 * rna_dna?
    # Then plot the equation using a linear range from min to max

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

def new_models(lizt, ITSs):
    """
    Define a linear ODE system that solves the movement on RNAP on the ITS.
    Optimize the system based on 3 parameters that will quantify the effect of
    RNA-DNA and Keq translocation bias.

    Your original discovery is a linear relationship between Keq and RNA-DNA.
    You'll need to recreate this in the rate equations somehow.

    The system is dX = AX

    Where X are the pre and pos-translocated states of each dinucleotide
    position, and A is a matrix of rate-coefficients

    All right. Now you have 2 models. You will probably have 3-4 models in the
    end to choose from. You'll need an easy way to test different models.

    First, different models will produce different A-matrices (rate constants).
    Thus, each model must be associated with its A-matrix. The different rate
    constants will be modified in different ways by constants for fitting + the
    variables (RNA-DNA, DNA-DNA, Keq, k1, kMinus1)

    I should let each ITS object have the above variables calculated for each
    dinucleotdie pair. Then I don't need to re-calculate them, just use the
    indexes. Then you can write k1 = c1*exp(c2*rna_dna)/keq - c3*dna-dna
    for each rate constant as you please.

    Different models will have different number of rate constants which need
    different transformations on them. You can't generalize that. Each model
    should have its own method (comment out if not run). Each model should
    produce comparable outcomes in terms of plots and other measures.

    Equilibrum of the post-pre step reflects a time-scale difference. We say
    that this reaction is fast compared to nucleotide incorporation. Is it?
    Maybe.
    """

    # Plot the movement in what? A 3d box plot?
    #plt.ion()
    #fig, axes = plt.subplots(10)
    #for r_nr, row in enumerate(soln[::300]):
        #axes[r_nr].plot(row)

    #For now, just return the concentration at the last stage 

    PYs = [itr.PY for itr in ITSs]
    its_len = 12 # how far to transcribe

    first_nonequilibrium(PYs, its_len)

def first_nonequilibrium(PYs, its_len):

    # RESULT the 'naive' approach does not work (I had actually thought that it
    # would). Am I comparing against the right thing? I'm copmaring against the
    # % of value in state 20. I'll start doing it until 15 instead.
    # NOTE you are having equal sign between RNA-DNA and the other... maybe you
    # should reverse it? How do I do that without compromizing the values? :S
    # TODO
    # 1) Try to fit 3 parameters to the PY values # not much better
    # 2) Try to introduce changes in the model # can try ...

    # initial tunable values
    initial_values = (1,1,-1)
    print initial_values

    states = its_len -1 # the number of i positions
    variables = states*2 # two sub-states for each state: pre- and posttranslocated

    y0 = [1] + [0 for i in range(variables)]
    # XXX The final value in 'y0' is the end product

    t = np.linspace(0, 150., 1500) # time grid
    # The time units are of course arbitrary

    # these transformations will work on k1, k2, and k3, or any further k's you
    # may make
    transformations = transf.const, transf.const, transf.expon

    arguments = (y0, t, its_len, states, variables, ITSs, transformations, PYs)

    plsq = optimize.leastsq(cost_function, initial_values, args = arguments)

    # TODO check the effect of the different parameters; to what degree do they
    # change the order of the final vals?

    fitted_param = plsq[0]
    print fitted_param
    #fitted_param = [2.02,  3.26, -0.91]
    # XXX result; with these fitted parameters you don't get a good fit. This
    # means you'll have to update the model.
    # You'll need a set of models. Each will have different parameters. The more
    # parameters, the more fitting :/
    # You can also try other minimizers.
    # Try a model without RNA-DNA for example; just aconstant. I get a bit
    # better correlation 0.45, but it converges slow as hell.

    # make a file with different transformations of k1, k2, k3 etc
    # maybe I should split them up into the individual transformations so that I
    # don't create 10 different usage examples.
    # The last one can be split into first exp and then constant!
    # The values take 'array, value'

    # get the distribution of RNAP in the final state of the simulation
    finals = final_state(ITSs, y0, t, transformations, fitted_param, its_len)

    print spearmanr(PYs, finals)
    # With the current model, there is no real optimization. The least squares
    # function finds no optimum. Probably means the underlying model is so
    # wrong.

    debug()

def final_state(ITSs, y0, t, transformations, parameters, its_len):
    """
    Just the loop
    """
    finals = []
    for its in ITSs:

        seq = its.sequence[:its_len]
        neigh = [seq[cnt] + seq[cnt+1] for cnt in range(len(seq)-1)]

        # note that what the paper calls k1 is my k2; their kminus1 is 9->8, but I'm
        # going the other way!
        k11 = np.array([Ec.kminus1[di] for di in neigh])
        k21 = np.array([Ec.k1[di] for di in neigh])
        k31 = np.array([Ec.NNRD[di] for di in neigh])

        a, b, c = parameters

        k1 = a*k11
        k2 = a*k21
        k3 = b*np.exp(c*k31)

        transf_rates = [k1, k2, k3]

        states = len(k1)
        variables = states*2

        A = parameter_matrix(transf_rates, states, variables)

        # solve the with the new param
        #arguments = (A,) # the abysmal synax requires a double dict as argument
        soln = scipy.integrate.odeint(rnap_solver, y0, t, args = (A,))

        finals.append(soln[-1][-1])

    return finals

def cost_function(parameters, y0, t, its_len, states, variables, ITSs,
                  transformations, PYs):
    """
    Cost function. Could the equations get out of hand? I hope not.
    """

    # get the tuning parameters and the rna-dna and k1, kminus1 values
    a, b, c = parameters

    finals = []
    for its in ITSs:

        seq = its.sequence[:its_len]
        neigh = [seq[cnt] + seq[cnt+1] for cnt in range(len(seq)-1)]

        k11 = np.array([Ec.kminus1[di] for di in neigh])
        k21 = np.array([Ec.k1[di] for di in neigh])
        k31 = np.array([Ec.NNRD[di] for di in neigh])

        # modify the rate constants by some formula
        # maybe you'll make a library of reaction constants here
        k1 = a*k11
        k2 = a*k21
        k3 = b*np.exp(c*k31)

        states = its_len - 1
        variables = states*2

        rates = (k1, k2, k3)

        A = parameter_matrix(rates, states, variables)

        soln = scipy.integrate.odeint(rnap_solver, y0, t, args = (A,))

        finals.append(soln[-1][-1])

    dist_vect = np.array(finals) - np.array(PYs)
    #dista = np.sqrt(sum(dist_vect**2))

    return dist_vect

def rnap_solver(y, t, A):
    """
    Solver.
    """

    dx = np.dot(A, y)

    return dx

def equlib_matrix(rate, state_nr):
    """
    Assume equlibrium for the reaction:
    X_f_0 <=> X_e_0 -> X_f_1

    Then you get d(Xf_1)/dt = k1[Xf_0]

    Where in practice k1 = c1*exp(-\Delta G)/Keq, but this value must come from
    the outside
    X_f_0 -> X_f_1 , k0
    X_f_1 -> X_f_2 , k1
    X_f_2 -> X_f_3 , k2

    X = [X_f_0, X_f_1, X_f_2, X_f_3]
    A = [[-k0,   0 , 0,  0],
         [k0 , -k1 , 0,  0],
         [0  ,  k1, -k2, 0],
         [0  ,  0  , k2, 0]]
    """

    # initialize with the first row
    rows = [  [rate[0]] + [0 for i in range(state_nr-1)] ]

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

def main():
    lizt, ITSs = ReadAndFixData() # read raw data
    #lizt, ITSs = StripSet(0, lizt, ITSs) # strip promoters (0 for nostrip)

    #RedlistInvestigator(ITSs)
    #RedlistPrinter(ITSs)
    #HsuRandomTester(ITSs)
    #PurineLadder(ITSs)

    #genome_wide()

    #new_genome()

    #for i in range(10):
        #new_ladder(lizt)

    #new_scatter(lizt, ITSs)

    # NOTE You have the following proposition
    # Implement your simple model in Matlab and optimize for PY in position 10,
    # 15, and 20 at given time points. Do cross validation by doing this for
    # halve the set and then testing with the second half -- see if you get
    # consistent parameter estimation and consistent validation. PS: don't
    # choose the same twice! You can do this many times and make a MC.
    # Show that you correlate well PY with a simple model. The differnce between
    # the fast-goers and slow-goers is the time they spent. Thus we find both a
    # sequence-specific regulation AND we find the time-dependence.

    # XXX to show that the Keq and RNA-DNA really matter, try to fit to random
    # values for Keq and RNA-DNA. What about -2 as a constant for RNA-DNA?

    # The final challenge is to show that a model with RNA-DNA energy + DNA-DNA
    # for bubble opening performs worse than my model.

    #new_long5UTR(lizt)

    # design new sequences!
    #new_designer(lizt)

    # the ODE models
    #new_models(lizt, ITSs)

    # the naive exp-fitting models
    #fitting_models(lizt, ITSs)

    #frequency_change()
    # NOTE the poly(A) and poly(T) tracts could be regulatory elements of
    # transcriptional slippage

    # TODO what would happen if I tried to fit the PY values using linear or
    # non-linear regression with 20 parameters (x11,x12,x21,x22,etc) for the
    # rna-dna and Keq values?

    # RESULT a weak signal when filtering by greA
    # That can be a plot
    # TODO make the plot, including the standard deviations :S
    # You have to do it at work; you haven't included the data in the repository
    # :S
    #greA_filter()

    # Now plotting the damn expression colors on the PCA plots
    #my_pca()
    #hsu_pca(lizt)
    # NOTE it's not clear what I should do now. There is this odd TTT repeating
    # going on. You should relate to super_en. In super_en the AA and TT have
    # totally opposite meanings. TTTT and AAAA are both associated with lower
    # rpkm.

    # They found promoter proximal stalling at these promoters
    # tnaA, cspA, cspD, rplK, rpsA and rpsU as well as lacZ. How do these stand
    # out in the crowd?

    # TODO you still need to define strong promoters and see if there's a
    # tradeoff there. For strong promoters I would expect a shift in mean energy
    # compared to the rest.
    #new_promoter_strength()

    # You have ribosomal genes + DPKM values for e coli genes
    # Is there any covariance between DPKM and ribosomal gene for the energy
    # terms you've found? I suggest using only 1-promoter genes

    # NOTE nothing on the AP ... maybe you got them wrong or somth. Dozntmatter.
    # last thing to check: abortive probabilities. Would be awzm if they match.
    # TODO: plot the AP against the super_values as bar-plots
    #new_AP(lizt, ITSs)
    # RESULT: variance of AP correlates with variance of energy
    # Does that mean that variance of AP correlates with PY? Yes, but
    # weakly and negatively negatively.

    # RESULTS energy[:msat]/msat gives .8 correlation

    # for scatter-plot: include error bars for both PY and measurements :) They
    # are bound to form a nice shape.

    # maybe it's time for another visit at the AB values? Is there high abortive
    # probability just at certain dinucleotides which correspond to those whose
    # translocation is short?

    # Q: what would be interesting to test? Can we induce abortion at certain
    # sites? What happens with a 'physical' model where rna-dna counts for 8 bp
    # and translocation only for 1?

    # Can you do smart experiments? Reproducing the original result will be
    # piece of cake since the sequences are practically randomized already.

    # What I can think of: "barriers" at different sits. But to do this you
    # should have studied the original 'barriers' too. For example, study all
    # sites with more than 50% probability of aborting. First you should plot
    # them.


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
