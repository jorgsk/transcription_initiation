""" Generates random DNA sequences of length 20 and makes a selection of these
sequences based on i) RNA-DNA energy in a uniform range (hypothesis testing),
ii) absence of promoter -10 motif (removing unwanted sigma rebinding (but it was
found that the -10 rebind was of an unrelated sequences..!),iii) absence of
polyA-T(-C-G?) streches (transcriptional slippage), and iv) ... """

from __future__ import division
from Bio import Seq
from matplotlib import pyplot as plt
import numpy as np
import random
import operator
# My own modules
import Filereader
import Energycalc
import time

def RanGen(gatc_bias = (0.2, 0.3, 0.3, 0.2)):
    """ A generator for random 20nt ITR sequences. Nucleotides are selected with
    the probabilities given in gatc_freq for the respective nucleotides. """
    nucleotides = ['G','A','T','C']
    while True:
        ITR = 'AT'+''.join([nucleotides[weighted_choice(gatc_bias)] for dummy in range(18)])
        yield ITR

def weighted_choice(weights):
    """ Get a weigheted random index from weights. Weights should be of the type
    [3,4,6] to give probabilities 0.3, 0.4, and 0.6 to be picked. """
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

minus10s = Filereader.MinusTen()
modMin = [row[2:] for row in minus10s] # using only the last six nts of the -10
modMin = list(set(modMin)) # Removing duplicates!!!!!!!!!!!!!!
curated = [('ACGAT',[2,7]), ('ACGCT',[2,7]), ('TAACGCT',[2,8]), ('TAATTTT',
                                                                 [10,15]),
           ('ATTGT',[]),('TAGGA',[]),('TATAAT', [])]

# All possible repeats from NNN to N*20
# NOTE you are no longer sure about screening with these. So far you're only
# screening against known sequences.
repeatss = [letter*x for letter in ['G','A','T','C'] for x in range(3,5)]
repeatss = [letter*x for letter in ['A','T'] for x in range(3,5)]
repeatsl = [letter*x for letter in ['A','T'] for x in range(8,21)]

def GenerateAndFilter(runs, gatc_bias):
    """ Return those sequences that do not contain redlisted
    subsequences. Problem: the energy range becomes too small with extended
    filtering."""
    t1 = time.time()
    hits = 0
    survivers = []
    for index, ITR in enumerate(RanGen(gatc_bias)): # calling the iterator
        keep = True
#        for redlist in repeatss:
#            if redlist in ITR[:6]:
#                hits = hits +1
#                keep = False
#                break
        #for redlist in repeatsl:
            #if redlist in ITR:
                #hits = hits +1
                #keep = False
                #break
        for redlist in curated:
            pos = redlist[1] # getting the location where it shouldnt be
            if pos == []: pos = [0,19] # if no specific loc -> everywhere
            if redlist[0] in ITR[pos[0]:pos[1]]:
                hits = hits +1
                keep = False
                break
        if keep:
            survivers.append(ITR)

        if index == runs: break
    print 'percentage of hits', hits/runs
    print time.time() - t1 # take time
    return survivers

# Where in the sequence should you value RNA-DNA energy most!? I say early
# sequence. Sort them primarily by +1 to +15, and secondarily from +1 to +10
# , and thirdly +15 to +20.
# Algorithm: Generate X sequences, and add to a list only those that escape the
# sequence list of death (TATAAT etc). From that list, finally, sort them into
# 40 (y, variable) positions ranked on RNA-DNA energy. The sorting should be
# done depending on the distribution of RNA-DNA energy +1 to +15. 

def EnergyAdder(survivers):
    """ Add energies to the ITR for sorting. """
    newsurv = []
    for itr in survivers:
        one_fift = Energycalc.RNA_DNAenergy(itr[:])
        one_ten = Energycalc.RNA_DNAenergy(itr[:10])
        fift_twent = Energycalc.RNA_DNAenergy(itr[15:])
        newsurv.append([itr, one_fift, one_ten, fift_twent])
    # Sorting first by fift_twent, then by one_ten, and then by one_fift, in
    # order to have list sorted the way I want.
    newsurv = sorted(newsurv, key=operator.itemgetter(3)) # sort by 15-20
    newsurv = sorted(newsurv, key=operator.itemgetter(2)) # sort by 1-10
    newsurv = sorted(newsurv, key=operator.itemgetter(1)) # sort by 1-15
    return newsurv

def ITRselecter(ITRs):
    """ Select ITRs evenly spaced in energy from max energy to min energy in
    itrs. Select using the +1 to +15 energies. Needs that the number of itrs is
    big (10000+) so that energies are guaranteed to be found."""
    nr_of_itrs = 40 #You need at least 40 sequences.
    # your desired energy range
    en_min = -19
    en_max = -8
    selected = []
    en_range = np.linspace(en_min, en_max, nr_of_itrs)
    for energy in en_range:
        itr_index = 0
        # ITRs is presorted with minimum energy first
        for ind, itr in enumerate(ITRs[itr_index:]):
            # if this itr has higher en, append the previous
            if itr[1] > energy:
                selected.append(ITRs[ind-1])
                itr_index = ind
                break
    for sel in selected:
        print sel[1]
    return selected

def Main():
    # nr of generated sequences
    runs = 10000
    # bias in probability of random selection of gatc nucleotides to get the
    # desired energy distribution
    gatc_bias = (15,35,35,15)
    survivers1 = GenerateAndFilter(runs, gatc_bias)
    survivers2 = EnergyAdder(survivers1)
    selected   = ITRselecter(survivers2)

    return selected

selected = Main()

# RESULT Hsu's sequences are not random! :S :)
# 0:19 sequences starting with AT (as in Hsu's set)

# RESULT Hsu's 1 to 15 energies go from -19 to -8 with the peak PY occuring
# around -12. Ideally your energies should go from -18 to -8 in 1 to 15
# the 1 to 10 energies go from -12 (one at -14) to -4
# RESULT I'm gettting the desired energy range with 0.15 and 0.35 probabilities for
# gc and at. RESULT selected are now as good as they come? You are content with
# this anyway and leave off from here.

# UPDATE IDEA: 1) Should not contain polyA/polyT in the first 4-5 nucleotides.
# After that polyN is OK, but less than 8. That's it! 
# Pause sites in:
# Sequence-Resolved Detection of Pausing by Single RNA Polymerase Molecules

# You have saved all the -10 and -15 sequences in 'sequence_data' folder. See
# the nature paper at the bottom of the zotero list for sigma for more info!
# Good info. Too similar to ATAAT is bad, because then the site will (can) be
# used for de novo initiation! Another reason to exclude it.

# Following sequences are from A. Hatoum and J. Roberts 2008 Mol. Microbiology
# lambda:ACGAT (+2-+6)
# phi80: ACGCT (+2-+6)
# 21:   TAACGCT (+2-+7)
# 82:   TAATTTT (+10-+15)
# lac site is from nature paper 2004
# lac:  ATTGT TODO: find out where it is.
# extended -10: -T-TG-TATAAT
# from 2009 JBC stepanova paper: TAGGA element TODO where is it????
#
# You can cite a new sigma (in zotero) paper that sigma can rebind and -10-likes
# everywhere, but maybe especially in the beginning of transcription. A -10 like
# caused pause in the ITR but not de novo initiation; when the sequence was
# mutated to be more similar to consensus, de novo synthesis was initiatied
# (nature 2004).
# NOTE: record how often you find these sequences from the randomly generated
# ones! RESULT: you find them a lot.
# 
# RESULT: it doesn't seem that sigma binding plays a role in the in vitro
# experiments. Still, I think you should remove curated sequences AT THEIR
# LOCATION, because the motifs might not be important outside their context.
