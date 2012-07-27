""" Calculate energy-levels: RNA/DNA and RNA/RNA """
from Bio.Seq import Seq
import copy
import numpy as np

# RNA-DNA duplexes from 2002 paper
# DNA bases are given.
NNRD = {'TT':-0.4,'TG':-1.6,'TC':-1.4,'TA':-1.0,'GT':-1.0,'GG':-1.5,'GC':-1.2,'GA':-0.9,'CT':-1.4,'CG':-2.4,'CC':-2.2,'CA':-1.5,'AT':-0.3,'AG':-0.8,'AC':-1.0,'AA':-0.2,'Initiation':1.0}
# EnLibRNA is for calculating expected random sequence values
EnLibRNA = copy.deepcopy(NNRD)
del(EnLibRNA['Initiation'])

# DNA-DNA duplexes from 2004 paper in the 5->3 direction
NNDD = {'AA':-1.0,'TT':-1.0,'AT':-0.88,'TA':-0.58,'CA':-1.45,'TG':-1.45,'GT':-1.44,'AC':-1.44,'CT':-1.28,'AG':-1.28,'GA':-1.3,'TC':-1.3,'CG':-2.17,'GC':-2.24,'GG':-1.84,'CC':-1.84,'TerminalAT':0.05,'Initiation':1.96,'Symmetry':0.43}
EnLibDNA = copy.deepcopy(NNDD)
del(EnLibDNA['Initiation'])

def RNA_DNAexpected(length):
    """ Returns expected RNA/DNA energy for random sequence of length 'length'. """
    mean = np.mean(EnLibRNA.values())
    # IT's flawed!! The nucleotides are not evenly distributed in the table!
    if length == 8:
        return -7.275
    if length == 9:
        return -8.447

    return mean*(length-1)

def DNA_DNAexpected(length):
    """ Returns expected DNA/DNA energy for random sequence of length 'length'. """
    mean = np.mean(EnLibDNA.values())
    return mean*(length-1)

def RNA_DNAenergy(sequence):
    """ Calculate the DNA/RNA binding energy of 'sequence'. Now skipping
    initiation cost. """
    if len(sequence) < 2:
        return 0

    indiv = list(sequence) # splitting sequence into individual letters
    neigh = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]
    energ = sum([NNRD[nei] for nei in neigh])
#    energ = energ + NNRD['Initiation']
    return energ

def DNA_DNAenergy(sequence):
    """ Calculate the DNA/DNA binding energy of 'sequence'. """
    if len(sequence) < 2:
        return 0

    indiv = list(sequence)
    neigh = [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]
    energ = sum([NNDD[nei] for nei in neigh])
    # Terminal AT penalty
    if neigh[-1][-1] in ['A','T']:
        energ = energ + NNDD['TerminalAT']
    if neigh[0][0] in ['A','T']:
        energ = energ + NNDD['TerminalAT']
    # Symmetry penalty
    if Seq(sequence).complement().tostring() == sequence[::-1]:
        energ = energ + NNDD['Symmetry']
    energ = energ + NNDD['Initiation']
    return energ

def PhysicalRNA(sequence, msat, msatyes,maxlen):
    """ Return RNA/DNA binding energies for ITS-sequence. If msatyes == 'yes':
        considering max size of RNA/DNA hybrid as 'maxlen', adding first up to
        'maxlen', and then follows a window of size 'maxlen' until either 20 or
        msat. (pos1,pos2) = (0,6) means first 6 sequences
        (1,2,...,6) in the real sequence. Adds trailing E[energy] if
        msatyes=='yes'."""
    seqlist = list(sequence)
    enlist = []
    for pos1 in range(len(seqlist)-maxlen+1):
        if pos1 == 0:
            for pos2 in range(pos1+2,pos1+maxlen+1):
                subseq = ''.join([seqlist[tic] for tic in range(pos1,pos2)])
                enlist.append([RNA_DNAenergy(subseq),[pos1, pos2]])
        else:
            subseq = ''.join([seqlist[tic] for tic in range(pos1,pos1+maxlen)])
            enlist.append([RNA_DNAenergy(subseq),[pos1,pos1+maxlen]])
    # Adding expected RNA/DNA energy for maxlen bond +1 after msat values for fair
    # comparison between transcripts of different lengths.
    expEnrgy = RNA_DNAexpected(maxlen)
    if msatyes == 'yes':
        for zeror in range(int(msat)-1,19):
            enlist[zeror][0] = expEnrgy
    return enlist

promoter = 'ATAATAGATTC' # Last 11 nt's of promoter sequence (next is +1)
bubsize = 13
def PhysicalDNA(sequence):
    """ Return DNA/DNA hybrid energy of ITS-sequence using the (-11, 20)-element.
    (pos1,pos2) = (0,13) really means that the sequence 0,1,...,12 is joined. """
    seqlist = list(promoter+sequence)
    enlist = []
    pos1 = 0
    for pos2 in range(pos1+bubsize,len(seqlist)+1):
        subseq = ''.join([seqlist[tic] for tic in range(pos1,pos2)])
        enlist.append((DNA_DNAenergy(subseq),(pos1,pos2)))
    return enlist


# add the stuff from new paper
from dinucleotide_values import resistant_fraction, k1, kminus1, Keq_EC8_EC9

# recalculate the Keq and the reverse for more accuracy

invEq = {}
reKeq = {}

for dinuc, val in kminus1.items():
    invEq[dinuc] = val/k1[dinuc]
    reKeq[dinuc] = k1[dinuc]/val

# try to normalize some values for invEq
# result: correlation reappears with 'sensible' values
#invEq['TA'] = 5
#invEq['TG'] = 6

def seq2din(sequence):
    indiv = list(sequence) # splitting sequence into individual letters
    return [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]

def dnaizer(in_dict):
    """
    in_dict has got RNA dinucleotides to energy, NN: float

        XXXXXNN
        YYYYYZZ

    return ZZ: float to get he DNA version
    """
    translate = {'A':'T', 'C': 'G', 'G':'C', 'T':'A', 'U':'A'}

    out_dict = {}

    for rna_din, val in in_dict.items():
        dna_din = ''.join([translate[nuc] for nuc in list(rna_din)])

        out_dict[dna_din] = val

    return out_dict

# XXX go from Keq to Delta G values. Do 1/en because Hein paper has got the
# equilibrium constant for the opposite reaction.
# Keq = exp(-DG/RT) -> DG = -ln(Keq)*RT
# Also go from RNA to DNA sequence

RT = 1.9858775*(37 + 273.15)
dna_keq = dnaizer(Keq_EC8_EC9)

delta_keq = {}
for din, en in dna_keq.items():
    delta_keq[din] = -RT*np.log(1/en)/1000  #divide by 1000 to get kcal

def Delta_trans(sequence):
    """
    Delta values for keq
    """

    return sum([delta_keq[din] for din in seq2din(sequence)])

def res_frac(sequence):
    """ Calculate the DNA/RNA binding energy of 'sequence'. Now skipping
    initiation cost. """
    if len(sequence) < 2:
        return 0

    return sum([resistant_fraction[din] for din in seq2din(sequence)])

def K1(sequence):
    """ Calculate the K1 of 'sequence' """
    if len(sequence) < 2:
        return 0

    return sum([k1[din] for din in seq2din(sequence)])

def Kminus1(sequence):
    """ Calculate the K_-1 of 'sequence'. """
    if len(sequence) < 2:
        return 0

    return sum([kminus1[din] for din in seq2din(sequence)])

def Keq(sequence):
    """ Calculate the Keq of 'sequence'. """
    if len(sequence) < 2:
        return 0

    return sum([Keq_EC8_EC9[din] for din in seq2din(sequence)])

