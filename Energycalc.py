""" Calculate energy-levels: RNA/DNA and RNA/RNA """
from Bio.Seq import Seq
import copy
import numpy as np


def complement(in_dict):
    """
    Return the dinucleotide -> value dict with the complementary dinucleotide.

    Example 'AC': 5 will be returned as 'TG': 5

    """
    translate = {'A':'T', 'C': 'G', 'G':'C', 'T':'A', 'U':'A'}

    out_dict = {}

    for rna_din, val in in_dict.items():
        if len(rna_din) == 2:
            dna_din = ''.join([translate[nuc] for nuc in list(rna_din)])

            out_dict[dna_din] = val
        else:
            # symmertry, terminal, etc. (non-dinucleotide entries)
            out_dict[rna_din] = val

    return out_dict


# RNA-DNA duplexes from Sugimoto 2002 paper
# BASES ARE FROM THE TEMPLATE DNA STRAND OF A TRANSCRIPTION BUBBLE WITH AN
# RNA-DNA HYBRID
NNRD = {'AA':-0.4,
        'AC':-1.6,
        'AG':-1.4,
        'AT':-1.0,
        'CA':-1.0,
        'CC':-1.5,
        'CG':-1.2,
        'CT':-0.9,
        'GA':-1.4,
        'GC':-2.4,
        'GG':-2.2,
        'GT':-1.5,
        'TA':-0.3,
        'TC':-0.8,
        'TG':-1.0,
        'TT':-0.2,
        'Initiation':1.0}
# EnLibRNA is for calculating expected random sequence values

EnLibRNA = copy.deepcopy(NNRD)
del(EnLibRNA['Initiation'])

# DNA-DNA duplexes from Santalucia 2004 paper in the 5->3 direction
NNDD = {'AA':-1.00,
        'TT':-1.00,
        'AT':-0.88,
        'TA':-0.58,
        'CA':-1.45,
        'TG':-1.45,
        'GT':-1.44,
        'AC':-1.44,
        'CT':-1.28,
        'AG':-1.28,
        'GA':-1.30,
        'TC':-1.30,
        'CG':-2.17,
        'GC':-2.24,
        'GG':-1.84,
        'CC':-1.84,
        'TerminalAT':0.05,
        'Initiation':1.96,
        'Symmetry':0.43}
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

    indiv = list(sequence)  # splitting sequence into individual letters
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

promoter = 'ATAATAGATTC'  # Last 11 nt's of promoter sequence (next is +1)
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
from dinucleotide_values import resistant_fraction, k_forward, k_reverse, Keq_EC8_EC9

# recalculate the Keq and the reverse for more accuracy
invEq = {}
reKeq = {}

for dinuc, val in k_reverse.items():
    invEq[dinuc] = val/k_forward[dinuc]
    reKeq[dinuc] = k_forward[dinuc]/val

    # The inverse has a large problem since some kf the k1 values are very small
    if dinuc == 'TA':
        invEq[dinuc] = 5
    if dinuc == 'TG':
        invEq[dinuc] = 6
    # You are fixing it by ad-hoc reducing some values. This helps to bring
    # positive correlation like you expected!


def seq2din(sequence):
    indiv = list(sequence)  # splitting sequence into individual letters
    return [indiv[cnt] + indiv[cnt+1] for cnt in range(len(indiv)-1)]

# XXX go from Keq to Delta G values. Do 1/Keq because Hein paper has got the
# equilibrium constant for the opposite reaction.
# Keq = exp(-DG/RT) -> DG = -ln(Keq)*RT
# Keq_EC8_EC9 is already in dna coding strand sequence

RT = 1.9858775*(37 + 273.15)
#dna_keq = complement(Keq_EC8_EC9)
dna_keq = Keq_EC8_EC9

## Jorgen: this is not keq, this is delta G!!
deltaG_f = {}
for din, en in dna_keq.items():
    deltaG_f[din] = RT*np.log(en)/1000  # divide by 1000 to get kcal

deltaG_b = {}
for din, en in dna_keq.items():
    deltaG_b[din] = -RT*np.log(en)/1000  # divide by 1000 to get kcal


def Delta_trans_forward(sequence):
    """
    Delta values for keq for the forward reaction
    """

    return sum([deltaG_f[din] for din in seq2din(sequence)])


def Delta_trans_backward(sequence):
    """
    Delta values for keq for the backward reaction
    """

    return sum([deltaG_b[din] for din in seq2din(sequence)])


def res_frac(sequence):
    """ Return the DNA/RNA binding energy of 'sequence'. Now skipping
    initiation cost. """
    if len(sequence) < 2:
        return 0

    return sum([resistant_fraction[din] for din in seq2din(sequence)])


def k_forward(sequence):
    """ Return the rate coefficient [/min] of forward translocation of 'sequence' """
    if len(sequence) < 2:
        return 0

    return sum([k_forward[din] for din in seq2din(sequence)])


def k_backward(sequence):
    """ Return the rate coefficient [/min] of reverse translocation of 'sequence'. """
    if len(sequence) < 2:
        return 0

    return sum([k_reverse[din] for din in seq2din(sequence)])


def Keq(sequence):
    """ Return the Keq of 'sequence'. """
    if len(sequence) < 2:
        return 0

    return sum([Keq_EC8_EC9[din] for din in seq2din(sequence)])


def Keq_mine(sequence):
    """ Return the Keq of 'sequence'. """
    if len(sequence) < 2:
        return 0

    return sum([k_forward[din]/k_reverse[din] for din in seq2din(sequence)])
