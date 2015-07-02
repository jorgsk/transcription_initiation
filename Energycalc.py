""" Calculate energy-levels: RNA/DNA and RNA/RNA """
from Bio.Seq import Seq
import copy
import numpy as np
from ipdb import set_trace as debug

# load nucleotide and dinucleotide values from various papers
from dinucleotide_values import resistant_fraction, Keq_EC8_EC9, NNRD, NNDD,\
malinen_et_al_nucleotide_addition_halflives_ms,\
malinen_et_al_forward_translocation_halflives_ms, pyrophosphorolysis_forward,\
pyrophosphorolysis_reverse, scaled_hein_translocation_forward_rate_constants,\
scaled_hein_translocation_reverse_rate_constants


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


# EnLibRNA is for calculating expected random sequence values
EnLibRNA = copy.deepcopy(NNRD)
del(EnLibRNA['Initiation'])

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

dinucleotide_deltaG_f = {}
for din, en in dna_keq.items():
    dinucleotide_deltaG_f[din] = RT*np.log(en)/1000  # divide by 1000 to get kcal

dinucleotide_deltaG_b = {}
for din, en in dna_keq.items():
    dinucleotide_deltaG_b[din] = -RT*np.log(en)/1000  # divide by 1000 to get kcal


def Ms2s(value):
    """
    Convert unit from miliseconds to seconds
    """
    return value * 1000


def HalfLife2RateConstant(hl):
    """
    Assuming first order kinetics
    """
    return 0.693/hl


def ScaleHeinRateConstants(hein_reverse_pyrophosphorolysis, direction='forward'):
    """
    Take Hein's reverse pyrophosphorolysis rate constants and scale them so
    that they are of similar order [/min -> /sec] to Malinen's measured values.

    Hein's values are for example AG = 0.3, TG = 0.1 /min,
    which is AG = 0.3/60 = 0.005/sec, TG = 0.1/60 = 0.00167/sec

    While Malinen's values are half lives for forward translocation reactions.
    For example, for a 3' rna ending with 'G' hl = 11.7 ms. Assuming first
    order reactions, this can be converted to a rate constant 0.059 / ms, which
    is the same as 59.12/sec

    The solution approach is to scale all of Hein's rate constants such that the
    average for all dinucleotides ending in 'G' is 59.12.

    Scale is derived for forward translocation, but apply the same to Hein's
    forward pyrophosphorolysis.

    After you scaled these, you put them in the dinucleotide value file
    """
    scaled_hein_translocation_forward_rate_constants_dict = {}
    scaled_hein_translocation_reverse_rate_constants_dict= {}

    gatc = ['G', 'A', 'T', 'C']
    for nt in gatc:
        malinen_half_life = malinen_et_al_forward_translocation_halflives_ms[nt]
        malinen_rate_constant_ms = HalfLife2RateConstant(malinen_half_life)
        malinen_rate_constant_sec = malinen_rate_constant_ms * 1000

        # gather all dinucleotides that end in nt
        nt_ending_dinucs = [nuc + nt for nuc in gatc]

        # get their average rate constant
        mean_rate_constant = np.mean([pyrophosphorolysis_reverse[dnt] for dnt
            in nt_ending_dinucs])

        # get the scale
        scale = mean_rate_constant / malinen_rate_constant_sec

        # do the scaling
        for dnt in nt_ending_dinucs:

            scaled_hein_translocation_forward_rate_constants_dict[dnt] =\
                pyrophosphorolysis_reverse[dnt] / scale
            scaled_hein_translocation_reverse_rate_constants_dict[dnt] =\
                pyrophosphorolysis_forward[dnt] / scale

    #return scaled_hein_translocation_forward_rate_constants, scaled_hein_translocation_reverse_rate_constants


def Delta_trans_forward(sequence):
    """
    Delta values for dinucleotide keq for the forward reaction
    """

    return sum([dinucleotide_deltaG_f[din] for din in seq2din(sequence)])


def Delta_trans_backward(sequence):
    """
    Delta values for dinucleotide keq for the backward reaction
    """

    return sum([dinucleotide_deltaG_b[din] for din in seq2din(sequence)])


def res_frac(sequence):
    """ Return the DNA/RNA binding energy of 'sequence'. Now skipping
    initiation cost. """
    if len(sequence) < 2:
        return 0

    return sum([resistant_fraction[din] for din in seq2din(sequence)])


def K_forward(sequence):
    """ Return the rate coefficient [/min] of forward translocation of 'sequence' """
    if len(sequence) < 2:
        return 0

    return sum([k_forward[din] for din in seq2din(sequence)])


def K_backward(sequence):
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
