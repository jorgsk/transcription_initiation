# coding: utf-8
"""
Dictionaries with dinucleotide values from Hein et al:

RNA Transcript 3′-Proximal Sequence Affects Translocation Bias of RNA Polymerase

And malinen et al

NOTE dinucleotides are in terms of the non-template DNA = 'same as RNA strand'

"""

# in terms of non-template DNA "same as RNA" strand
malinen_et_al_nucleotide_addition_halflives_ms = {
        'A': 19.0, 'C': 8.5,
        'G': 20.3, 'T': 23.5
        }

# in terms of non-template DNA "same as RNA" strand
malinen_et_al_forward_translocation_halflives_ms = {
        'A': 7.2,  'C': 8.7,
        'G': 11.7, 'T': 8.6
        }

# RNA-DNA duplexes from Sugimoto 2002 paper
# BASES ARE FROM THE TEMPLATE DNA STRAND OF A TRANSCRIPTION BUBBLE WITH AN
# RNA-DNA HYBRID XXX TODO: look into this, are you gettting this right? ALL
# your input data should be in terms of the non-template strand.
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

resistant_fraction = {
'AA': 0.64,
'AC': 0.60,
'AG': 0.80,
'AT': 0.29,
'CT': 0.26,
'CA': 0.74,
'CC': 0.66,
'CG': 0.78,
'GT': 0.33,
'GA': 0.68,
'GG': 0.78,
'GC': 0.81,
'TT': 0.27,
'TA': 0.92,
'TC': 0.71,
'TG': 0.96,
}
resistant_fraction_pm = {
'AA': 0.03,
'AC': 0.02,
'AG': 0.01,
'AT': 0.03,
'CT': 0.03,
'CA': 0.02,
'CC': 0.02,
'CG': 0.01,
'GT': 0.03,
'GA': 0.01,
'GG': 0.01,
'GC': 0.01,
'TT': 0.03,
'TA': 0.01,
'TC': 0.01,
'TG': 0.01,
}
# /min, indicates reverse translocation
pyrophosphorolysis_forward = {
'AA':0.06, 'AC':0.08, 'AG':0.04, 'AT':0.31,
'CA':0.04, 'CC':0.10, 'CG':0.03, 'CT':0.29,
'GA':0.08, 'GC':0.08, 'GG':0.05, 'GT':0.36,
'TA':0.01, 'TC':0.04, 'TG':0.007,'TT':0.17,  # interestingly, TN is lowest for all
}
pyrophosphorolysis_forward_pm = {
'AA':0.01,
'AC':0.01,
'AG':0.01,
'AT':0.05,
'CT':0.04,
'CA':0.01,
'CC':0.02,
'CG':0.01,
'GT':0.09,
'GA':0.01,
'GG':0.01,
'GC':0.02,
'TT':0.02,
'TA':0,
'TC':0.01,
'TG':0
}
# /min, indicates forward translocation
pyrophosphorolysis_reverse = {
'AA':0.10, 'AC':0.11, 'AG':0.13, 'AT':0.12,
'CA':0.11, 'CC':0.19, 'CG':0.10, 'CT':0.10,
'GA':0.16, 'GC':0.36, 'GG':0.16, 'GT':0.17,  # interestingly, GN is highest for all N
'TA':0.14, 'TC':0.10, 'TG':0.14, 'TT':0.06,
}
pyrophosphorolysis_reverse_pm {
'AA':0.03,
'AC':0.02,
'AG':0.03,
'AT':0.03,
'CT':0.02,
'CA':0.03,
'CC':0.03,
'CG':0.02,
'GT':0.05,
'GA':0.02,
'GG':0.02,
'GC':0.07,
'TT':0.01,
'TA':0,
'TC':0.02,
'TG':0,
}
Keq_EC8_EC9 = {
'AA':0.6,
'AC':0.7,
'AG':0.3,
'AT':2.6,
'CT':2.9,
'CA':0.4,
'CC':0.5,
'CG':0.3,
'GT':2.1,
'GA':0.5,
'GG':0.3,
'GC':0.2,
'TT':2.8,
'TA':0.09,
#'TA':0.18,
'TC':0.4,
'TG':0.05,
#'TG':0.15,
}
Keq_EC8_EC9_pm = {
'AA':0.2,
'AC':0.2,
'AG':0.1,
'AT':0.8,
'CT':0.7,
'CA':0.1,
'CC':0.1,
'CG':0.1,
'GT':0.8,
'GA':0.1,
'GG':0.1,
'GC':0.1,
'TT':0.6,
'TA':0,
'TC':0.1,
'TG':0,
}

