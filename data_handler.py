from __future__ import division
import Energycalc as Ec
import Workhouse
import pandas
from numpy import mean, std

from ipdb import set_trace as debug

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

        # These are optional data which are used if the raw quantitations are
        # available
        self.quantitations = []
        # all data lists go from RNA-mer 2 to 21
        self.abortiveProb = [] # mean
        self.abortiveProb_std = [] # std

        # Raw quants
        self.rawData = {} # one entry for each quantitation
        self.rawDataMean = -1 # one entry for each quantitation
        self.rawDataStd = -1  # one entry for each quantitation

        # FL
        self.fullLength = {}
        self.fullLengthMean = -1
        self.fullLengthStd = -1

        # helper arrays
        self._prctYield = {}
        self._RNAprodFromHere = {}

        # Redlisted sequences 
        self.redlist = []
        self.redlisted = False

        # make a dinucleotide list of the ITS sequence
        __seqlen = len(self.sequence)
        __indiv = list(self.sequence)
        __dinucs = [__indiv[c] + __indiv[c+1] for c in range(__seqlen-1)]

        # Make di-nucleotide vectors for all the energy parameters
        # goes from 2-mer till length of sequence
        self.rna_dna_di = [Ec.NNRD[di] for di in __dinucs]
        self.dna_dna_di = [Ec.NNDD[di] for di in __dinucs]
        self.keq_delta_di_f = [Ec.delta_keq_f[di] for di in __dinucs]
        self.keq_delta_di_b = [Ec.delta_keq_b[di] for di in __dinucs]

    def __repr__(self):
        return "{0}, PY: {1}".format(self.name, self.PY)

    def averageRawDataAndFL(self):
        """
        More averaging and stding
        """

        if not self.sane():
            return

        # average mean
        self.fullLengthMean = mean(self.fullLength.values())
        self.fullLengthStd = std(self.fullLength.values())

        # average raw data
        self.rawDataMean = mean(self.rawData.values(), axis=0)
        self.rawDataStd = std(self.rawData.values(), axis=0)

    def sane(self):
        """
        Verify that raw abortive data and full length data are in place
        """

        if self.rawData == []:
            print('No raw data to work with!')
            return False

        elif self.fullLength == {}:
            print('No full length (FL) to work with!')
            return False

        else:
            return True

    def calc_PY(self):
        """
        Use the raw data to calculate PY (mean and std)
        """

        if not self.sane():
            return

        # store the PY for each quantitation
        self._PYraw = {}

        for quant in self.quantitations:

            totalRNA = sum(self.rawData[quant]) + self.fullLength[quant]
            self._PYraw[quant] = self.fullLength[quant]/totalRNA

        self.PY = mean([py for py in self._PYraw.values()])
        self.PY_std = std([py for py in self._PYraw.values()])


    def calc_AP(self):
        """
        Use the raw data to calculate AP (mean and std)
        """

        if not self.sane():
            return

        # store the AP for each quantitation
        self._APraw = {}
        self._totAbort = {}
        self._totRNA = {}

        # go through each quantitation
        for quant in self.quantitations:

            # define tmp arrays to work with
            rawD = self.rawData[quant]
            fullL = self.fullLength[quant]

            totalAbortive = sum(rawD)
            totalRNA = totalAbortive + fullL
            # Save these two for posterity
            self._totAbort[quant] = totalAbortive
            self._totRNA[quant] = totalRNA

            rDRange = range(len(rawD))

            # percentage yield at this x-mer
            prctYield = [(rD/totalRNA)*100 for rD in rawD]

            # amount of RNA produced from here and out
            RNAprodFromHere = [sum(rawD[i:]) + fullL for i in rDRange]

            # percent of RNAP reaching this position
            prctRNAP = [(rpfh/totalRNA)*100 for rpfh in RNAprodFromHere]

            # probability to abort at this position
            # More accurately: probability that an abortive RNA of length i is
            # produced (it may backtrack and be slow to collapse, producing
            # little abortive product but reducing productive yield)
            self._APraw[quant] = [prctYield[i]/prctRNAP[i] for i in rDRange]

        self.abortiveProb = mean([ap for ap in self._APraw.values()], axis=0 )
        self.abortiveProb_std = std([ap for ap in self._APraw.values()], axis=0 )


def PYHsu(filepath):
    import csv
    """ Read Hsu dg100 csv-file with PY etc data. """
    f = open(filepath, 'rb')
    a = csv.reader(f, delimiter='\t')
    b = [[Workhouse.StringOrFloat(v) for v in row] for row in a]
    f.close()

    lizt = [[row[0], row[1], row[2], row[3], row[4], row[6], row[8], row[10],
             row[11]] for row in b]

    # Making a list of instances of the ITS class.
    ITSs = [ITS(row[1], row[0], row[2], row[3], row[6], row[7]) for row in lizt]

    return ITSs

def read_dg400(path, files):
    """
    Read data for the DG400 series so you can work with them later
    """

    # the row labels
    labels = ["{0} mer".format(val) for val in range(2, 22)]
    ITSs = {}

    # first get the DNA sequences
    seqs = {}
    seq_file = '/home/jorgsk/Dropbox/phdproject/hsuVitro/output_seqs/tested_seqs.txt'
    last5 = 'GAGTT'
    for line in open(seq_file, 'rb'):
        variant, seq = line.split()
        # special case to catch spelling
        if variant.endswith('A1anti'):
            seqs['N25/A1anti'] = seq + last5
        else:
            seqs[variant] = seq + last5

    import os

    for fNr, fName in files.items():
        target = os.path.join(path, fName)
        rawData = pandas.read_csv(target, index_col=0)

        for variant in rawData:

            # turn nans to zeros
            entry = rawData[variant].fillna(value=-99)

            # not replica unles contains '.'
            replica = ''
            if '.' in variant:
                variant, replica = variant.split('.')

            # if exists (even as replica), append to it
            if variant in ITSs:
                itsObj = ITSs[variant]
            # if not, create new
            else:
                itsObj = ITS(seqs[variant], variant)
                ITSs[variant] = itsObj

            saveas = str(fNr)
            if replica != '':
                saveas = saveas + '.' + replica

            # get the raw reads for each x-mer
            rawReads = [entry[l] for l in labels]
            itsObj.rawData[saveas] = rawReads
            itsObj.fullLength[saveas] = entry['FL']
            itsObj.quantitations.append(saveas)

    # calculate AP and PY
    for name, itsObj in ITSs.items():
        itsObj.calc_AP()
        itsObj.calc_PY()
        itsObj.averageRawDataAndFL()
        itsObj.labels = labels

    # return a list sorted by ITS name/PY to be consistent
    from operator import attrgetter as atrb
    #ITSs = sorted([obj for obj in ITSs.values()], key=atrb('name'))
    ITSs = sorted([obj for obj in ITSs.values()], key=atrb('PY'))

    return  ITSs

def add_AP(ITSs):
    """
    Parse the AP file for the DG100 library and get the AP in there
    """
    dg100ap = 'Hsu_original_data/AbortiveProbabilities/abortiveProbabilities_mean.csv'
    APs = pandas.read_csv(dg100ap, index_col=0).fillna(value=-999)
    for its in ITSs:
        its.abortiveProb = APs[its.name].tolist()

    return ITSs

def ReadData(dataset):
    """ Read Hsu data. """

    # Selecting the dataset you want to use
    if dataset == 'dg100':
        path = 'sequence_data/Hsu/csvHsu'
        ITSs = PYHsu(path) # Unmodified Hsu data

        ITSs = add_AP(ITSs)

    elif dataset == 'dg400':
        path = '/home/jorgsk/Dropbox/phdproject/hsuVitro/prediction_experiment/raw_data'
        files = {'16_first':  'quant16_raw.csv',
                 '27_second': 'quant27_raw.csv',
                 '23_first':  'quant23_raw.csv'}
        #files = {'27': 'quant27_raw.csv',
                 #'16':  'quant16_raw.csv',
                 #'23':  'quant23_raw.csv'}

        ITSs = read_dg400(path, files)

    else:
        print('Provide valid dataset input to ReadData!')
        return 0

    return ITSs
