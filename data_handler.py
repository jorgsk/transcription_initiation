from __future__ import division
import Workhouse
import pandas
import os

from ITS import ITS

from ipdb import set_trace as debug  # NOQA


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


def read_raw(path, files, dset, skipN25=False):
    """
    Read raw quantitation data into ITS objects and calculate relevant values.
    """

    # 1) test: assert that the names of the promoter variants are the same in
    # all the raw-files (and that the set size is identical)

    # 2) test assert that there is a match between the names in the sequence
    # file and the names in the should I?

    cwd = '/home/jorgsk/Dropbox/phdproject/hsuVitro/'

    ITSs = {}
    seqs = {}

    # specify the location of the ITS DNA sequence
    if dset == 'dg400':
        seq_file = cwd + 'output_seqs/tested_seqs.txt'

    if dset == 'dg100':
        seq_file = cwd + 'sequence_data/Hsu/dg100Seqs'

    for line in open(seq_file, 'rb'):
        variant, seq = line.split()

        # special case to catch spelling
        if variant.endswith('A1anti'):
            seqs['N25/A1anti'] = seq
        else:
            seqs[variant] = seq

    for fNr, fName in files.items():
        target = os.path.join(path, fName)
        rawData = pandas.read_csv(target, index_col=0)

        # 2-mer to 22-mer, depending on the gel.
        labels = [i for i in rawData.index if not i.startswith('F')]

        for variant in rawData:

            # skip all N25s if requested
            if skipN25 and 'N25' in variant:
                continue

            # turn nans to zeros
            entry = rawData[variant].fillna(value=-99)

            # deal with replicas that contain a '.'
            replica = ''
            if '.' in variant:
                variant, replica = variant.split('.')

            # Some N25antis have different names
            if 'N25anti' in variant:
                variant = 'N25anti'

            # if exists (even as replica), append to it
            if variant in ITSs:
                itsObj = ITSs[variant]
            # if not, create new
            else:
                if variant[:-1] in ITSs:
                    var = variant[:-1]
                    itsObj = ITSs[var]
                else:
                    itsObj = ITS(seqs[variant], variant)
                    ITSs[variant] = itsObj

            saveas = str(fNr)
            if replica != '':
                saveas = saveas + '.' + replica

            # get the raw reads for each x-mer
            rawReads = [entry[l] for l in labels]
            itsObj.rawData[saveas] = rawReads
            itsObj.fullLength[saveas] = entry['FL']  # first gel has FL2?
            itsObj.quantitations.append(saveas)

    # calculate AP and PY
    for name, itsObj in ITSs.items():
        itsObj.calc_AP()
        itsObj.calc_PY()
        itsObj.averageRawDataAndFL()
        itsObj.calc_AbortiveYield()
        itsObj.labels = labels

    return ITSs


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
    """ Read Hsu data.

    Possible input is dg100, dg100-new, dg400.

    dg100 is the version of the dg100 dataset you have used the past 3 years.
    Here you just used the PY values calculated by Lilian.

    dg100-new uses the raw transcription data. This allows you to calculate PY
    and AP yourself.

    dg400 also uses raw transcription data.
    """
    cwd = '/home/jorgsk/Dropbox/phdproject/hsuVitro/'

    # Selecting the dataset you want to use
    if dataset == 'dg100':
        path = cwd + 'sequence_data/Hsu/csvHsu'
        ITSs = PYHsu(path)  # Unmodified Hsu data
        ITSs = add_AP(ITSs)

    elif dataset == 'dg100-new':
        path = cwd + 'Hsu_original_data/2006/2013_email'
        files = {'1122_first':  'quant1122.csv',
                 '1207_second': 'quant1207.csv',
                 '1214_third':  'quant1214.csv'}

        ITSs = read_raw(path, files, dset='dg100')

    elif dataset == 'dg400':
        path = cwd + 'prediction_experiment/raw_data'
        files = {'16_first':  'quant16_raw.csv',
                 '27_second': 'quant27_raw.csv',
                 '23_first':  'quant23_raw.csv'}

        ITSs = read_raw(path, files, dset='dg400')

    else:
        print('Provide valid dataset input to ReadData!')
        return 0

    # Make into a list for backwards compatability and sort the ITSs
    from operator import attrgetter as atrb
    #ITSs = sorted([obj for obj in ITSs.values()], key=atrb('name'))
    ITSs = sorted([obj for obj in ITSs.values()], key=atrb('PY'))

    return ITSs
