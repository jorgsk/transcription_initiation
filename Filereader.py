""" Program to read csv text data and return it to th program the way you want
it. Prior to this reading, the data should be properly saved in a csv file using
Formatfixer.py
"""
def run_from_ipython():
    try:
        __IPYTHON__active
        return True
    except NameError:
        return False

if run_from_ipython():
    #from IPython.Debugger import Tracer
    from IPython.core.debugger import Tracer
    debug = Tracer()
else:
    def debug():
        pass

import csv
import numpy as np
import os
# My own modules
import Workhouse

homedir = os.getcwd()

csv.register_dialect('jorgsk', delimiter='\t', quotechar="'",
                     quoting=csv.QUOTE_NONNUMERIC)

def AbortiveP():
    """ Import Hsu's abortive probabilities. Return dictionary with key
    promotername, which again contains a dict with keys 'Mean' and 'Std'.

    The data is not the same as in the 2006 paper.
    """
    fr = open('sequence_data/Hsu/abortiveProbabilities.csv','rt')
    reader = csv.reader(fr, delimiter='\t')
    temp = [line for line in reader]
    # Putting zeros where temp has empty string
    for row in temp[2:]:
        for ind, element in enumerate(row):
            if element == '':
                row[ind]=0
    tamp = np.array(temp)
    tamp = np.transpose(tamp)
    apDict = dict([(promoter,dict()) for promoter in temp[0][1::2]])
    # Even columns in tamp[1:] have the promoter name (even = 0,2,4,..). The
    # next carries the std and is reached by ind+1
    ost = tamp[1:]
    for ind, column in enumerate(ost):
        if not np.mod(ind,2): # This is an even column
            apDict[column[0]]['Mean'] = \
            np.array(Workhouse.StringOrFloat(column[2:-1]))/100
            apDict[column[0]]['Std'] = \
            np.array(Workhouse.StringOrFloat(ost[ind+1][2:-1]))/100
            apDict[column[0]]['FL+std'] = (float(column[-1]),
                                           float(ost[ind+1][-1]))
    return apDict

# Labels of CSV file ['Name','Sequence','PY','PYst','RPY','RPYst','RIF','RIFst','APR','MSAT','R']

def Rahmi104(adapt):
    """ Read Rahmi's 104 sequences and return as dictionary. """
#    Read the fixed Rahmi csv data and return a dict. 
    f = open(homedir+'/sequence_data/Rahmi/full_sequences_Fixed_for_all_FINAL','rt')
    # In FINAL I have changed some of the '-' values for induced/uninduced
    a = csv.reader(f, delimiter='\t')
    dicie = dict()
    for line in a:
        dicie[line[0]] = dict()
        dicie[line[0]]['Induced'] = float(line[1])
        dicie[line[0]]['Uninduced'] = float(line[2])
        sequence = line[3].replace('-','') # removing nucleotide deletion markers
        dicie[line[0]]['Sequence'] = sequence

    # adapt to the format of Hsu
    if adapt:
        lazt = []
        for name, subdict in dicie.items():

            # make entry in format accepted by downstream analysis
            entry = []
            entry.append(name)
            entry.append(subdict['Sequence'][:20].replace('U', 'T'))
            entry.append(subdict['Induced'])
            entry = entry + [1 for i in range(9)]

            lazt.append(entry)

        return lazt

    else:
        return dicie

def MinusTen():
    """ Reading the minus 10 elements of E. coli K12 as found on regulonDB. """
    fread = open('sequence_data/minusTens.csv','rt')
    min10s = []
    for row in fread.readlines():
        min10s.append(row.rstrip())
    return min10s

# Only run if used as standalone program
if __name__ == '__main__':
    Fried()

