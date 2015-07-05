"""
Do a quick analysis of the abortive and full length transcript amounts.
"""

class Quant(object):
    """
    Hold the quantification objects
    """

    def __init__(self, name, FL, AB, PY):
        self.name = name
        self.FL = float(FL)
        self.AB = float(AB)
        self.PY = float(PY)

    def __repr__(self):
        return "{0}, PY: {1}".format(self.name, self.PY)

file1 = 'summary_quant_first.csv'
file2 = 'rna_quant_summary_second_quant.csv'

f1_info = {}
f2_info = {}

for filepath, filedict in [(file1, f1_info), (file2, f2_info)]:

    for line in open(filepath, 'rb'):
        if line.split() == []:
            continue
        else:
            info = line.split()
            filedict[info[0]] = info[1:]

quant1_obj = {}

for name, fl, ab, py in zip(f1_info['Promoter'], f1_info['FL'], f1_info['Ab'],
                            f1_info['%PY']):
    quant1_obj[name] = Quant(name, fl, ab, py)

quant2_obj = {}
for name, fl, ab, py in zip(f2_info['Promoter'], f1_info['FL'], f1_info['Ab'],
                            f1_info['%PY']):
    quant2_obj[name] = Quant(name, fl, ab, py)


# plot abortive vs abortive and full length vs full length

names = quant1_obj.keys()

fl1 = [quant1_obj[name].FL for name in names]
ab1 = [quant1_obj[name].AB for name in names]
py1 = [quant1_obj[name].PY for name in names]

fl2 = [quant2_obj[name].FL for name in names]
ab2 = [quant2_obj[name].AB for name in names]
py2 = [quant2_obj[name].PY for name in names]

from matplotlib import pyplot as plt

#plt.scatter(fl2, py2)
plt.scatter(py1, py2)
plt.show()
