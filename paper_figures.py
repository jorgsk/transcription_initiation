"""
New, class based attempt at getting some rigour into creating figures for your
papers. You need more flexibility in terms of subfiguring and not re-running
simulations just to change figure data.

A de-coupling of data generation and figure creation.
"""
from __future__ import division

########### Make a debug function ##########
from ipdb import set_trace as debug  # NOQA

# Another debugger?
#from pudb import set_trace

import data_handler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import optim
import os
import cPickle as pickle  # NOQA
import copy
from operator import attrgetter
from scipy.stats import spearmanr, pearsonr, nanmean, nanstd, nanmedian  # NOQA
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy import optimize
import numpy.ma as ma

# Global variables :)
#here = os.getcwd()  # where this script is executed from! Beware! :)
here = os.path.abspath(os.path.dirname(__file__))  # where this script is located
fig_dir1 = os.path.join(here, 'figures')
fig_dir2 = '/home/jorgsk/Dropbox/The-Tome/my_papers/rna-dna-paper/figures'
fig_dirs = (fig_dir1, fig_dir2)


class Fit(object):

    def linear_function(self, B, x):
        """
        Linear fit -- for comparison
        """
        return B[0]*x + B[1]

    def linear_error(self, B, x, y):
        """
        """
        return y - self.linear_function(B, x)

    def sigmoid_function(self, B, x):
        """
        B is the parameters and x is the SE_15
        """
        #return B[0]/(B[1]+B[2]*np.exp(-x))
        return (B[1]+B[2]*np.exp(-B[3]*x))/B[0]

    def sigmoid_error(self, B, x, y):
        """
        """
        return y - self.sigmoid_function(B, x)

    def exponential_function(self, B, x):
        """
        B is the parameters and x is the SE_15
        """

        #return B[0] + B[1]*np.exp(B[2]*x)
        #return B[0] + B[1]/np.exp(B[2]*x)
        return B[1]/np.exp(B[2]*x)

    def exponential_error(self, B, x, y):
        """
        """
        return y - self.exponential_function(B, x)


class SimulationResult(object):
    """
    Holds the data from each simulation.
    """

    def __init__(self):
        self.name = ''
        self.parameters = {}
        self.inData = {}
        self.outData = {}

        self.simulated = False


class Calculator(object):
    """
    Performs the calculations needed for the plotter. This ensures that
    plotting and value-calculating can be run separately.

    Each calculator class has a corresponding plotting class
    """

    def __init__(self, dg100, dg400, calcName, testing, dset, coeffs, topNuc):

        self.coeffs = coeffs
        self.topNuc = topNuc
        # horrible thing
        # you want the ability to let dg100 and dg400 both be dset, but you
        # also want dg100 and dg400 to always be available! how horrible. you
        # should solve this by sending an extra keyword 'dgxxx' to the method
        # in question, but now's not the time
        if dset:
            self.ITSs = dset
        else:
            self.ITSs = dg100
        self.dg100 = dg100
        self.dg400 = dg400
        self.testing = testing
        self.name = calcName
        self.results = None

    def calc(self):
        """
        Run simulations according to calc.name
        """

        # call the method by its name
        methodByName = getattr(Calculator, self.name)
        self.results = methodByName(self)

        return self.results

    def AP_vs_Keq(self):

        # you're modifying it so make a copy
        dset = copy.deepcopy(self.dg100 + self.dg400)
        #dset = copy.deepcopy(self.dg100)

        c1, c2, c3 = self.coeffs
        for its in dset:
            its.calc_keq(c1, c2, c3)

        movSize = 0  # effectively NOT a moving winding, but nt-2-nt comparison

        # Ignore 2, since it's just AT and very little variation occurs.
        xmers = range(2, 21)
        # the bar height will be the correlation with the above three values for
        # each moving average center position

        result = {}
        for norm in ['Non-Normalized', 'Normalized']:
            # in the second loop
            if norm == 'Normalized':
                normalize_AP(dset)

            bar_height = []
            pvals = []

            for mer_center_pos in xmers:

                attr = 'abortiveProb'
                movAP = get_movAv_array(dset, center=mer_center_pos,
                        movSize=movSize, attr=attr, prePost='both')

                attr = 'keq'
                movKeq = get_movAv_array(dset, center=mer_center_pos,
                        movSize=movSize, attr=attr, prePost='both')

                corr, pval = spearmanr(movAP, movKeq)

                bar_height.append(corr)
                pvals.append(pval)

            result[norm] = xmers, bar_height, pvals, movSize

        return result

    def AP_Keq_examples(self):
        """
        Do AP and Keq calculations for one DG100, one DG400, and average for
        DG100 and DG400

        Return dict with keys as plot indices
        """

        # calculate keq values using latest parameter values
        c1, c2, c3 = self.coeffs

        # calc keq for all
        for its in self.dg100 + self.dg400:
            its.calc_keq(c1, c2, c3)

        output = {}
        # How to return in a way that is easy for plotting?
        # The whole dictionary approach is getting tedious.
        # Can you use tuples as keys? yes!
        dg1xx = 'DG164'
        dg4xx = 'DG444'
        dg100Obj = [d for d in self.dg100 if d.name == dg1xx][0]
        dg400Obj = [d for d in self.dg400 if d.name == dg4xx][0]

        row = 0  # AP
        for col in range(4):
            key = (row, col)
            if col == 0:
                data = dg100Obj.abortiveProb
                data_std = dg100Obj.abortiveProb_std

                output[key] = (data, data_std, dg1xx, 'AP')

            if col == 1:
                data = dg400Obj.abortiveProb
                data_std = dg400Obj.abortiveProb_std

                output[key] = (data, data_std, dg4xx, 'AP')

            if col == 2:
                data = np.mean([d.abortiveProb for d in self.dg100], axis=0)
                data_std = np.std([d.abortiveProb for d in self.dg100], axis=0)

                output[key] = (data, data_std, 'DG100 library', 'AP')

            if col == 3:
                data = np.mean([d.abortiveProb for d in self.dg400], axis=0)
                data_std = np.std([d.abortiveProb for d in self.dg400], axis=0)

                output[key] = (data, data_std, 'DG400 library', 'AP')

        row = 1  # Keq
        for col in range(4):
            key = (row, col)
            if col == 0:
                data = dg100Obj.keq
                data_std = []

                output[key] = (data, data_std, dg1xx, 'Keq')

            if col == 1:
                data = dg400Obj.keq
                data_std = []

                output[key] = (data, data_std, dg4xx, 'Keq')

            if col == 2:
                data = np.mean([d.keq for d in self.dg100], axis=0)
                data_std = np.std([d.keq for d in self.dg100], axis=0)

                output[key] = (data, data_std, 'DG100 library', 'Keq')

            if col == 3:
                data = np.mean([d.keq for d in self.dg400], axis=0)
                data_std = np.std([d.keq for d in self.dg400], axis=0)

                output[key] = (data, data_std, 'DG400 library', 'Keq')

        return output

    def cumulativeAbortive(self):
        """
        Calculate the cumulative amount of abortive product

        TODO update with flexible dg100 dg400
        """

        cumlRawProd = []
        cumlAP = []
        for inx in range(2, 21):
            cumlRawProd.append(sum(sum([its.rawDataMean[:inx] for its in self.ITSs])))
            cumlAP.append(sum(sum([its.abortiveProb[:inx] for its in self.ITSs])))
            #cumlRawProd.append(sum(sum([its.rawDataMean[2:inx] for its in self.ITSs])))
            #cumlAP.append(sum(sum([its.abortiveProb[2:inx] for its in self.ITSs])))

        return cumlRawProd, cumlAP

    def sumAP(self):
        """
        Return dict {dg100/400: PY, sum(Keq), sum(AP), sum(APnorm)}

        """
        outp = {
                'dg100': {'PY':[], 'keq': [], 'sumAP': [], 'sumAPnorm': []},
                'dg400': {'PY':[], 'keq': [], 'sumAP': [], 'sumAPnorm': []}
                }

        # calculate keq values using latest parameter values
        c1, c2, c3 = [0.13, 0, 0.93]

        for dset, name in [(self.dg100, 'dg100'), (self.dg400, 'dg400')]:
            for its in dset:
                its.calc_keq(c1, c2, c3)
                outp[name]['keq'].append(sum(its.keq))
                outp[name]['PY'].append(its.PY)
                apS = sum([v for v in its.abortiveProb if v > 0])
                outp[name]['sumAP'].append(apS)
                apSnorm = sum([v for v in its.abortiveProb/apS if v > 0])
                outp[name]['sumAPnorm'].append(apSnorm)

        return outp

    def delineateCombo(self):
        """
        See the effect of combining DG3N, DGRNADNA, DGDNADNA
        """
        import itertools

        freeEns = ['DNA', 'RNA', '3N']

        # get all unique combinations of the free energies
        # skip the single-combinations (just 'RNA' f.ex)
        combos = []
        rawCombos = [set(_) for _ in itertools.product(freeEns, repeat=3)]
        for c in rawCombos:
            if c in combos or len(c)==1:
                continue
            else:
                combos.append(c)

        # create a list of true/false dicts based on the combos
        varCombinations = []
        for c in combos:
            combDict = {}
            for en in freeEns:
                if en in c:
                    combDict[en] = True
                else:
                    combDict[en] = False
            varCombinations.append(combDict)

        # for testing; only all three true
        hei = []
        for v in varCombinations:
            if np.all(v.values()):
                hei.append(v)
        varCombinations = hei

        varCombinations = [v for v in varCombinations]
        # set the range for which you wish to calculate correlation
        itsMin = 2
        itsMax = 20
        itsRange = range(itsMin, itsMax+1)
        if self.testing:
            #itsRange = [itsMin, 5, 10, 15, itsMax]
            #itsRange = [itsMin, 3, 4, 5, 7, 10, 12, 15, 17, itsMax]
            itsRange = range(itsMin, itsMax+1)

        analysis = 'Normal'

        delinResults = {}

        for combo in varCombinations:
            print('\n---- delineate Combo -------')

            comboKey = '_'.join([str(combo[v]) for v in freeEns])

            result, grid_size, aInfo = optimizeParam(self.ITSs, itsRange,
                    self.testing, analysis=analysis, variableCombo=combo)

            onlySignCoeff=0.05
            # Print some output and return one value for c1, c2, and c3
            print(combo)
            c1, c2, c3 = get_print_parameterValue_output(result, itsRange,
                    onlySignCoeff, analysis, self.name, grid_size, aInfo=aInfo,
                    combo=combo)

            # add the constants to the ITS objects and calculate Keq
            for its in self.ITSs:
                its.calc_keq(c1, c2, c3)

            # calculate the correlation with PY
            corr, pvals = self.correlateSE_PY(itsRange)

            delinResults[comboKey] = [itsRange, corr, pvals]
            print('\n-----------')

        return delinResults

    def delineate(self):
        """
        See the individual effect of DG3N, DGRNADNA, DGDNADNA
        """

        delineateResultsOrder = ['DNA', 'RNA', '3N']

        # include the 3 variables sequentially
        varCombinations = [{'DNA':False, 'RNA':False, '3N':True},
                           {'DNA':True,  'RNA':False, '3N':False},
                           {'DNA':False, 'RNA':True,  '3N':False}]

        # set the range for which you wish to calculate correlation
        itsMin = 2
        itsMax = 20
        itsRange = range(itsMin, itsMax+1)
        if self.testing:
            #itsRange = [itsMin, 5, 10, 15, itsMax]
            itsRange = [itsMin, 3, 4, 5, 7, 10, 12, 15, 17, itsMax]
            itsRange = range(itsMin, itsMax+1)

        analysis = 'Normal'

        delinResults = {}

        for combo in varCombinations:
            print('\n---- delineate -------')

            comboKey = '_'.join([str(combo[v]) for v in delineateResultsOrder])

            result, grid_size, aInfo = optimizeParam(self.ITSs, itsRange, self.testing,
                    analysis=analysis, variableCombo=combo)

            onlySignCoeff=0.05
            # Print some output and return one value for c1, c2, and c3
            print(combo)
            c1, c2, c3 = get_print_parameterValue_output(result, itsRange,
                    onlySignCoeff, analysis, self.name, grid_size, aInfo=aInfo,
                    combo=combo)

            # add the constants to the ITS objects and calculate Keq
            for its in self.ITSs:
                its.calc_keq(c1, c2, c3)

            # calculate the correlation with PY
            corr, pvals = self.correlateSE_PY(itsRange)
            print('Correlation and p-values:')
            print(zip(corr, pvals))
            print('\n-----------')

            delinResults[comboKey] = [itsRange, corr, pvals]

        return delinResults

    def correlateSE_PY(self, itsRange):
        """
        Return three lists: indices, corr-values, and p-values
        Correlation between SE_n and PY
        """

        PY = [i.PY for i in self.ITSs]

        corr = []
        pvals = []

        # keq[0] corresponds to 2-nt
        # therefore, ind=2 -> sum(i.keq[:1]) = sum(i.keq[0])
        for ind in itsRange:

            SEn = [sum(i.keq[:ind-1]) for i in self.ITSs]

            co, pv = spearmanr(SEn, PY)

            corr.append(co)
            pvals.append(pv)

        return corr, pvals

    def dg400_validation(self):
        """
        Calculate the SE_n and correlate them with the experimentally measured
        SE_n values for the DG400 library.
        """

        if self.coeffs:
            c1, c2, c3 = self.coeffs

        else:
            print("omg no!")

        c1, c2, c3 = self.coeffs
        # add the constants to the ITS objects and calculate Keq
        for its in self.dg400:
            its.calc_keq(c1, c2, c3)

        # get PY and PY standard deviation
        PY = [i.PY for i in self.dg400]
        PYstd = [i.PY_std for i in self.dg400]

        # Get the SE15
        SE15 = [sum(i.keq[:15]) for i in self.dg400]

        # write all of this to an output file
        write_py_SE15(self.dg400)

        return self.dg400, SE15, PY, PYstd

    def PYvsSE(self):
        """
        Correlate PY with SE and return corr, pval, and indx for plotting

        topNuc is the 'best' correlating nucleotide position
        """

         #set the range of values for which you wish to calculate correlation
        itsMin = 2
        itsMax = 20
        itsRange = range(itsMin, itsMax+1)

        #if self.testing:
            #itsRange = [itsMin, 5, 10, 15, itsMax]

        if self.coeffs:
            c1, c2, c3 = self.coeffs
        else:
            analysis = 'Normal'
            onlySignCoeff=0.05
            result, grid_size, aInfo = optimizeParam(self.ITSs, itsRange,
                    self.testing, analysis=analysis)
            c1, c2, c3 = get_print_parameterValue_output(result, itsRange,
                    onlySignCoeff, analysis, self.name, grid_size, aInfo=aInfo)

        # add the constants to the ITS objects and calculate Keq
        for its in self.ITSs:
            its.calc_keq(c1, c2, c3)

        corr, pvals = self.correlateSE_PY(itsRange)
        indx = itsRange

        # print the results for each nucleotide
        for combo in zip(indx, corr, pvals):
            print combo

        # also return PY and PY std, and SEmax (usually SE20)
        PYs = [i.PY for i in self.ITSs]
        PYstd = [i.PY_std for i in self.ITSs]

        # the the highest nt you tested (probably 20)
        ntMax = max(indx)
        SEmax = [sum(i.keq[:ntMax+1]) for i in self.ITSs]
        SEbest = [sum(i.keq[:self.topNuc]) for i in self.ITSs]

        for (ind, cor) in zip(indx, corr):
            print ind, cor

        return self.topNuc, indx, corr, pvals, PYs, PYstd, SEmax, SEbest

    def crossCorrRandom(self, testing=True):
        """
        Cross-correlation and randomization of ITS sequences: effect on SE_n-PY
        correlation.
        """

        # set the range of values for which you wish to calculate correlation
        itsRange = range(2, 21)

        if self.testing:
            #itsRange = [5, 7, 10, 13, 15, 18, 20]
            itsRange = range(2, 21)
            #itsRange = [14]

        analysis2stats = {}

        #for analysis in ['Normal', 'Random', 'Cross Correlation']:
        for analysis in ['Cross Correlation']:

            result, grid_size, aInfo = optimizeParam(self.ITSs, itsRange,
                    testing=self.testing, analysis=analysis)

            # CrossCorr and Normal. How is this printed? How is this plotted?
            corr_stds = [r[1].corr_std for r in sorted(result.items())]
            # corr_stds corresponds to itsRange
            indx = itsRange

            # if normal, pick the median c1, c2, c3 values as for Figure 2
            if analysis == 'Normal':
                # Print some output and return one value for c1, c2, and c3
                onlySignCoeff = 0.05
                c1, c2, c3 = get_print_parameterValue_output(result, itsRange,
                                onlySignCoeff, analysis, self.name, grid_size,
                                aInfo=aInfo)

                # add the constants to the ITS objects and calculate Keq with
                # the new c1, c2, and c3 mean values
                for its in self.ITSs:
                    its.calc_keq(c1, c2, c3)

                corr, pvals = self.correlateSE_PY(itsRange)

            # if random or cross-correlation, pick the average of the best
            # results for each nucleotide
            else:
                # I considered the median for correlation coefficient -- it
                # does improve the values slightly, and the distribution is
                # slightly skewed so I can justify using it. I'll have a look.
                corr = [r[1].corr_mean for r in sorted(result.items())]
                #pvals = [r[1].pvals_mean for r in sorted(result.items())]
                pvals = [r[1].pvals_median for r in sorted(result.items())]
                # the median is a much better measure for pvals!!! (at least
                # for cross corr)

                onlySignCoeff = 0.95  # doesn't make sense really
                c1, c2, c3 = get_print_parameterValue_output(result, itsRange,
                                onlySignCoeff, analysis, self.name, grid_size,
                                aInfo=aInfo)

            analysis2stats[analysis] = [indx, corr, corr_stds, pvals]

        return analysis2stats


class Plotter(object):
    """
    Object that deals with plotting. Knows about the plotting parameters
    (ticks, number of axes) and the figure and axes objects.

    Contains routines for plotting onto the axes. These routines are the plots
    that enter the paper.
    """

    def __init__(self, YaxNr=1, XaxNr=1, shareX=False, shareY=False,
                    plotName='MyPlot', p_line=True, labSize=12, tickLabelSize=10,
                    lineSize=2, tickLength=3, tickWidth=1):

        # squeeze=False guarantees that axes will be a 2d array
        self.figure, self.axes = plt.subplots(YaxNr, XaxNr, squeeze=False,
                                              sharex=shareX, sharey=shareY)

        self.plotName = plotName
        self.labSize = labSize
        self.tickLabelSize = tickLabelSize
        self.lineSize = lineSize
        self.tickLength = tickLength
        self.tickWidth = tickWidth

        # draw a pvalue=0.05 line
        self.p_line = p_line

        # number of sublots in the x and y direction
        self.YaxNr = YaxNr
        self.XaxNr = XaxNr
        self.nrSubplots = YaxNr*XaxNr  # convenience value

        # keep track of the current plot position
        self._nextXaxNr = 0
        self._nextYaxNr = 0

        self._thisXaxNr = 0
        self._thisYaxNr = 0

    def setFigSize(self, xCm, yCm):

        widthInch = mm2inch(xCm*10)
        heightInch = mm2inch(yCm*10)

        self.figure.set_size_inches(widthInch, heightInch)

    def getNextAxes(self):
        """
        Return the next axes. The axes are cycled through from top to down,
        from left to right. The axes for a 2X3 subplot would be returned in the
        order x_1y_1, x_1y_2, x_2y_1, x_2y_2, x_3y_1, x_3y_2
        """
        #set_trace()

        # assume last axis has been reached
        provideAx = False

        # test if there is room for more subplots. Special cases of the test if
        # one axis has only 1 entry
        if (self.XaxNr == 1) and (self._nextYaxNr < self.YaxNr):
            provideAx = True
        if (self.YaxNr == 1) and (self._nextXaxNr < self.XaxNr):
            provideAx = True

        # general case: more than 1 entry in both directions
        if self._nextXaxNr*self._nextYaxNr <= self.nrSubplots:
            provideAx = True

        if provideAx:
            # update this axis nr
            self._thisXaxNr = self._nextXaxNr
            self._thisYaxNr = self._nextYaxNr

            nextAx = self.axes[self._thisYaxNr, self._thisXaxNr]
        else:
            print('No more subplots will fit on these axes!! >:|')

        # update the count for obtaining the next axis element
        # this proceedure selects elements column by column
        if self._nextYaxNr < (self.YaxNr-1):
            self._nextYaxNr += 1
        else:
            self._nextYaxNr = 0
            self._nextXaxNr +=1

        return nextAx

    def addLetters(self, letters=('A', 'B'), positions=('UL', 'UL'), shiftX=0,
            shiftY=0):
        """
        letters = ('A', 'B') will add A and B to the subplots
        """

        xy = {'x': {'UL': 0.03, 'UR': 0.85},
              'y': {'UL': 0.97, 'UR': 0.97}}

        # reset the subplot-counter
        self._nextXaxNr = 0
        self._nextYaxNr = 0

        for pos, label in zip(positions, letters):
            ax = self.getNextAxes()  # for the local figure
            ax.text(xy['x'][pos] + shiftX,
                    xy['y'][pos] + shiftY,
                    label, transform=ax.transAxes, fontsize=12,
                    fontweight='bold', va='top')

    ##### Below: functions that produce plots | above: helper functions #####

    def AP_Keq_examplesPlot(self, results):
        """
        """

        for indices, (data, data_std, descr, PYorKeq) in results.items():

            ax = self.axes[indices]

            # plot error bars if present
            if data_std != []:
                ax.bar(range(2, len(data)+2), data, yerr=data_std,
                        align='center')
            else:
                ax.bar(range(2, len(data)+2), data, align='center')

            # set x labels
            if indices[0] == 1:
                ax.set_xlabel('ITS nucleotide')

            # set y labels
            if indices[1] == 0:
                if PYorKeq == 'AP':
                    ax.set_ylabel('AP')
                elif PYorKeq == 'Keq':
                    ax.set_ylabel('Keq $(kcal/mol)$')

            # set title
            if indices[0] == 0:
                ax.set_title(descr)

            # set xlim
            ax.set_xlim(1, 21.5)

            # set yticks and ylim
            # if AP
            if PYorKeq == 'AP':
                ax.set_ylim(0, 0.71)

                if indices[1] == 0:
                    # y ticks
                    yticks = np.arange(0, 0.8, 0.2)
                    ax.set_yticks(yticks)
                else:
                    ax.set_yticklabels([])

            # if Keq
            if PYorKeq == 'Keq':
                ax.set_ylim(0, 3.1)

                if indices[1] == 0:
                    # y ticks
                    yticks = np.arange(0, 5, 1)
                    ax.set_yticks(yticks)
                else:
                    ax.set_yticklabels([])

            # set grid
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)

            # make sure grid falls under bars
            ax.axes.set_axisbelow(True)

    def moving_average_ap_keq(self, results, xlab=True, xticks=True):
        """
        Moving average between AP and Keq

        What is the interpretation? If Keq is high in a given area the AP tends to
        be high as well. There is a correlation in the sense that several high/low
        Keq in a row leads to high/low AP for the corresponding x-mers.

        What does it mean? How does translocation for length x RNA affect the
        abortive probability of length x+/- 1/2 RNA? There could be nonlinear
        effects of backtracking; backtracking could affect

        Cofounder: some correlation between sum(AP) and PY. However, the
        correlation between moving window Keq and AP remains after normalization of
        AP to remove the PY correlation.

        However, I could argue that the reason for AP - PY correlation goes through
        the Keq.
        """
        xmers, bar_height, pvals, movSize = results

        # get a moving average window for each center_position
        # calculating from an x-mer point of view
        Q = 0.05  # minimum false discovery rate
        # Use a less conservative test for the paper

        # Add colors to the bar-plots according to the benjamini hochberg method
        # sort the p-values from high to low; test against plim/

        colors = benjami_colors(pvals, Q=Q, nonsigncol='gray', signcol='orange')

        # plotit
        ax = self.getNextAxes()  # for the local figure
        ax.bar(left=xmers, height=bar_height, align='center', color=colors)
        ax.set_xticks(xmers)
        ax.set_xlim(xmers[0]-1, xmers[-1]+1)

        # labels
        if xlab:
            ax.set_xlabel('ITS position', size=self.labSize)
        ax.set_ylabel('Correlation: Keq and AP', size=self.labSize)

        # ticks
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)

        # X: show only positions with significant positions or 5, 10, 15, 20
        xIndx = []
        for inx, colr in enumerate(colors, start=2):
            if (inx%2 == 0):
                xIndx.append(str(inx))
            else:
                xIndx.append('')

        if xticks:
            ax.set_xticklabels(xIndx)
        else:
            ax.set_xticklabels([])

        # Y: show every other tick with a label
        yIndx = [g for g in np.arange(-0.2, 0.8, 0.2)]
        ax.set_yticks(yIndx)
        ax.set_ylim(-0.12, 0.63)

        return ax

    def sumAP_SE20(self, results, norm=True, xlab=True, xticks=True):

        ax = self.getNextAxes()

        keq = results['dg100']['keq'] + results['dg400']['keq']
        #keq = results['dg400']['keq']
        if norm:
            sumAP = results['dg100']['sumAPnorm'] + results['dg400']['sumAPnorm']
            #sumAP = results['dg400']['sumAPnorm']
        else:
            sumAP = results['dg100']['sumAP'] + results['dg400']['sumAP']
            #sumAP = results['dg400']['sumAP']

        # make into fake percentage
        sumAP = [s*100 for s in sumAP]
        ax.scatter(keq, sumAP, c='gray')
        print('Spearmanr keq and sum AP')
        if norm:
            print('Normalized')
        else:
            print('Not normalized')
        #print pearsonr(keq, sumAP)
        print spearmanr(keq, sumAP)

        # labels
        if xlab:
            ax.set_xlabel('SE$_{20}$ $(kcal/mol)$', size=self.labSize)
        ax.set_ylabel('Sum of abortive probabilities ($\%$)', size=self.labSize)

        # ticks
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)

        if not xticks:
            ax.set_xticklabels([])

        if norm:
            yrang = np.arange(0.996, 1.004, 0.002)
        else:
            yrang = np.arange(50, 500, 100)
            ax.set_yticks(yrang)
            ax.set_ylim(70, 460)

        y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(y_formatter)

    def normalizedApPlot(self, results):

        axL = self.getNextAxes()
        axR = self.getNextAxes()

        keq100 = results['dg100']['keq']
        sumAP100 = results['dg100']['sumAP']
        sumAP100norm = results['dg100']['sumAPnorm']
        print('DG100: Correlation sum(AP) and SE_20')
        print(spearmanr(keq100, sumAP100))

        keq400 = results['dg400']['keq']
        sumAP400 = results['dg400']['sumAP']
        sumAP400norm = results['dg400']['sumAPnorm']
        print('DG400: Correlation sum(AP) and SE_20')
        print(spearmanr(keq400, sumAP400))

        print('Correlation for dg100 and dg400')
        print(spearmanr(keq100+keq400, sumAP100+sumAP400))

        axL.scatter(keq100, sumAP100, c='b')
        axL.scatter(keq400, sumAP400, c='g')
        axL.set_xlabel('SE$_{20}$ $(kcal/mol)$')
        axL.set_ylabel('Sum AP')

        axR.scatter(keq100, sumAP100norm, c='b')
        axR.scatter(keq400, sumAP400norm, c='b')
        axL.set_xlabel('SE$_{20}$ $(kcal/mol)$')
        axL.set_ylabel('Sum AP')

    def delineatorComboPlot(self, delineateResults):
        """
        delineateResults is a double dictionary
        Double dictionary [FalseFalseTrue][name] -> [indx, corr, pvals]
        where the the three lists are the indices correlation and pvaleus for
        correlation between PY and SE_indx

        Where the falsetrues are in the order DNA RNA 3N
        So the above example would be the keq values from optimizing only on 3N
        """

        ymin = -0.3  # correlation is always high
        ymax = 1.1

        ax = self.getNextAxes()

        colors = ['g', 'b', 'k', 'c']
        lstyle = ['--', '-', '-.',':']

        # create labels from the combinations
        enOrder = ['$\Delta{DNA-DNA}$', '$\Delta{RNA-DNA}$', '$\Delta{3N}$']

        # convert from dict-keys to plot labels. bad stuff.
        key2label = {}

        for key in delineateResults.keys():
            for inx, t in enumerate(key.split('_')):
                if t == 'True':
                    if key in key2label:
                        key2label[key] += ' + ' + enOrder[inx]
                    else:
                        key2label[key] = enOrder[inx]

        for key, label in key2label.items():
            color = colors.pop()
            ls = lstyle.pop()

            # indices, correlation coefficients, and pvalues SE - PY
            indx, corr, pvals = delineateResults[key]

            # make x-axis (same as incoming index)
            incrX = indx
            # invert the correlation for 'upward' plot purposes
            corr = [-c for c in corr]
            # check for nan in corr (make it 0)
            corr, pvals = remove_nan(corr, pvals)

            ax.plot(incrX, corr, label=label, linewidth=self.lineSize, color=color, ls=ls)

            # interpolate pvalues (x, must increase) with correlation (y) and
            # obtain the correlation for p = 0.05 to plot as a black
            p_line = False
            if p_line and (label == '$\Delta{3N}$'):
                # hack to get pvals and corr coeffs sorted
                pv, co = zip(*sorted(zip(pvals, corr)))
                f = interpolate(pv, co, k=1)
                ax.axhline(y=f(0.05), ls='--', color='r', linewidth=3)

            indxMax = indx[-1]
            xticklabels = [str(integer) for integer in range(3, indxMax)]

            #Make sure ymin has only one value behind the comma
            ymin = float(format(ymin, '.1f'))
            yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -ymax, -0.1)]

            # legend
            ax.legend(loc='upper left', prop={'size': 5}, handlelength=3.3)

            # xticks
            ax.set_xticks(range(3, indxMax))
            #ax.set_xticklabels(xticklabels)
            ax.set_xticklabels(odd_even_spacer(xticklabels, oddeven='odd'))
            ax.set_xlim(3, indxMax)
            ax.set_xlabel("RNA length, $n$", size=self.labSize)

            #  setting the tick font sizes
            ax.tick_params(labelsize=self.tickLabelSize)

            ax.set_ylabel("Correlation: PY and SE$_n$", size=self.labSize)

            ax.set_yticks(np.arange(ymin, ymax, 0.1))
            ax.set_yticklabels(odd_even_spacer(yticklabels, oddeven='odd'))
            ax.set_ylim(ymin, ymax)
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)
            ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)

    def delineatorPlot(self, delineateResults):
        """
        delineateResults is a double dictionary
        Double dictionary [FalseFalseTrue][name] -> [indx, corr, pvals]
        where the the three lists are the indices correlation and pvaleus for
        correlation between PY and SE_indx

        Where the falsetrues are in the order DNA RNA 3N
        So the above example would be the keq values from optimizing only on 3N
        """

        ymin = -0.3  # correlation is always high
        ymax = 1.1

        ax = self.getNextAxes()

        colors = ['k', 'g', 'b']
        dashes = [(1,1,1,1,1), (1,3,1,3,1,3), False]

        # convert from dict-keys to plot labels
        key2label = {
                'False_False_True': '$\Delta{3N}$',
                'True_False_False': '$\Delta{DNA-DNA}$',
                'False_True_False': '$\Delta{RNA-DNA}$',
                }

        for key, label in key2label.items():
            color = colors.pop()
            dash = dashes.pop()

            # indices, correlation coefficients, and pvalues SE - PY
            indx, corr, pvals = delineateResults[key]

            # make x-axis (same as incoming index)
            incrX = indx
            # invert the correlation for 'upward' plot purposes
            corr = [-c for c in corr]
            # check for nan in corr (make it 0)
            corr, pvals = remove_nan(corr, pvals)

            if dash:
                ax.plot(incrX, corr, label=label, linewidth=self.lineSize,
                        color=color, dashes=dash)
            else:
                ax.plot(incrX, corr, label=label, linewidth=self.lineSize,
                        color=color)

            # interpolate pvalues (x, must increase) with correlation (y) and
            # obtain the correlation for p = 0.05 to plot as a black
            p_line = False
            if p_line and (label == '$\Delta{3N}$'):
                # hack to get pvals and corr coeffs sorted
                pv, co = zip(*sorted(zip(pvals, corr)))
                f = interpolate(pv, co, k=1)
                ax.axhline(y=f(0.05), ls='--', color='r', linewidth=3)

            indxMax = indx[-1]

            #Make sure ymin has only one value behind the comma
            ymin = float(format(ymin, '.1f'))
            yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -ymax, -0.1)]

            # legend
            ax.legend(loc='upper left', prop={'size':6}, handlelength=3)

            # xticks
            ax.set_xticks(range(3, indxMax))

            xtickLabels = []
            for i in range(3, indxMax):
                if i%2 == 0:
                    xtickLabels.append(str(i))
                else:
                    xtickLabels.append('')

            ax.set_xticklabels(xtickLabels)

            ax.set_xlim(3, indxMax)
            ax.set_xlabel("RNA length, $n$", size=self.labSize)

            # setting the tick font sizes
            ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                    width=self.tickWidth)

            ax.set_ylabel("Correlation: PY and SE$_n$", size=self.labSize)

            ax.set_yticks(np.arange(ymin, ymax, 0.1))
            ax.set_yticklabels(odd_even_spacer(yticklabels, oddeven='odd'))
            ax.set_ylim(ymin, ymax)
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)
            ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                          alpha=0.5)

    def dg400validater(self, dg400, SE15, PY, PYstd, fit_function=False):

        ax = self.getNextAxes()

        print("DG400 correlation SE15 PY")
        print spearmanr(SE15, PY)

        # multiply by 100 for plot
        PY = [p*100 for p in PY]
        PYstd = [p*100 for p in PYstd]

        ax.scatter(SE15, PY, s=12, color='b', zorder=2)
        ax.errorbar(SE15, PY, yerr=PYstd, fmt=None, zorder=1)

        ########### Set figure and axis properties ############
        ax.set_xlabel('SE$_{15}$ $(kcal/mol)$', size=self.labSize)
        ax.set_ylabel('Productive yield ($\%$)', size=self.labSize)

        xmin, xmax = min(SE15), max(SE15)
        xscale = (xmax-xmin)*0.1
        ax.set_xlim(xmin-xscale, xmax+xscale)

        ymin, ymax = min(PY), max(PY)
        yscale_low = (ymax-ymin)*0.2
        yscale_high = (ymax-ymin)*0.3
        ax.set_ylim(ymin-yscale_low, ymax+yscale_high)
        ax.set_yticks(np.linspace(0, 30, 4))

        # tick parameters
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)

        # add a polynomial or sigmoid fit
        if fit_function:
            self.add_fitted_function(ax, PY, SE15)

        dg400dict = dict(((o.name, o) for o in dg400))

        specials = ['N25', 'N25/A1anti', 'DG115a', 'DG133']
        for name in specials:

            # maybe you've already skipped these
            if name not in dg400dict:
                continue

            SE15 = sum(dg400dict[name].keq[:15])
            PY = dg400dict[name].PY*100

            box_x = 20
            box_y = 10

            if name == 'N25_A1anti' or name == 'N25/A1anti':
                name = 'N25/A1anti'
                box_x = -12
                box_y = -14

            if name == 'DG133':
                box_x = 20
                box_y = 20

            if name == 'DG115a':
                box_x = 5
                box_y = -20

            colr = 'yellow'
            ax.annotate(name,
                    xy=(SE15, PY), xytext=(box_x,box_y), textcoords='offset points',
                        bbox=dict(boxstyle='round, pad=0.5', fc=colr,
                                  alpha=0.5), ha='right', va='bottom',
                        arrowprops=dict(arrowstyle='->',
                                        connectionstyle='arc3,rad=0'), size=4)

    def add_fitted_function(self, ax, PY, SE15):
        """
        Add a sigmoid fit to the plot
        """

        x = np.array(SE15)
        y = np.array(PY)

        # List with tuples of the differnt functions you fit to

        F = Fit()

        ffuncs = [('Sigmoid', F.sigmoid_error, F.sigmoid_function),
                 ('Linear', F.linear_error, F.linear_function),
                 ('Exponential', F.exponential_error, F.exponential_function)]

        mean_y = np.mean(y)
        B0 = [1, 3, 0.2, 0.01]
        #B0 = [3, 10, 0.2]
        for (fname, f_error, f_function) in ffuncs:
            fit = optimize.leastsq(f_error, B0, args=(x, y))
            outp_args = fit[0]
            fit_vals = [f_function(outp_args, x_val) for x_val in x]

            # residual sum of squares
            sst = sum([(y_1 - mean_y)**2 for (y_1, f_1) in zip(y, fit_vals)])
            # total sum of squares (proportional to variance)
            sse = sum([(y_1 - f_1)**2 for (y_1, f_1) in zip(y, fit_vals)])

            # Coefficient of determination R squared:
            Rsq = 1 - sse/sst
            print('{0}: coefficient of determination: {1}'.format(fname, Rsq))

        # Normal least squares on sigmoid fit
        #B0 = [1, 1, 1]
        ##391.535639527
        ##15.0590630587

        # Normal least squares on linear fit
        #B0 = [1, 1]
        #fit = optimize.leastsq(linear_error, B0, args=(x, y))
        #outp_args = fit[0]
        #fit_vals = [linear_function(outp_args, x_val) for x_val in x]
        ##359.388266403 total
        ##13.8226256309 averaged

        F = Fit()

        # Normal least squares on exponential fit
        B0 = [1, 40, 0.02, 0.01]
        fit = optimize.leastsq(F.exponential_error, B0, args=(x, y))
        outp_args = fit[0]
        fit_vals = [F.exponential_function(outp_args, x_val) for x_val in x]

        #B0 = [1, 3, 10, 0.01]
        #fit = optimize.leastsq(F.sigmoid_error, B0, args=(x, y))
        #outp_args = fit[0]
        #fit_vals = [F.sigmoid_function(outp_args, x_val) for x_val in x]

        # sort by increasing x-value
        (sort_x, sort_fit) = zip(*sorted(zip(x, fit_vals)))

        # residual sum of squares
        sst = sum([(y_1 - mean_y)**2 for (y_1, f_1) in zip(y, fit_vals)])
        # total sum of squares (proportional to variance)
        sse = sum([(y_1 - f_1)**2 for (y_1, f_1) in zip(y, fit_vals)])

        # Coefficient of determination R squared:
        Rsq = 1 - sse/sst

        #print('Coefficient of determination: {0}'.format(Rsq))

        # odd_ least squares
        #my_model = odr.Model(sigmoid_function)
        #my_data = odr.Data(x,y)
        #my_odr = odr.ODR(my_data, my_model, beta0=B0)
        ## fit type 2 for least squares
        #my_odr.set_job(fit_type=2)
        #fit = my_odr.run()
        # XXX I didn't get a nice fit this way, but keep code just in case

        ax.plot(sort_x, sort_fit, linewidth=2)

    def PYvsSEscatter(self, PYs, PYstd, SEmax):
        """
        Plotting just the best correlation (at 14 usually)
        """

        # the index for which corr has the highest value
        print("Spearman for ntMax (usually 20) and PY")
        print(spearmanr(SEmax, PYs))
        print(pearsonr(SEmax, PYs))

        ax = self.getNextAxes()

        # multiply by 100 for plot
        PYs = [p*100 for p in PYs]
        PYstd = [p*100 for p in PYstd]

        ax.errorbar(SEmax, PYs, yerr=PYstd, fmt=None, ecolor='b', zorder=1)
        ax.scatter(SEmax, PYs, c='b', s=12, linewidth=0.6, zorder=2)
        # XXX get the errorbars to be gray too!!!!!!!!

        ax.set_ylabel("Productive yield ($\%$)", size=self.labSize)
        ax.set_xlabel("SE$_{20}$ $(kcal/mol)$", size=self.labSize)

        ymin = -0.1

        # setting the tick number sizes
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)

        xmin, xmax = min(SEmax), max(SEmax)
        xscale = (xmax-xmin)*0.06
        ax.set_xlim(xmin-xscale, xmax+xscale)

        ymin, ymax = min(PYs), max(PYs)
        #yscale_low = (ymax-ymin)*0.03
        yscale_high = (ymax-ymin)*0.34
        ax.set_ylim(0, ymax+yscale_high)

        return ax

    def cumulativeAbortive(self, cumulRaw, cumulAp, corr):
        """
        Plot the cumulative abortive probability on top of the correlation
        ladder.
        """

        # use the current axis
        mainAx = self.axes[self._thisYaxNr, self._thisXaxNr]

        cumulRawNorm = np.array(cumulRaw)/cumulRaw[-1]
        cumulApNorm = np.array(cumulAp)/cumulAp[-1]

        #choice = 'AP'
        choice = 'raw'

        xlim = mainAx.get_xlim()
        cumlAx = mainAx.twinx()  # make a twin axis

        if choice == 'AP':
            # cumulApNorm starts with 2. make it start with 3 using [1:]
            cumlAx.plot(np.arange(xlim[0]-1, xlim[1]+1), cumulApNorm, ls='--',
                    linewidth=2, color='g')

            cumlAx.set_ylabel('Cumulative abortive probability',
                    size=self.labSize, color='green')

        else:
            cumlAx.plot(np.arange(xlim[0]-1, xlim[1]+1), cumulRawNorm, ls='-',
                    linewidth=1, color='g')
                    #linewidth=1, color='g', marker='D', markersize=3)

            cumlAx.set_ylabel('Abortive product (normalized, cumulative)',
                    size=self.labSize-1, color='green')

        # tick parameters
        cumlAx.tick_params(axis='y', labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth, colors='green')

        yrange = np.arange(0,1.1,0.1)
        yticklabels_skip = odd_even_spacer(yrange)

        cumlAx.set_yticks(yrange)
        cumlAx.set_yticklabels(yticklabels_skip)

        its_max = int(cumlAx.get_xlim()[-1])
        cumlAx.set_xticks(range(2, its_max + 1))

        xtickLabels = []
        for i in range(2, its_max+1):
            if i%5 == 0:
                xtickLabels.append(str(i))
            else:
                xtickLabels.append('')

        cumlAx.set_xticklabels(xtickLabels)

        cumlAx.set_xlim(2-0.3, its_max + 0.3)

        # correlation between correlation coeffieicnet and cumul prob?
        corr[0] = 0
        print spearmanr(np.array(corr)*(-1), cumulRawNorm)
        print('High linear correlation between increase in correlation and'
               ' cumulative probability to abort transcription initiation.'
               ' But this is not a good measure ... youll need to do bettrr ')
        print pearsonr(np.array(corr)*(-1), cumulRawNorm)

    def PYvsSEladder(self, indx, corr, pvals, PYs, SEbest, b, inset=False):
        """
        The classic ladder plot, including inset scatterplot
        """

        ax = self.getNextAxes()

        # check for nan in corr (make it 0)
        corr, pvals = remove_nan(corr, pvals)

        its_max = indx[-1]

        ## the ladder plot
        colr = 'b'
        revCorr = [c*(-1) for c in corr]  # to get an increasing correlation
        ax.plot(indx, revCorr, linewidth=self.lineSize, color=colr, marker='s',
                markersize=3)

        ## the little inset scatter plot
        if inset and b in indx:

            x = b
            y = -corr[x-2]  # adjust for starting from 2

            # add axes within axes: add_axes[left, bottom, width, height]
            axsmall = self.figure.add_axes([0.61, 0.77, 0.14, 0.16], axisbg='y')
            axsmall.scatter(PYs, SEbest, s=2)
            #axsmall.text(0.035, 2100, '$r$=-{0:.2f}'.format(y), size=10)

            axsmall.set_xlabel('SE$_{13}$', size=6)
            axsmall.set_ylabel('Productive yield ($\%$)', size=6)
            axsmall.xaxis.labelpad = 1
            axsmall.yaxis.labelpad = 1

            xlim = axsmall.get_xlim()
            ylim = axsmall.get_ylim()

            # adjust the bounding box a bit
            axsmall.set_xlim(xlim[0] + 0.016, xlim[1] - 0.016)
            axsmall.set_ylim(ylim[0] + 1.5, ylim[1] - 2)

            for l in axsmall.get_xticklines() + axsmall.get_yticklines():
                l.set_markersize(2)

            # add an arrow (annotate is better than arrow for some
            # reason)
            # XXX damn thing does not change well with general figure changes
            #ax.annotate('', xy=(x-1.7, y+0.12), xytext=[x,y+0.01],
                                 #textcoords=None,
                                 #arrowprops=dict(arrowstyle='->',
                         #connectionstyle="arc, angleA=90, armA=45, rad=20"))

            ax.scatter(b, y, s=20)

            axsmall.set_yticklabels([])
            axsmall.set_xticklabels([])

        if self.p_line:
            # sort pvals and corr and interpolate
            pv, co = zip(*sorted(zip(pvals, corr)))
            # interpolate pvalues (x, must increase) with correlation
            # (y) and obtain the threshold for p = 0.05
            f = interpolate(pv[:-2], co[:-2], k=1)
            ax.axhline(y=(-1)*f(0.05), ls=':', color='r',
                                label='p = 0.05 threshold', linewidth=2)

        ymin = -0.1
        #ymin = 0
        yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -1.1, -0.1)]
        yticklabels_skip = odd_even_spacer(yticklabels, oddeven='odd')

        ax.set_yticks(np.arange(ymin, 1.1, 0.1))
        ax.set_yticklabels(yticklabels_skip)
        ylim0 = -0.05
        ax.set_ylim(ylim0, 1.001)
        ax.set_ylabel("Correlation: PY and SE$_n$", size=self.labSize, color='blue')

        # set tick label size
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)
        ax.tick_params(axis='y', colors='blue')

        # xticks
        xtickLabels = []
        for i in range(3, its_max):
            if i%5 == 0:
                xtickLabels.append(str(i))
            else:
                xtickLabels.append('')

        ax.set_xticks(range(3, its_max))
        ax.set_xticklabels(xtickLabels)
        #ax.set_xticklabels([])
        ax.set_xlim(3, its_max)
        ax.set_xlabel("RNA length, $n$", size=self.labSize)

        # bbox_to_anchor= x, y, width, height
        ax.legend(bbox_to_anchor=(0.8, 0.1, 0.2, 0.1), loc='best',
                    prop={'size':4.5})

        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)
        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

        return ax

    def crossCorrPlot(self, analysis2stats):
        """
        Ladder-plot with random ITS and cross correlation
        """

        ax = self.getNextAxes()

        colors = ['b', 'g', 'k']

        for analysis, stats in analysis2stats.items():
            indx, corr, corr_stds, pvals = stats
            colr = colors.pop()

            # points on x-axis for plotting -- same as its_lenghts
            incrX = indx

            # change sign for correlation to get an 'upward' plot
            corr = [-c for c in corr]  # change sign

            if analysis == 'Normal':

                # check for nan in corr (make it 0)
                corr, pvals = remove_nan(corr, pvals)

                lab = 'Full DG100 dataset'
                ax.plot(incrX, corr, label=lab, linewidth=self.lineSize,
                        color=colr, marker='s', markersize=3)

                p_line = False
                if p_line:
                    # sort pvals and corr and interpolate
                    pv, co = zip(*sorted(zip(pvals, corr)))
                    # interpolate pvalues (x, must increase) with correlation
                    # (y) and obtain the threshold for p = 0.05
                    f = interpolate(pv[:-2], co[:-2], k=1)
                    ax.axhline(y=f(0.05), ls='--', color='r',
                                        label='p = 0.05 threshold', linewidth=2)

            # if random, plot with errorbars
            elif analysis == 'Random':
                lab = 'random sequences'
                ax.errorbar(incrX, corr, yerr=corr_stds, label=lab,
                                     linewidth=self.lineSize, color=colr,
                                     marker='*', markersize=3)

            elif analysis == 'Cross Correlation':
                lab = 'cross-validation'
                ax.errorbar(incrX, corr, yerr=corr_stds, label=lab,
                                     linewidth=self.lineSize, color=colr,
                                     marker='x', markersize=3)

        its_max = incrX[-1]

        ymin = -0.5
        ymax = 1.1
        # yticks
        yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -ymax, -0.1)]
        yticklabels_skip = odd_even_spacer(yticklabels, oddeven='odd')

        yticks = np.arange(ymin, ymax, 0.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels_skip)
        ax.set_ylim(ymin, 1.1)
        ax.set_ylabel("Correlation: PY and SE$_n$", size=self.labSize)

        # xticks
        xticklabels = [str(integer) for integer in range(3, its_max)]
        xticklabels_skip = odd_even_spacer(xticklabels, oddeven='odd')

        ax.set_xticks(range(3, its_max))
        ax.set_xticklabels(xticklabels_skip)
        ax.set_xlim(3, its_max)
        ax.set_xlabel("RNA length, $n$", size=self.labSize)

        ax.legend(loc='upper left', prop={'size':6})

        # setting the tick font sizes
        ax.tick_params(labelsize=self.tickLabelSize)

        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)
        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)


class AP(object):
    def __init__(self, keq, ap, dg3d=-1, dgDna=-1, dgRna=-1):
        self.keq = keq
        self.dg3d = dg3d
        self.ap = ap
        self.dgDna = dgDna
        self.dgRna = dgRna


def odd_even_spacer(vals, oddeven='even', integer=False):
    out = []
    if oddeven == 'odd':
        for val_nr, val in enumerate(vals):
            if val_nr % 2:
                if integer:
                    out.append(int(val))
                else:
                    out.append(val)
            else:
                out.append(' ')
    else:
        for val_nr, val in enumerate(vals):
            if not val_nr % 2:
                if integer:
                    out.append(int(val))
                else:
                    out.append(val)
            else:
                out.append(' ')

    return out


def remove_nan(corr, pvals):
    """
    Replace nans in corr and pval with 0 and 1. If not dont replace anything.
    """

    newcorr = []
    if True in np.isnan(corr):
        for inx, anan in enumerate(np.isnan(corr)):
            if anan:
                newcorr.append(0)
            else:
                newcorr.append(corr[inx])
    else:
        newcorr = corr

    newpval = []
    if True in np.isnan(pvals):
        for inx, anan in enumerate(np.isnan(pvals)):
            if anan:
                newpval.append(1)
            else:
                newpval.append(pvals[inx])
    else:
        newpval = pvals

    return newcorr, newpval


def mm2inch(mm):
    return float(mm)/25.4


def visual_inspection(ITSs, variable='AP'):
    """
    1) Bar plot of the abortive probabilities 3 by 3
    """

    attrs = {'AP':  ['abortiveProb', 'abortiveProb_std'],
             'Raw': ['rawDataMean',  'rawDataStd']}

    rowPlotNr = 3
    colPlotNr = 3
    plotPerFig = rowPlotNr*colPlotNr

    # make a map from nucleotide to color
    nuc2clr = {'G': 'b', 'A': 'g', 'T': 'k', 'C': 'y'}

    # nr of figures
    nr_figs = int(np.ceil(len(ITSs)/plotPerFig))

    # partition the ITSs into as many groups as there are figures
    ITSgroups = [ITSs[i*plotPerFig:i*plotPerFig+plotPerFig] for i in range(nr_figs)]

    for fig_nr, ITSgrp in enumerate(ITSgroups):

        fig, axes = plt.subplots(rowPlotNr, colPlotNr)
        rowNr = 0
        colNr = 0

        for oITS in ITSgrp:

            seq = list(oITS.sequence[1:21])
            colors = [nuc2clr[n] for n in seq]
            left_corners = range(1,21)

            # assigxn a plot and advance counter
            ax = axes[rowNr, colNr]
            y = getattr(oITS, attrs[variable][0])
            err = getattr(oITS, attrs[variable][1])
            ax.bar(left_corners, y, yerr=err, color=colors)

            ax.set_xticks([])
            ax.set_title(oITS.name)

            # reset the counter if at end of row
            if rowNr == 2:
                rowNr = 0
                colNr += 1
            else:
                rowNr += 1

        plt.show()


def write_py_SE15(dg400):

    outPath = '/home/jorgsk/Dropbox/The-Tome/my_papers/rna-dna-paper/dgtable.tex'
    outp = open(outPath, 'wb')

    for its in sorted(dg400, key=attrgetter('PY'), reverse=True):
        seq = its.sequence[:15]
        seq_split = seq[:10] + ' ' + seq[10:]
        SE15 = format(sum(its.keq[:15]), '.1f')
        PY = format(its.PY*100, '.0f')
        purine_count = seq.count('A') + seq.count('G')

#DG451 & \texttt{ATACAAGAGA AGAAT} & 67.1 & 18.7 & 12\\
#DG445 & \texttt{ATTAAAGAAC CAGGT} & 155.0 & 18.3 & 10\\
#DG448 & \texttt{ATTAGAAGAA ACAGG} & 155.0 & 16.9 & 12\\
        outp.write(' '.join(
                [
                its.name, '&',
                '\\texttt{' + seq_split + '}', '&',
                SE15, '&',
                PY, '&',
                str(purine_count)+'\\\\'
                ]
                            ) + '\n')

    print('Written to dgtable.tex')
    outp.close()


def optimizeParam(ITSs, its_range, testing=True, analysis='Normal',
        variableCombo={'DNA':True, 'RNA':True, '3N':True}):
    """
    Optimize c1, c2, c3 to obtain max correlation with PY/FL/TA/TR

    # equation:
    k1 = exp(-(c1*rna_dna_i + c2*dna_dna_{i+1} + c3*Keq_{i-1}) * 1/RT)

    As of right now, you only have PY for DG100 ... would be interesting to get
    FL in there as well. Do you have this information? Maybe verify with Lilian.

    PY = Productive Yield
    FL = Full Length
    TA = Total Abortive
    TR = Total RNA
    """

    # grid size
    grid_size = 15
    if testing:
        grid_size = 9

    # defalt: do nothing
    analysisInfo = {
            'Normal':{'run':False},
            'CrossCorr': {'run':False, 'value':0},
            'Random': {'run':False, 'value':0},
            }

    if analysis == 'Normal':
        analysisInfo['Normal']['run'] = True
    if analysis == 'Random':
        # nr of random sequences tested for each parameter combination
        analysisInfo['Random']['run'] = True
        #analysisInfo['Random']['value'] = 100
        analysisInfo['Random']['value'] = 15
    elif analysis == 'Cross Correlation':
        # nr of samplings of 50% of ITSs for each parameter combination
        analysisInfo['CrossCorr']['run'] = True
        #analysisInfo['CrossCorr']['value'] = 100
        analysisInfo['CrossCorr']['value'] = 15
    elif analysis != 'Normal':
        print('Give correct analysis parameter name')
        1/0

    # Parameter ranges you want to test out
    if variableCombo['RNA']:
        c1 = np.linspace(0, 1.0, grid_size)
    else:
        c1 = np.array([0])

    if variableCombo['DNA']:
        c2 = np.linspace(0, 1.0, grid_size)
    else:
        c2 = np.array([0])

    if variableCombo['3N']:
        c3 = np.linspace(0, 1.0, grid_size)
    else:
        c3 = np.array([0])

    par_ranges = (c1, c2, c3)

    all_results = optim.main_optim(its_range, ITSs, par_ranges, analysisInfo)

    # return results for requested analysis type ('normal' by default)
    return all_results[analysis], grid_size, analysisInfo


def get_print_parameterValue_output(results, its_range, onlySignCoeff,
        analysis, callingFunctionName, grid_size, aInfo=False, combo=''):
    """
    Analyze and print some of the output parameter values

    It's not straightforward how to calculate return parameter values c1 c2,
    c3. You have tried to average all, and average just significant values.

    """

    logDir = 'outputLog'
    if not os.path.isdir(logDir):
        os.mkdir(logDir)

    analysisNoGap = analysis.replace(' ', '_')
    logFilePath = os.path.join('outputLog',
                        '_'.join(['log', analysisNoGap, callingFunctionName,
                            str(grid_size), str(aInfo)]))
    # sometimes the same function calls, but with different combinations of the
    # free enregy variables
    if combo != '':
        logFilePath += '_' + str(combo)
    logFilePath += '.log'

    logFileHandle = open(logFilePath, 'wb')
    logFileHandle.write('calling function: ' + callingFunctionName + '\n')
    logFileHandle.write('combo: ' + str(combo) + '\n')
    logFileHandle.write('grid_size: ' + str(grid_size) + '\n')

    if aInfo:
        for asis, settings in aInfo.items():
            runThis = settings['run']
            logFileHandle.write('analysis: ' + asis + ' ' + str(runThis))
            if 'value' in settings:
                howMany = settings['value']
                logFileHandle.write(' nr: ' + str(howMany) + '\n')

    # output
    outp = {}

    minpvals = [results[pos].pvals_min for pos in its_range]
    meanpvals = [results[pos].pvals_mean for pos in its_range]
    medianpvals = [results[pos].pvals_median for pos in its_range]
    maxcorr = [results[pos].corr_max for pos in its_range]
    meancorr = [results[pos].corr_mean for pos in its_range]
    mediancorr = [results[pos].corr_median for pos in its_range]

    # print the mean and std of the estimated parameters
    for param in ['c1', 'c2', 'c3']:
        # is params_best always what I want? For Normal, yes, but Random and
        # Cross-correlation? Look into params_best bearing in mind that this is
        # the value which is output

        # for cross-corr, params_best is not the best of the params! Neither is
        # it what you want to output !!!

        # for random you again want the median value ... this is
        if analysis == 'Normal':
            parvals = [results[pos].params_best[param] for pos in its_range]
        elif analysis == 'Cross Correlation':
            parvals = [results[pos].params_median[param] for pos in its_range]
        elif analysis == 'Random':
            # perhaps the mean is best since it's random?
            parvals = [results[pos].params_median[param] for pos in its_range]

        Parvals = []
        if analysis == 'Normal':
            pvals = minpvals  # min of 20 best
        else:
            pvals = medianpvals  # median of X cross-corr, random

        for ix, pval in enumerate(pvals):
            # if set, only collect parameters corresponsing to significant
            if onlySignCoeff and (pval < onlySignCoeff):
                continue

            # don't consider rna-dna until after full hybrid length is reached
            if param == 'c1' and ix < 8:
                Parvals.append(0)
            else:
                Parvals.append(parvals[ix])

        print param, Parvals

        # ignore nan in mean and std calculations
        mean = nanmean(Parvals)
        median = nanmedian(Parvals)
        normal_std = nanstd(Parvals)
        std = mad_std(Parvals)  # std, but using the mad as an estimator

        outpString = '{0}: {1:.2f} (mean) +/- {2:.2f} or {0}: {3:.2f}'\
                        ' (median) +/- {4:.2f}'.format(param, mean, normal_std,
                                median, std)
        print(outpString)
        logFileHandle.write(analysis + '\n')
        logFileHandle.write(outpString + '\n')

        outp[param] = mean

    # print correlations
    if analysis == 'Normal':
        infoStr = "Normal analysis: max correlation and corresponding p-value"
        print(infoStr)
        logFileHandle.write(infoStr)
        print analysis

        for nt, c, p in zip(its_range, maxcorr, pvals):
            output = (nt, c, p)
            print(output)
            logFileHandle.write(str(output) + '\n')
            print nt, c, p
    else:
        infoStr = "Random or cross-correlation: mean (median) correlation and p-values"
        print(infoStr)
        logFileHandle.write(infoStr)
        print analysis
        for nt, c, p, cm, pm in zip(its_range, meancorr, meanpvals, mediancorr,
                medianpvals):
            output = (nt, c, p, cm, pm)
            logFileHandle.write(str(output) + '\n')
            print output

    logFileHandle.close()

    return outp['c1'], outp['c2'], outp['c3']


def data_overview(ITSs):
    """
    This is the deleted stuff

    Plots the FL, PY, total Abortive and total RNA for all ITSs
    """

    #hooksMean = [('PY', '_PYraw'), ('FL', 'fullLength'),
            #('Total Abortive', 'totAbort'), ('Total RNA', 'totRNA')]
    hooksMean = [('PY', '_PYraw'), ('FL', 'fullLength'),
            ('Total Abortive', 'totAbort')]

    # separate the different quantitations
    fig, axes = plt.subplots(len(hooksMean))

    names = [i.name for i in ITSs]
    xvals = range(1, len(names) + 1)

    dsets = set(sum([i.quantitations for i in ITSs], []))
    # strip the N25 controls
    quantitations = [d for d in dsets if not d.endswith('.1')]

    for pltNr, (plotTag, attribute) in enumerate(hooksMean):

        ax = axes[pltNr]

        vals = {}

        # 1 Plot the attribute value for each dataset
        #for quant in quantitations:

            #y = [getattr(i, attribute)[quant] for i in ITSs]
            #ax.plot(xvals, y, label=quant, linewidth=2, marker='o')

        # 1 Plot mean attribute value
        vals = [[getattr(i, attribute)[quant] for i in ITSs] for quant in
                quantitations]

        ymean = np.mean(vals, axis=0)
        ystd = np.std(vals, axis=0)

        ax.errorbar(xvals, ymean, label='Mean', linewidth=2, marker='o',
                yerr=ystd)

        ax.set_xticklabels(names, rotation=30)
        ax.set_xticks(xvals)

        ax.legend(loc='best')
        ax.set_title(plotTag)

    plt.tight_layout()
    plt.show()


def fold_difference(ITSs):
    """
    Establish that there is much more variation in abortive RNA than in full
    length RNA, showing that the difference in PY between different promoteres
    is driven primarily by a difference in abortive yield.
    """

    PY = [i.PY for i in ITSs]
    FL = [i.fullLengthMean for i in ITSs]
    AR = [i.totAbortMean for i in ITSs]
    SE = [i.SE for i in ITSs]

    print('Correlation PY and FL:')
    print spearmanr(PY, FL)

    print('Correlation PY and Abortive:')
    print spearmanr(PY, AR)
    print('')

    print('Correlation SE and FL:')
    print spearmanr(SE, FL)

    print('Correlation SE and Abortive:')
    print spearmanr(SE, AR)
    print('')

    max_FL = np.mean([i.fullLengthMean for i in ITSs[-3:]])
    min_FL = np.mean([i.fullLengthMean for i in ITSs[:3]])
    print('Full Length fold: {0}'.format(min_FL/max_FL))

    max_AR = np.mean([i.totAbortMean for i in ITSs[-3:]])
    min_AR = np.mean([i.totAbortMean for i in ITSs[:3]])
    print('Abortive RNA fold: {0}'.format(min_AR/max_AR))


def sortITS(ITSs, attribute='PY'):
    """
    Sort the ITSs
    """

    ITSs = sorted(ITSs, key=attrgetter(attribute))

    return ITSs


def separateByScrunch(ITSs, sortby, nr='two'):
    """
    Screen by scrunch in the 0 to 15 value

    Divide in two. High has around -19, low aroudn -24
    """

    if nr == 'three':

        scrunches = {'Low scrunch': [],
                     'High scrunch': [],
                     'Medium scrunch': []}

        low = scrunches['Low scrunch']
        high = scrunches['High scrunch']
        med = scrunches['Medium scrunch']

        dividor = len(ITSs)/3

        for nr, its in enumerate(sortITS(ITSs, sortby)):
            if nr <= dividor:
                low.append(its)
            elif dividor < nr < 2*dividor:
                med.append(its)
            else:
                high.append(its)

    elif nr == 'two':

        scrunches = {'Low scrunch': [],
                     'High scrunch': []}

        low = scrunches['Low scrunch']
        high = scrunches['High scrunch']

        dividor = len(ITSs)/2

        for nr, its in enumerate(sortITS(ITSs, sortby)):
            if nr <= dividor:
                low.append(its)
            else:
                high.append(its)

    elif nr == 'one':
        scrunches = {'All': ITSs}

    return scrunches


def getITSdnaSep(ITSs, nr='two', upto=15):
    """
    Separate the ITSs based on DGDNA in a low, mid, high group

    Can also be in a low, high group (better for publication)
    """

    scrunches = separateByScrunch(ITSs, 'DgDNA15', nr=nr)

    arranged = {}

    for lab, ITSs in scrunches.items():

        aProbs = []
        keqs = []

        fro = 2
        to = upto

        # build arrays
        for its in ITSs:
            #for (ap, keq) in zip(its.abortiveProb[fro:to], its.dg3d[fro:to]):
            for (ap, keq) in zip(its.abortiveProb[fro:to], its.keq[fro:to]):

                aProbs.append(ap)
                keqs.append(keq)

        arranged[lab] = (aProbs, keqs)

    return arranged


def shifted_ap_keq(ITSs, start=2, upto=15, plusmin=5):
    """
    Sort values with the DNA DNA value up to that point

    Optionally subtract the RNA-DNA value (length 10 RNA-DNA)

    How to do it? Make a basic class for AP! This makes it super-easy to change
    sortings: just loop through a keyword.

    Return arrays of AP and DG3N values in different batches. For example, the
    values could be split in 2 or 3. What determines the difference in
    correlation between those two groups will then be how you split the

    Maybe ... just maybe you have to do this correlation on a
    position-by-position basis and use a weighting factor to control for.
    Another method. After. Then you cover both ranges.
    """

    # add objects here and sort by score
    objects = {}

    for pm in range(-plusmin, plusmin+1):
        apobjs = []
        for its in ITSs:
            for ap_pos in range(start, upto):

                keq_pos = ap_pos + pm
                # only use keq_pos for the same range as ap_pos
                if keq_pos in range(start, upto):

                    apobjs.append(AP(its.keq[keq_pos], its.abortiveProb[ap_pos]))

        objects[pm] = apobjs

    return objects


def getDinosaur(ITSs, nr=2, upto=15, sortBy='dgDna'):
    """
    Sort values with the DNA DNA value up to that point

    Optionally subtract the RNA-DNA value (length 10 RNA-DNA)

    How to do it? Make a basic class for AP! This makes it super-easy to change
    sortings: just loop through a keyword.

    Return arrays of AP and DG3N values in different batches. For example, the
    values could be split in 2 or 3. What determines the difference in
    correlation between those two groups will then be how you split the
    """

    # add objects here and sort by score
    objects = []

    for its in ITSs:
        for pos in range(1, upto):

            # correct for a max 10 nt rna-dna hybrid
            rnaBeg = max(pos-10, 0)

            if its.abortiveProb[pos] < 0:
                continue

            #apObj = AP(sum(its.keq[pos:pos+1]),
            apObj = AP(its.keq[pos-1],
                       its.dg3d[pos],
                       its.abortiveProb[pos],
                       sum(its.dna_dna_di[:pos]),
                       sum(its.rna_dna_di[rnaBeg:pos]))

            objects.append(apObj)

    # sort objects by one of your metrices (small to large)
    if sortBy:
        objects.sort(key=attrgetter(sortBy))

    arranged = {}
    # divide the list into two/three etc
    ssize = int(len(objects)/nr)
    splitObj = [objects[i:i+ssize] for i in range(0, len(objects), ssize)]

    # XXX todo if onw group is small, add to the closest.

    for spNr, sub_objs in enumerate(splitObj):
        if spNr == 0:
            label = '1 Smallest values'
        elif spNr == nr-1:
            label = '{0} Largest values'.format(nr)
        else:
            label = '{0} Intermediate value'.format(spNr+1)

        label = label + ' ({0} objects)'.format(len(sub_objs))

        aprobs = [o.ap for o in sub_objs]
        #dg3d = [o.dg3d for o in sub_objs]
        dg3d = [o.keq for o in sub_objs]

        arranged[label] = (aprobs, dg3d)

    return arranged


def keq_ap_raw(ITSs, method='initial', upto=15):
    """
    Correlation between Keq and AP and Keq and raw reads

    Divide the population of sites in two: those with low and high 'scrunched
    energy' terms.

    What is the conclusion from these things?

    The conclusion is that there is a weak correlation.
    It seems to be located in (2,10).

    Interestingly, the raw correlation decreases with longer sequence, but not
    the AP, which is normalized.

    This actually indicates that there is a weak but present correlation.

    Furthermore, this correlation is present in both DGXXX sets :)

    OK. Can you now get this correlation to increase/decrease by 'correcting'
    etc using DGDNA DGRNA?

    Make a high scrunch and a low scrunch group

    The little scrunch has lower correlation that the high scrunch.

    This could mean that high scrunch makes an abortive release more certain
    once backtracking has occured.

    This is much more pronounced if you ignore the first 2nt band!

    How can we further increase the 0.30 correlation? Your method is crude, but
    actually pretty good. I think you'll be hard pressed finding a better
    correlation. But I think you should try.

    Try to separate each individual Keq by DNA-DNA up to that position -- first
    by absolute DGDNA then by normalization by length.

    The 'upto' parameter variation is very interesting. With upto=11 there is
    no significant correlation for the high scrunch group, but at 14 there is.
    For the low scrunch group, correlation decreases from 11 to 14.

    They have opposite behavior. However, this is getting very technical for
    people who aren't 'into it'. You'll need to simplify.

    It seems that by looking only at sites with a large dg3d and/or large AP
    you're getting a good correlation.

    Is this even more important than checking for dgdna? if so, can you combine
    them to reach a higher score? For example, you could divide the group into
    4 and compare only the top/low 25% for example.

    Either way what you have is good enough.

    How robust is it to a +/- 1 shift in ap values?

    OMG! Strangeness: the keq is strongly anticorrelated with AP-1 at the keq
    position; it is then weakly positively correlated with AP-2, before being
    zero-corrlated with AP-3.

    It is also zero correlated with ap+1 and somewhat correlated with ap+2/3/4

    How can you describe this systematically?

    Now you normalized AP -- and should ignore positions with ap < 0

    """

    #fig, ax = plt.subplots()

    #method = 'initial'
    method = 'dinosaur'
    upto = 15

    # TODO do a systematic +/- 1,2,3,4 analysis of the AP and keq to look for
    # correlation. Find out what the relationship really really means. Make
    # sure that the AP matches the keq/dg3d so you KNOW what you are comparing

    #for srtb in ['dg3d', 'ap', 'dgDna', 'dgRna', 'dgDnaNorm', 'dgRnaNorm',
            #'dgControl', 'dgControlNorm']:
    #for srtb in ['dg3d', 'ap', 'dgDna', 'dgRna']:
    for srtb in ['dgDna']:
        #cols = ['r', 'g', 'k']

        if method == 'initial':
            arranged = getITSdnaSep(ITSs, nr='two', upto=upto)

        elif method == 'dinosaur':
            arranged = getDinosaur(ITSs, nr=1, upto=upto, sortBy=srtb)

        print('\n'+ '-------- ' + srtb + '--------')
        for labl in sorted(arranged):

            if len(arranged[labl][0]) < 30:
                continue

            aprobs, keqs = arranged[labl]

            corr, pvals = spearmanr(aprobs, keqs)

            print(labl)
            print('Corr: {0}, pval: {1}'.format(corr, pvals))
            #ax.scatter(aprobs, keqs, label=labl, color=cols.pop())
        print('-------- ' + srtb + '--------')

    #ax.legend()

    # keqs


def normalize_AP(ITSs):
    """
    Sum(AP) for a given ITS is large for low PY. Therefore, normalize it to a
    value between 0 and 1 for each its.
    """
    for its in ITSs:
        # don't count the zero AP ones (represented with negative values)
        apS = sum([v for v in its.abortiveProb if v > 0])
        its.abortiveProb = its.abortiveProb/apS

    # Control: no correlation any more
    pys, aps = zip(*[(i.PY, sum([v for v in i.abortiveProb if v >0])) for i in ITSs])
    print spearmanr(pys, aps)


def positionwise_shifted_ap_keq(ITSs, x_merStart, x_merStop, plusmin,
        keqAs='keq'):
    """
    Return array from 'x-mer' to 'y-mer'; each array has all the
    AP at those positions AND the Keq shifted by plusmin from those positions.

    Normally use keq for correlation, but also accept dg3d
    """

    # add objects here and sort by score
    objects = {}

    # in the its-arrays, abortiveProb[0] is for 2-mer
    # similarly, keq[0] is for the 2-nt RNA
    # therefore, if x_merStart is 2; we need to access ap_pos 0
    # in genral, it would have been nice if its's were indexed with x-mer
    arrayStart = x_merStart-2
    arrayStop = x_merStop-2
    for ap_pos in range(arrayStart, arrayStop):

        ap_val = []
        keqshift_val = []

        keq_pos = ap_pos + plusmin
        if keq_pos in range(arrayStart, arrayStop):
            for its in ITSs:

                ap_val.append(its.abortiveProb[ap_pos])
                if keqAs == 'keq':
                    keqshift_val.append(its.keq[keq_pos])
                elif keqAs == 'dg3d':
                    keqshift_val.append(its.dg3d[keq_pos])

        objects[ap_pos+2] = {'ap': ap_val, 'keq': keqshift_val}

    return objects


def position_wise_correlation_shifted(dg100, dg400):
    """
    Obtain the correlation between Keq and AP at each position from start to
    upto. At each position also shift the relationship: find out if the AP at a
    given position for all ITSs is correlated with a keq at another position.

    plan: the same as you get now, but obtain lists for each position 2-mer and
    3-mer and up, but produce 6 plots from -5 to +5 shift, and remember to
    bonferroni correct for each its-position. probably you should correct for
    each position * 6.

    When doing so, you obtain a strong, consistent negative correlation between
    the AP at pos and the Keq at pos+1.

    Looking at the scatter-plot, we see that when Keq is high at pos+1, then AP
    is invariably low at pos. When Keq is low, AP can be both low and high.

    So: pretranslocated state is favored at pos+1, means AP is low at pos.

    How do we explain this?

    Normalizing the AP changes the correlation between sum(SE) and sum(AP).
    Before normalization, the correlation is positive, indicating probably that
    sum(AP) is strongly correlated with PY. After normalization sum(AP) is
    NEGATIVELY correlated with sum(SE). The normalization is making the AP
    values more position-rich in information.

    It is not strange that the sign of the correlation between the positions
    and Keq do not change with normalization.

    What is strange is that there is a negative correlation at all.
    """

    #ITSs = dg400 + dg100
    ITSs = dg400
    #ITSs = dg100

    x_merStart = 2
    x_merStop = 15

    plusmin = 1
    pm_range = range(-plusmin, plusmin+1)

    for (dg, ITSs) in [('DG100', dg100), ('DG400', dg400), ('Both', dg100+dg400)]:

        # make one figure per pm val
        for pm in pm_range:
            objects = positionwise_shifted_ap_keq(ITSs, x_merStart=x_merStart,
                    x_merStop=x_merStop, plusmin=pm, keqAs='dg3d')

            fig, ax = plt.subplots()

            bar_heights = []
            p_vals = []

            nr_corl = 0

            for xmer, vals in sorted(objects.items()):

                keqs = vals['keq']
                aps = vals['ap']

                # if the shift of keq is too far away from ap, no values will be
                # returned
                if keqs != []:

                    corr, pval = spearmanr(keqs, aps)

                    bar_heights.append(corr)
                    p_vals.append(pval)

                    nr_corl += 1  # nr of non-zero correlations for bonferroni
                else:
                    bar_heights.append(0)
                    p_vals.append(0)

                # see the scatter plots for the +1 case
                #if pm == 1:
                    #figu, axu = plt.subplots()
                    #axu.scatter(keqs, aps)
                    #axu.set_xlabel('Keq at pos {0}'.format(xmer+pm))
                    #axu.set_ylabel('AP at pos {0}'.format(xmer))

                #if xmer == 2 and pm == 0:
                    #debug()

            # the positions along the x-axis (got nothing to do with labels)
            xpos = range(1, len(bar_heights)+1)
            # this is affected by xlim

            #plim = 0.05/(nr_corl*len(pm_range))  # simple bonferroni testing
            plim = 0.05/(nr_corl)  # simple bonferroni testing

            colors = []
            for pval in p_vals:
                if pval < plim:
                    colors.append('g')
                else:
                    colors.append('k')

            ax.bar(left=xpos, height=bar_heights, align='center', color=colors)
            ax.set_xticks(xpos)  # to make the 'ticks' physically appear
            ax.set_xlim(0, x_merStop-2)

            xticknames = range(x_merStart, x_merStop+1)
            ax.set_xticklabels(xticknames)

            ax.set_ylabel('Spearman correlation coefficient between AP and shifted Keq')
            ax.set_xlabel('The AP of corresponding x-mer')
            if pm < 0:
                shift = str(pm)
            else:
                shift = '+' + str(pm)

            ax.set_title('{0}: Keq shifted by {1}'.format(dg, shift))
            ax.yaxis.grid()

            ax.set_ylim(-0.81, 0.81)
            ax.set_yticks(np.arange(-0.8, 0.9, 0.2))


def plus_minus_keq_ap(dg100, dg400):
    """
    Make a histogram plot of keq correlations with AP +/- nts of Keq.
    """

    #Indicate in some way the statistically significant results.
    #Do one from 1: and one from 3:
    ITSs = dg100 + dg400

    # Try both versions: one where you don't consider results when the +/-
    # thing is above/below 3/14
    upto = 14
    start = 2
    plusmin = 10

    objects = shifted_ap_keq(ITSs, start=start, upto=upto, plusmin=plusmin)

    # make the plot
    fig, ax = plt.subplots()

    bar_heights = []
    p_vals = []
    x_tick_names = range(-plusmin, plusmin+1)

    for pm in x_tick_names:
        apObjs = objects[pm]
        aps, keqs = zip(*[(apObj.ap, apObj.keq) for apObj in apObjs])
        corr, pval = spearmanr(aps, keqs)

        bar_heights.append(corr)
        p_vals.append(pval)

    xticks = range(1, len(x_tick_names)+1)
    plim = 0.05/len(xticks)  # simple bonferroni testing
    colors = []
    for pval in p_vals:
        if pval < plim:
            colors.append('g')
        else:
            colors.append('k')

    ax.bar(left=xticks, height=bar_heights, align='center', color=colors)
    ax.set_xticks(xticks)

    namesminus = [str(x) for x in range(-plusmin, 1)]
    namesplus = ['+'+str(x) for x in range(1, plusmin+1)]

    ax.set_xticklabels(namesminus + namesplus)

    ax.set_ylabel('Spearman correlation coefficient between AP and Keq')
    ax.set_xlabel('Position of Keq value relative to AP value')
    ax.set_title('Correlation between Keq and Abortive Probability '
                 'at shifted positions')
    ax.yaxis.grid()


def apRawSum(dg100, dg400):
    """
    Simple bar plot to show where the raw reads are highest and where the
    abortive probabilities are highest across all variants.

    Result: 6 and 8 are big values. Wonder if Lilian has seen smth like this.

    Big steps at 6 and 13 in AP. These sites correspond to big jumps also in
    your ladder-plot.

    I think Lilian should see these plots. I don't think she has seen them
    before. Make them when you send her an email.
    """

    #ITSs = dg100
    ITSs = dg400
    #ITSs = dg400 + dg100

    mers = range(0, 19)
    x_names = range(2, 21)

    #rawSum = [sum([i.rawDataMean[x] for i in ITSs]) for x in mers]
    #apSum = [sum([i.abortiveProb[x] for i in ITSs]) for x in mers]

    rawMean = [np.mean([i.rawDataMean[x] for i in ITSs]) for x in mers]
    apMean = [np.mean([i.abortiveProb[x] for i in ITSs]) for x in mers]

    rawStd = [np.std([i.rawDataMean[x] for i in ITSs]) for x in mers]
    apStd = [np.std([i.abortiveProb[x] for i in ITSs]) for x in mers]

    for heights, stds, descr in [(rawMean, rawStd, 'Raw'), (apMean, apStd, 'AP')]:

        fig, ax = plt.subplots()

        ax.bar(left=mers, height=heights, align='center', yerr=stds)
        ax.set_xticks(mers)
        ax.set_xticklabels(x_names)

        ax.set_xlabel('X-mer')
        ax.set_ylabel(descr)


def basic_info(ITSs):
    """
    Basic question: why is there a better correlation between PY and sum(AP)
    than between PY and TotAbort?

    Strange: the correlation between sum(AP) and SE is NEGATIVE when AP is
    normalized, but POSITIVE when AP is not normalized. That makes no sense:
    the changes go in the same direction.

    This seems to be related to the 2-mer. When I exclude it the correlation
    doesn't change direction. However, apSum becomes

    It is also related to keeping the -AP values. When you exclude them and
    exclude the 2-mer you get the same correlation between sum(keq) and
    sum(AP). Including the 2-mer just makes things nearly 1 so that numerical
    errors come into the picture.

    So after normalization: the fraction of abortive probability accounted for
    beyond the 2-nt correlates with sum(keq)
    This is MUCH more evident in the DG400 series.

    Indeed. The normalized AP of the 2-mer in the DG400 series has a very
    strong correlation 0.74 with PY: the fraction of aborted product which
    happens at the 2-mer expains most of the PY variation. However this is not
    evident when looking at the other data.

    # NEW NEW NEW NEW
    There is DECISIVELY something fishy about the abortive probabilities.

    For DG100 you need sum(AP[:10]) before you reach any decent correlation.
    That is up to 11-mer!!!! And the 14-mer also makes the difference.

    The 6-7 mer makes a big difference too.

    However!! Excluding the 2-mer and the 3-mer improves this drastically. The
    early AP do matter, but the 2-mer and 3-mer confound things. It could be
    that they are 'cheap' abortive products, because they happen very fast.

    It seems that sum(AP) early is almost positively correlated with PY! It's
    only when considering the late AP that the correlation between sum(AP) and
    PY becomes negative, like you'd expect.

    High PY variants are likely to abort early, but unlikely to abort late.
    Relative to each other.

    The AP after the 10-mer is predictive of PY. AP before 10-mer is not very
    predictive. This is at first appears counter-intuitive, since most abortive
    product is short. However, this must be balanced by the fact that it is
    more detrimental to PY to abort late. Perhaps late abortive products take
    longer time to release? When a full RNA-DNA hybrid is formed, backtracking
    and bubble collapse could be slower.

    So it has both taken longer time to produce the product, but it also takes
    longer time to backtrack and abort.

    What underlies the difference? Different stresses causing late and early
    collapse? Or just low/high probability of late collapse.

    Here is a case study to try to understand the AP values (and raw)

    Imagine an initiating complex that aborts a 2-nt or 3-nt. This happens soon
    after initating and the initiating complex is not stable. We imagine that
    this complex can rapidly re-initialize.

    Imagine an initiating complex that aborts a 11-nt or a 12-nt. Count the
    time (t_11 > t_2). This alone will give less raw product at the 11-nt mark,
    and therefore a lower AP -- not because the actual collapse probability is
    different. You'll just see less evidence for it.

    has a full and stabilizing RNA-DNA hybrid
    contribution to the AP and raw.

    TIME is missing. TIME could explain some things. TIME is part of a more
    complex model: ap(t) is needed to explain what's going on here. You don't
    have time for this, however. You must make do with what you've got.

    If we normalize the RAW data with respect to time, we see that it's not so
    unusual to reach late transcription stages.

    Can you make a sensitivity plot? How sensitive the PY or the total abortive
    product is to the AP in a certain region?
    """

    py = [i.PY for i in ITSs]
    fl = [i.fullLengthMean for i in ITSs]
    ta = [i.totAbortMean for i in ITSs]

    #print 'FL and TotAbort: ', spearmanr(fl, ta)

    #se = [i.SE for i in ITSs]
    #ap2mer = [i.abortiveProb[0] for i in ITSs]
    ap23mer = [sum(i.abortiveProb[7:19]) for i in ITSs]

    #apSum = [sum([a for a in i.abortiveProb if a >0]) for i in ITSs]
    #apSum = [sum([a for a in i.abortiveProb[1:] if a >0]) for i in ITSs]
    #apSum = [sum(i.abortiveProb[1:]) for i in ITSs]
    #apSum = [sum(i.abortiveProb) for i in ITSs]
    # ~= [1,...,1] after normalization
    #for d, inf in [(py, 'PY'), (fl, 'FL'), (ta, 'TotAbort'), (apSum, 'sum(AP)')]:
        #print 'SE', inf, spearmanr(se, d)

    #print '2-mer AP and PY: ', spearmanr(ap2mer, py)
    #print '2-mer AP and FL: ', spearmanr(ap2mer, fl)
    #print '2-mer AP and ta: ', spearmanr(ap2mer, ta)

    #print('')

    print '2+3-mer AP and PY: ', spearmanr(ap23mer, py)
    print '2+3-mer AP and FL: ', spearmanr(ap23mer, fl)
    print '2+3-mer AP and ta: ', spearmanr(ap23mer, ta)
    print('')

    #how does the AP correlate with the raw values?
    # around 0.7
    #for i in ITSs:
        #ap = i.abortiveProb
        #raw = i.rawDataMean

        #print spearmanr(ap, raw)


def get_movAv_array(dset, center, movSize, attr, prePost):
    """
    Hey, it's not an average yet: you're not dividing by movSize .. but you
    could.
    """

    movAr = []

    # the last coordinate at which you find keq and AP info
    for i in dset:

        # for exampe abortiveProb, Keq, etc.
        array = getattr(i, attr)

        idx = center-2
        # check if there is enough space around the center to proceed
        # if center is 1, and movSize is 3, nothing can be done, since you must
        # average center-3:center+4
        if (idx - movSize) < 0 or (idx + movSize + 1) > 18:
            movAr.append(None)  # test for negative values when plotting

        else:
            if prePost == 'both':
                movAr.append(sum(array[idx-movSize:idx+movSize+1]))
            elif prePost == 'pre':
                #movAr.append(sum(array[idx-movSize:idx+1]))
                movAr.append(sum(array[idx-movSize:idx]))
            elif prePost == 'post':
                #movAr.append(sum(array[idx+2:idx+movSize+1]))
                movAr.append(sum(array[idx+1:idx+movSize+1]))
            else:
                print ":(((())))"

    return movAr


def moving_average_ap(dg100, dg400):
    """
    Define a moving average size movSize and correlate each moving average
    window of either abortive probability or raw abortive product with the PY/FL/TA

    The results are insteresting: dg100 and dg400 have slightly different
    profiles
    """

    movSize = 0
    xmers = range(2, 21)
    #attr = 'abortiveProb'
    attr = 'rawDataMean'
    plim = 0.05

    #for title, dset in [('DG100', dg100), ('DG400', dg400),
            #('DG100 + DG400', dg100+dg400)]:
    for title, dset in [('DG100', dg100), ('DG400', dg400)]:

        py = [i.PY for i in dset]
        #fl = [i.fullLengthMean for i in dset]
        #ta = [i.totAbortMean for i in dset]

        # get a moving average window for each center_position
        # calculating from an x-mer point of view
        # Use a less conservative test for the paper

        #for label, meas in [('PY', py), ('FL', fl), ('TotalAbort', ta)]:
        for label, meas in [('PY', py)]:
        #for label, meas in [('FL', fl), ('TotalAbort', ta)]:

            # the bar height will be the correlation with the above three values for
            # each moving average center position
            bar_height = []
            pvals = []

            nr_tests = 0

            for mer_center_pos in xmers:
                movArr = get_movAv_array(dset, center=mer_center_pos,
                        movSize=movSize, attr=attr, prePost='both')

                corr, pval = spearmanr(meas, movArr)

                if not np.isnan(corr):
                    nr_tests += 1

                bar_height.append(corr)
                pvals.append(pval)

            # if no tests passed, make to 1 to avoid division by zero
            if nr_tests == 0:
                nr_tests = 1

            colors = []
            for pval in pvals:
                if pval < (plim/nr_tests):
                    colors.append('g')
                else:
                    colors.append('k')

            fig, ax = plt.subplots()
            ax.bar(left=xmers, height=bar_height, align='center', color=colors)
            ax.set_xticks(xmers)
            ax.set_xlim(xmers[0]-1, xmers[-1])

            ax.set_xlabel('Nucleotide')
            ax.set_ylabel('Correlation coefficient')
            #ax.set_title('{0} -- {1}: window size: '
                            #'{2}'.format(label, attr, movSize))
            ax.set_title('{0}: Correlations between sum of abortive product at each ITS'
                    ' position and PY'.format(title))


def benjami_colors(pvals, Q, signcol, nonsigncol):
    """
    Proceedure: sort the p-values. Then, make a new array which holds the order
    of the sorted p-vals. Then add color based on the order and the pval.
    """
    colors = []

    # recall that 0 implies that no test was done.
    # you'll need to avoid this: remove all 0s
    no_zero_pvals = [p for p in pvals if p != 0]

    no_zero_pvals_sorted = sorted(no_zero_pvals)

    nr_tests = len(no_zero_pvals)

    for pvl in pvals:
        if pvl == 0:
            colors.append('k')
        else:
            # two index operations: first the index in the sorted
            if pvl < ((no_zero_pvals_sorted.index(pvl)+1)/nr_tests)*Q:
                colors.append(signcol)
            else:
                colors.append(nonsigncol)

    return colors


def pickle_wrap(data='', path='', action=''):
    """
    Load or save pickle-data
    """
    if (path == '') or (action == ''):
        print('Must provide at least a path and an action to pickle_wrap! >:S')
        return 0/1

    if action == 'save':
        saveHandle = open(path, 'wb')
        pickle.dump(data, saveHandle)
        saveHandle.close()

    elif action == 'load':
        if os.path.isfile(path):
            loadHandle = open(path, 'rb')
            result = pickle.load(loadHandle)
            loadHandle.close()

            return result
        else:
            print('{0} not found to be a valid file!'.format(path))
    else:
        print('Either save or load data .. duh')
        return 0/1


def abortive_bar(dg100, dg400):
    """
    Simply a bar-plot showing the amount of abortive product
    """

    fig, axes = plt.subplots(2)

    xmers = range(2, 21)
    attr = 'rawDataMean'

    for ax_nr, (title, dset) in enumerate([('DG100', dg100), ('DG400', dg400)]):

        ax = axes[ax_nr]

        bar_height = []

        for nc in xmers:
            bar_height.append(sum([getattr(i, attr)[nc-2] for i in dset]))

        ax.bar(left=xmers, height=bar_height, align='center')
        ax.set_xticks(xmers)
        ax.set_xlim(xmers[0]-1, xmers[-1])

        ax.set_xlabel('Nucleotide')
        ax.set_ylabel('Abortive product')
        ax.set_title(title)

    fig.suptitle('Sum of abortive product at each nucleotide position')


def doFigureCalculations(fig2calc, pickDir, figures, calculateAgain, testing, dg100,
                            dg400):
    """
    Perform all calculations necessary for plotting. Each figure can have
    several 'plot' functions.
    """

    def calcWrapper(name, dset=False, coeff=False, topNuc=False):
        """
        Wrapper for calculating and returning plot data. Pickling if necessary.
        """

        filePath = os.path.join(pickDir, name)

        if not os.path.isfile(filePath) or calculateAgain:
            calcr = Calculator(dg100, dg400, name, testing, dset, coeff,
                                topNuc)
            results = calcr.calc()

            # pickle the results for re-usage
            pickle_wrap(data=results, path=filePath, action='save')

        else:
            results = pickle_wrap(path=filePath, action='load')

        return results
    # Prepare the figures
    # Return the result of the sub calc through calcResults
    calcResults = {}

    for fig in figures:
        for subCalcName in fig2calc[fig]:
            coeff = [0.14, 0.00, 0.93]  # median of 20 runs
            #coeff = [0, 0, 1]  # median of 20 runs
            calcResults[subCalcName] = calcWrapper(subCalcName, topNuc=13,
                                                    coeff=coeff)
    return calcResults


def mad_std(a, c=0.6745, axis=None):
    """
    Median Absolute Deviation along given axis of an array:

    median(abs(a - median(a))) / c

    c = 0.6745 is the constant to convert from MAD to std; it is used by
    default

    """

    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        # I don't want the array to change so I have to copy it?
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)

    return m


def main():
    #remove_controls = True
    remove_controls = False

    dg100 = data_handler.ReadData('dg100-new')  # read the DG100 data
    dg400 = data_handler.ReadData('dg400')  # read the DG100 data

    if remove_controls:
        controls = ['DG133', 'DG115a', 'N25', 'N25anti', 'N25/A1anti']
        dg400 = [i for i in dg400 if not i.name in controls]
        #dg100 = [i for i in dg100 if not i.name in controls]

    # Normalize the AP -- removes correlation between sum(PY) and sum(AP)
    #normalize_AP(ITSs)

    # basic correlations
    #basic_info(ITSs)

    ## plot data when sorting by SE
    #ITSs = sortITS(ITSs, 'SE')

    #fold_difference(ITSs)
    ## Plot FL, PY, total RNA for the different quantitations
    #data_overview(ITSs)

    # plot correlation between Keq and AP and Raw
    # add possibility for screening for high/low dnadna dnarna values
    #keq_ap_raw(ITSs)

    # ap keq for all positions
    #plus_minus_keq_ap(dg100, dg400)

    # ap keq at each position
    #position_wise_correlation_shifted(dg100, dg400)

    # bar plot of the sum of AP and the sum of raw
    #apRawSum(dg100, dg400)

    # abortive bar plot
    #plt.ion()
    #abortive_bar(dg100, dg400)

    # moving average of AP vs PY/FL/TA
    #moving_average_ap(dg100, dg400)

    # Do not recalculate when tweaking plots
    #calculateAgain = False
    calculateAgain = True

    testing = True
    #testing = False

    figures = [
            #'megaFig',  # One big figure for the first 4 plots
            #'Figure2',  # SE vs PY
            #'Figure3',  # Delineate + DG400
            #'Figure32',  # DG400
            #'Figure4old',  # Keq vs AP
            #'FigureX',  # 2x2 Keq vs AP, effect of normalization
            #'FigureX2',  # 1x2 Keq vs AP
            #'CrossRandomDelineateSuppl',  # CrossCorrRandomDelineate in one
            'Suppl1',   # Random and cross-corrleation (supplementary)
            #'Suppl2',   # Delineate -- all combinations
            #'Suppl3',   # PY vs sum(AP) before and after normalization
            #'Suppl4',   # Examples of Keq and AP for 4 variants
            ############ Below: not figures, just single calculations
            #'DeLineate',   # Just to do the delineate calculation
            ]

    # Dictionary that maps figures to calculations
    fig2calc = {
            'megaFig': ['PYvsSE', 'cumulativeAbortive', 'delineate', 'dg400_validation'],
            'Figure2': ['PYvsSE', 'cumulativeAbortive'],
            'Figure3': ['delineate', 'dg400_validation'],
            'Figure32': ['dg400_validation'],  # only do dg400 validation
            'Figure4old': ['AP_vs_Keq'],
            'FigureX': ['AP_vs_Keq', 'sumAP'],
            'FigureX2': ['AP_vs_Keq', 'sumAP'],
            'CrossRandomDelineateSuppl': ['crossCorrRandom', 'delineate', 'delineateCombo'],
            'Suppl1':  ['crossCorrRandom'],
            'Suppl2':  ['delineateCombo'],
            'Suppl3':  ['sumAP'],
            'Suppl4':  ['AP_Keq_examples'],
            'DeLineate':  ['delineate']
            }

    ### XXX Below this line are figures that appear in the paper XXX ###

    # collect the figures you want to save to file
    saveMe = {}

    # save your calculations here for reuse
    pickDir = 'pickledData'

    #plt.ion()
    plt.ioff()
    # Do calculations
    calcResults = doFigureCalculations(fig2calc, pickDir, figures,
                                       calculateAgain, testing, dg100, dg400)

    #debug()
    # Do plotting
    for fig in figures:

        plotr = None

        ##################### 4 main plots in one #######################
        if fig == 'megaFig':

            plotr = Plotter(YaxNr=1, XaxNr=4, plotName='OneBigFig2', p_line=True)

            # The first two plots
            topNuc, indx, corr, pvals, PYs, PYstd, SEmax, SEbest = calcResults['PYvsSE']
            resCumulAb = calcResults['cumulativeAbortive']
            plotr.PYvsSEscatter(PYs, PYstd, SEmax)
            plotr.PYvsSEladder(indx, corr, pvals, PYs, SEbest, topNuc, inset=True)
            plotr.cumulativeAbortive(*resCumulAb, corr=corr)

            # The next two plots
            resDelin = calcResults['delineate']
            calcdDG400, SE15, PY, PYstd = calcResults['dg400_validation']
            plotr.delineatorPlot(resDelin)  # plot the delineate plot
            plotr.dg400validater(calcdDG400, SE15, PY, PYstd)

            # why does this give a worse result than Fig2 and Fig3 next to each
            # other? Makes no sense.
            plotr.figure.subplots_adjust(left=0.06, top=0.95, right=0.99,
                    bottom=0.20, wspace=0.5)

            plotr.setFigSize(18, 5)

            saveMe[fig] = plotr.figure

        ##################### FIGURE SE vs PY ########################
        if fig == 'Figure2':

            # standard PY vs SEn
            topNuc, indx, corr, pvals, PYs, PYstd, SEmax, SEbest = calcResults['PYvsSE']
            # cumulative amount of abortive probability
            results2 = calcResults['cumulativeAbortive']

            plotr = Plotter(YaxNr=1, XaxNr=2, plotName='SE15 vs PY',
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=2,
                    tickLength=2, tickWidth=0.5)

            ax_scatr = plotr.PYvsSEscatter(PYs, PYstd, SEmax)
            ax_lad = plotr.PYvsSEladder(indx, corr, pvals, PYs, SEbest, topNuc,
                                        inset=False)
            # XXX plot settings for ax_lad are reset below
            plotr.cumulativeAbortive(*results2, corr=corr)

            # Should be 8.7 cm
            plotr.setFigSize(8.7, 4.5)
            #plt.tight_layout()
            plotr.figure.subplots_adjust(left=0.08, top=0.96, right=0.90,
                    bottom=0.16, wspace=0.35)
            #plotr.figure.subplots_adjust(wspace=0.45)
            # make the correlation label come closer
            ax_lad.yaxis.labelpad = 1
            ax_scatr.yaxis.labelpad = 1

            plotr.addLetters(shiftX=0)

            saveMe['PYvsSE'] = plotr.figure

        ###################### FIGURE DELINEATE + DG400 ########################
        # A figure that combines the 'delineate' and scatter plot for DG400 figures
        if fig == 'Figure3':

            # delineation of effects of different energies
            delinResults = calcResults['delineate']
            # the DG400 scatter plot
            calcdDG400, SE15, PY, PYstd = calcResults['dg400_validation']

            plotr = Plotter(YaxNr=1, XaxNr=2, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1.5,
                    tickLength=2, tickWidth=0.5)

            plotr.delineatorPlot(delinResults)  # plot the delineate plot
            plotr.dg400validater(calcdDG400, SE15, PY, PYstd)

            # Should be 8.7 cm
            plotr.setFigSize(8.7, 4.7)
            plotr.figure.subplots_adjust(left=0.13, top=0.98, right=0.99,
                    bottom=0.18, wspace=0.4)

            plotr.addLetters()

            saveMe['Delineate_And_DG400'] = plotr.figure

        ###################### FIGURE  DG400 ########################
        # A figure with scatter plot for DG400 figures
        if fig == 'Figure32':

            # the DG400 scatter plot
            calcdDG400, SE15, PY, PYstd = calcResults['dg400_validation']

            plotr = Plotter(YaxNr=1, XaxNr=1, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1.5,
                    tickLength=2, tickWidth=0.5)

            plotr.dg400validater(calcdDG400, SE15, PY, PYstd)

            # Should be 8.7 cm
            plotr.setFigSize(5.7, 4.2)
            plotr.figure.subplots_adjust(left=0.15, top=0.98, right=0.99,
                    bottom=0.18, wspace=0.4)

            saveMe['DG400'] = plotr.figure

        ###################### FIGURE KEQ vs AP ########################
        if fig == 'Figure4old':
            """
            AP vs Keq correlation.
            """

            name = 'AP_vs_Keq'
            results = calcResults[name]['Normalized']
            # moving average between AP and Keq
            # the AP - Keq correlation depends only on DG3N, not on DGRNA-DNA etc.
            plotr = Plotter(YaxNr=1, XaxNr=1, plotName='SE15 vs PY')

            plotr.moving_average_ap_keq(results)

            plotr.setFigSize(12, 6)
            plt.tight_layout()

            saveMe[name] = plotr.figure

        if fig == 'FigureX2':

            """
            Display the sum(AP), SE_20 correlation as well as the nt-2-nt
            correlation (which doesn't match as well).
            """
            plotr = Plotter(YaxNr=1, XaxNr=2, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=2,
                    tickLength=2, tickWidth=0.5)

            # scatterplot
            resAP = calcResults['sumAP']
            plotr.sumAP_SE20(resAP, norm=False)

            # bar plot
            resAPvsKeq = calcResults['AP_vs_Keq']['Non-Normalized']
            axM1 = plotr.moving_average_ap_keq(resAPvsKeq)

            plotr.setFigSize(8.7, 4.0)
            #plt.tight_layout()
            plotr.figure.subplots_adjust(left=0.11, top=0.97, right=0.995,
                    bottom=0.18, wspace=0.33)

            axM1.yaxis.labelpad = 0.4

            letters = ('A', 'B')
            positions = ['UL', 'UR']
            plotr.addLetters(letters, positions)

            saveMe[fig] = plotr.figure

        ###################### FIGURE KEQ vs AP advanced ########################
        if fig == 'FigureX':

            """
            Display the sum(AP), SE_20 correlation as well as the nt-2-nt
            correlation (which doesn't match as well).

            Show that the nt-2-nt correlation persists after normalizing
            """
            plotr = Plotter(YaxNr=2, XaxNr=2, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=2,
                    tickLength=2, tickWidth=0.5)

            resNormalizedAP = calcResults['sumAP']

            plotr.sumAP_SE20(resNormalizedAP, norm=False, xlab=False,
                    xticks=False)
            plotr.sumAP_SE20(resNormalizedAP, norm=True)

            resAPvsKeqNorm = calcResults['AP_vs_Keq']['Normalized']
            resAPvsKeqNonNorm = calcResults['AP_vs_Keq']['Non-Normalized']

            axM1 = plotr.moving_average_ap_keq(resAPvsKeqNonNorm, xlab=False,
                    xticks=False)
            axM2 = plotr.moving_average_ap_keq(resAPvsKeqNorm)

            plotr.setFigSize(8.7, 6.5)
            #plt.tight_layout()
            plotr.figure.subplots_adjust(left=0.13, top=0.97, right=0.98,
                    bottom=0.12, wspace=0.35, hspace=0.1)

            axM1.yaxis.labelpad = 0.1
            axM2.yaxis.labelpad = 0.1

            letters = ('A', 'B', 'C', 'D')
            positions = ['UL', 'UL', 'UR', 'UR']
            plotr.addLetters(letters, positions)

            saveMe[fig] = plotr.figure

        ###################### FIGURE Cross-corr and Random ########################
        # Used to be in the main paper, now supplementary
        if fig == 'CrossRandomDelineateSuppl':

            plotr = Plotter(YaxNr=1, XaxNr=3, plotName=fig,
                    p_line=False, labSize=6, tickLabelSize=6, lineSize=2,
                    tickLength=2, tickWidth=0.5)

            # cross corrrlation and random
            crossCorrResults = calcResults['crossCorrRandom']
            # delineation of effects of different energies
            delinResults = calcResults['delineate']
            # all combinations of all DGs
            results = calcResults['delineateCombo']

            plotr.delineatorPlot(delinResults)  # plot the delineate plot
            plotr.delineatorComboPlot(results)
            plotr.crossCorrPlot(crossCorrResults)

            letters = ('A', 'B', 'C')
            positions = ['UL', 'UL', 'UL']
            plotr.addLetters(letters, positions, shiftX=-0.27, shiftY=0.03)

            plotr.setFigSize(17.4, 6)

            plotr.figure.subplots_adjust(left=0.1, top=0.97, right=0.98,
                    bottom=0.12, wspace=0.35, hspace=0.1)

            saveMe[fig] = plotr.figure

        ###################### FIGURE Cross-corr and Random ########################
        # Used to be in the main paper, now supplementary
        if fig == 'Suppl1':

            name = 'crossCorrRandom'
            # cross corrrlation and random
            results = calcResults[name]

            plotr = Plotter(YaxNr=1, XaxNr=1, lineSize=3, tickLabelSize=7)
            plotr.crossCorrPlot(results)
            plotr.setFigSize(12, 9)

            plotr.figure.subplots_adjust(left=0.14, top=0.97, right=0.98,
                    bottom=0.12, wspace=0.35, hspace=0.1)

            saveMe[name] = plotr.figure

        ###################### FIGURE All combinations of DGs ########################
        # Supplementary to justify that you're using the whole model
        if fig == 'Suppl2':

            name = 'delineateCombo'
            # all combinations of all DGs
            results = calcResults[name]

            plotr = Plotter(YaxNr=1, XaxNr=1, plotName='delineate combo')
            plotr.delineatorComboPlot(results)
            plotr.setFigSize(12, 9)

            saveMe[name] = plotr.figure

        ###################### FIGURE AP normalized ########################
        # Showing the effect of AP normalization
        if fig == 'Suppl3':

            name = 'sumAP'
            results = calcResults[name]

            plotr = Plotter(YaxNr=1, XaxNr=2)
            plotr.normalizedApPlot(results)

            plotr.setFigSize(12, 6)
            saveMe[name] = plotr.figure

        ###################### FIGURE Keq AP examples ########################
        # 2x4 figure: 2 first are Keq and AP examples for 2 selected promoters
        # DG1XX, DG4XX. Last two are average Keq and AP for dg100 and dg400
        # libaries
        if fig == 'Suppl4':
            name = 'AP_Keq_examples'

            # AP and Keq for two examples and mean for DG100 and DG400
            results = calcResults[name]

            plotr = Plotter(YaxNr=2, XaxNr=4)
            plotr.AP_Keq_examplesPlot(results)

            plotr.setFigSize(18, 8)
            #plotr.figure.tight_layout()
            saveMe[name] = plotr.figure

            plotr.figure.subplots_adjust(left=0.072, top=0.92, right=0.99,
                    bottom=0.14, wspace=0.20, hspace=0.2)

        # make sure that these variables are only used once
        del plotr

    # save figures as pdf
    for fName, figure in saveMe.items():
        for fig_dir in fig_dirs:
            #for formt in ['pdf', 'eps', 'png']:
            for formt in ['pdf']:
                odir = os.path.join(fig_dir, formt)

                if not os.path.isdir(odir):
                    os.makedirs(odir)

                fpath = os.path.join(odir, fName) + '.' + formt
                figure.savefig(fpath, transparent=True, format=formt)
                print("Wrote {0}".format(fpath))

    return dg100


if __name__ == '__main__':
    ITSs = main()
    ga = ITSs[0]  # just for testing attributes in the interpreter...
