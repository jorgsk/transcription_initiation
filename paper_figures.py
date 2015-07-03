"""
Program to make plots for Equilibrium paper.
"""
from __future__ import division

########### Make a debug function ##########
from ipdb import set_trace as debug  # NOQA

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


# Specify better colors than the 'ugh' default colors in matplotlib
import brewer2mpl
bmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
brew_red, brew_blue, brew_green, brew_purple, brew_orange, brew_yellow, brew_brown, brew_pink, brew_gray = bmap.mpl_colors

# Global settings for matplotlib
from matplotlib import rcParams
#rcParams['axes.labelsize'] = 9
#rcParams['xtick.color'] = 'gray'
#rcParams['ytick.labelsize'] = 9
#rcParams['legend.fontsize'] = 9

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
rcParams['text.usetex'] = True


# Global variables :)
#here = os.getcwd()  # where this script is executed from! Beware!
here = os.path.abspath(os.path.dirname(__file__))  # where this script is located
fig_dir1 = os.path.join(here, 'figures')
fig_dir2 = '/home/jorgsk/Dropbox/The-Tome/my_papers/rna-dna-paper/figures'
fig_dirs = (fig_dir1, fig_dir2)

# Figure sizes in centimeter. Height is somethign you might set maually but
# width should always be the maximum
nar_width = 8.7
biochemistry_width = 8.5
current_journal_width = biochemistry_width


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

    # below functions made for use with new scipy function curve_fit
    def normal_distribution(self, a, b, x):
        """
        """
        return (1.0 / (a*np.sqrt(2*np.pi))) * np.exp(-((x-b)**2)/(2*a**2))

    def lognormal_distribution(self, a, b, x):
        """
        """
        return (1.0 / (x*a*np.sqrt(2*np.pi))) * np.exp(-((np.log(x)-b)**2)/(2*a**2))


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

    Each calculator method has a corresponding plotting method
    """

    def __init__(self, dg100, dg400, calcName, testing, dset, coeffs, topNuc,
                 msat_normalization, average_for_plots_and_output,
                 msat_param_estimate):

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
        self.msat_normalization = msat_normalization
        self.type_of_average = average_for_plots_and_output
        self.msat_param_estimate = msat_param_estimate

    def calc(self):
        """
        Run simulations according to calc.name
        """

        # call the method by its name
        methodByName = getattr(Calculator, self.name)
        self.results = methodByName(self)

        return self.results

    def AP_vs_Keq(self):
        """
        For the new msat method there is something weird happening after +15
        for the dg100 library. Correlation is strongly negative for 16-19, and
        seemingly very large for 20. This does not happen for dg400.

        Could this be one reason why random sequences are also biased after +15
        or so?
        """

        # you're modifying it so make a copy
        dset = copy.deepcopy(self.dg100 + self.dg400)
        #dset = copy.deepcopy(self.dg100)
        #dset = copy.deepcopy(self.dg400)

        if not self.coeffs:
            print('Coeffs must be set for this figure!')
            1/0

        c1, c2, c3 = self.coeffs
        for its in dset:
            its.calc_keq(c1, c2, c3, self.msat_normalization, rna_len=20)

        movSize = 0  # effectively NOT a moving winding, but nt-2-nt comparison

        # Ignore 2, since it's just AT and very little variation occurs.
        #xmers = range(2, 21)
        xmers = range(3, 16)
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

                ## testing the N25_anti
                #movAP = get_movAv_array(dset, center=mer_center_pos,
                        #movSize=movSize, attr='abortiveProb', prePost='post')

                #movKeq = get_movAv_array(dset, center=mer_center_pos,
                        #movSize=0, attr='keq', prePost='both')

                # This is how the figure currently is
                movAP = get_movAv_array(dset, center=mer_center_pos,
                                        movSize=movSize, attr='abortiveProb',
                                        prePost='both')

                movKeq = get_movAv_array(dset, center=mer_center_pos,
                                         movSize=movSize, attr='keq',
                                         prePost='both')

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
                'dg100': {'PY': [], 'keq': [], 'sumAP': [], 'sumAPnorm': []},
                'dg400': {'PY': [], 'keq': [], 'sumAP': [], 'sumAPnorm': []}}

        # calculate keq values using latest parameter values
        # this must be kept in copy-paste sync :(

        c1, c2, c3 = self.coeffs

        for dset, name in [(self.dg100, 'dg100'), (self.dg400, 'dg400')]:
            for its in dset:
                its.calc_keq(c1, c2, c3)
                outp[name]['keq'].append(sum(its.keq))
                outp[name]['PY'].append(its.PY)
                # not to 20, but to 15?
                apS = sum([v for v in its.abortiveProb[:15] if v > 0])
                outp[name]['sumAP'].append(apS)
                apSnorm = sum([v for v in its.abortiveProb/apS if v > 0])
                outp[name]['sumAPnorm'].append(apSnorm)

        return outp

    def averageAPandKbt(self):
        """
        Return dict {dg100/400: PY, average(Keq), average(AP)}

        Return averages up to 15 only, ignoring msat above that, in order to
        compare DG100 with DG400.
        """
        outp = {
                'dg100': {'avgkbt': [], 'averageAP': [], 'PY': []},
                'dg400': {'avgkbt': [], 'averageAP': [], 'PY': []}
               }

        c1, c2, c3 = self.coeffs

        for dset, name in [(self.dg100, 'dg100'), (self.dg400, 'dg400')]:
            for its in dset:
                outp[name]['PY'].append(its.PY)

                #its.calc_keq(c1, c2, c3, self.msat_normalization, rna_len=20)
                #outp[name]['avgkbt'].append(np.nanmean(its.keq))

                #apAboveZero = [a for a in its.abortiveProb[:its.msat] if a > 0]
                #outp[name]['averageAP'].append(np.nanmean(apAboveZero))

                its.calc_keq(c1, c2, c3, self.msat_normalization, rna_len=15)
                outp[name]['avgkbt'].append(np.nanmean(its.keq[:15]))

                apAboveZero = [a for a in its.abortiveProb[:15] if a > 0]
                outp[name]['averageAP'].append(np.nanmean(apAboveZero))

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
            if c in combos or len(c) == 1:
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

        #varCombinations = [v for v in varCombinations]
        # set the range for which you wish to calculate correlation
        rnaMin = 2
        rnaMax = 20
        rnaRange = range(rnaMin, rnaMax+1)

        # Be awareof the msat_param estimate approach vs old
        # optimize-for-each-sublength-approach
        if self.msat_param_estimate:
            rnaRangeSmallestPossible = range(20, 21)
        else:
            rnaRangeSmallestPossible = rnaRange

        analysis = 'Normal'
        delinResults = {}

        for combo in varCombinations:
            print('\n---- delineate Combo -------')
            print(combo)

            comboKey = '_'.join([str(combo[v]) for v in freeEns])

            result, grid_size, aInfo = optimizeParam(self.ITSs,
                                  rnaRangeSmallestPossible, self.testing, analysis,
                                  combo,'AvgKbt', self.msat_normalization,
                                  self.msat_param_estimate)

            onlySignCoeff=0.05
            # Print some output and return one value for c1, c2, and c3
            c1, c2, c3 =\
                    get_print_parameterValue_output(result,
                    rnaRangeSmallestPossible, onlySignCoeff, analysis,
                    self.name, grid_size, self.msat_normalization,
                    self.type_of_average, self.msat_param_estimate,
                    aInfo=aInfo, combo=combo)

            # add the constants to the ITS objects and calculate Keq
            for rna in self.ITSs:
                rna.calc_keq(c1, c2, c3, self.msat_param_estimate, rnaMax)

            # calculate the correlation with PY
            corr, pvals = self.correlateMeasure_PY(rnaRange, measure='AvgKbt')

            delinResults[comboKey] = [rnaRange, corr, pvals]

        return delinResults

    def delineate(self):
        """
        See the individual effect of DG3N, DGRNADNA, DGDNADNA
        """

        delineateResultsOrder = ['DNA', 'RNA', '3N']

        # include the 3 variables sequentially
        varCombinations = [{'DNA': False, 'RNA': False, '3N': True},
                           {'DNA': True,  'RNA': False, '3N': False},
                           {'DNA': False, 'RNA': True,  '3N': False}]

        # set the range for which you wish to calculate correlation
        rnaMin = 2
        rnaMax = 20
        rnaRange = range(rnaMin, rnaMax+1)

        # Be awareof the msat_param estimate approach vs old
        # optimize-for-each-sublength-approach
        if self.msat_param_estimate:
            rnaRangeSmallestPossible = range(20, 21)
        else:
            rnaRangeSmallestPossible = rnaRange

        analysis = 'Normal'
        delinResults = {}

        for combo in varCombinations:
            print('\n---- delineate -------')

            comboKey = '_'.join([str(combo[v]) for v in delineateResultsOrder])

            result, grid_size, aInfo = optimizeParam(self.ITSs,
                    rnaRangeSmallestPossible, self.testing,
                    analysis, combo, 'AvgKbt', self.msat_normalization,
                    self.msat_param_estimate)

            # For this type of analysis you do NOT want to do any screening;
            # otherwise you won't be getting any plots. It's for the reporting
            # / logging to file that you want to do the screening. These should
            # have been separated. Now you're stuck with the situation where
            # you'll have to re-run the simulations after making plots just to
            # get the output correlation coefficients. As an alternative, can
            # you separate these two? Avoid problems (try not to make new
            # problems).
            onlySignCoeff=0.05

            c1, c2, c3 = get_print_parameterValue_output(result,
                    rnaRangeSmallestPossible,
                    onlySignCoeff, analysis, self.name, grid_size,
                    self.msat_normalization, self.type_of_average,
                    self.msat_param_estimate, aInfo=aInfo, combo=combo)

            # add the constants to the ITS objects and calculate Keq
            for rna in self.ITSs:
                rna.calc_keq(c1, c2, c3, self.msat_normalization, rnaMax)

            # calculate the correlation with PY
            corr, pvals = self.correlateMeasure_PY(rnaRange, measure='AvgKbt')

            delinResults[comboKey] = [rnaRange, corr, pvals]

        return delinResults

    def correlateMeasure_PY(self, rna_lengths, measure='AvgKbt', dset='dg100'):
        """
        Return three lists: indices, corr-values, and p-values
        Correlation between SE_n and PY

        ATC -> [AT], [TC] always n-1 translocation steps.

        keq[0] is 2-nt, keq[1] is 3-nt
        """

        if dset == 'dg100':
            ITSs = self.ITSs
        elif dset == 'dg400':
            ITSs = self.dg400
        else:
            print('Scusme.')
            1/0

        PY = [i.PY for i in ITSs]

        corr = []
        pvals = []

        # keq[0] corresponds to 2-nt
        # therefore, ind=2 -> sum(i.keq[:1]) = sum(i.keq[0])
        # therefore, ind=20 -> corresponds to 20-nt

        for rna_len in rna_lengths:

            if measure == 'SE':
                measure_values = [sum(i.keq[:rna_len-1]) for i in ITSs]
            elif measure == 'product':
                measure_values = [np.prod(i.keq[:rna_len-1]) for i in ITSs]
            elif measure == 'AvgKbt':
                measure_values = [np.nanmean(i.keq[:rna_len-1]) for i in ITSs]
            elif measure == 'SumPurines':
                measure_values = [sum(i.purines[:rna_len]) for i in ITSs]

            co, pv = spearmanr(measure_values, PY)

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
            1/0

        c1, c2, c3 = self.coeffs
        # add the constants to the ITS objects and calculate Keq
        for its in self.dg400:
            its.calc_keq(c1, c2, c3, self.msat_normalization, 15)

        # get PY and PY standard deviation
        PY = [i.PY for i in self.dg400]
        PYstd = [i.PY_std for i in self.dg400]

        # Get the SE15
        values = [sum(i.keq[:14])/14 for i in self.dg400]

        # write all of this to an output file
        write_py_SE15(self.dg400)

        return self.dg400, values, PY, PYstd

    def dg400_scatter_ladder(self):
        """
        Calculate correlation for the DG400 library between AvgKbt and PY, and
        also for the number of purines and PY
        """

        if self.coeffs:
            c1, c2, c3 = self.coeffs
        else:
            print("omg no!")
            1/0

        rnaMin = 2
        rnaMax = 15
        rnaRange = range(rnaMin, rnaMax+1)

        c1, c2, c3 = self.coeffs
        # add the constants to the ITS objects and calculate Keq
        for its in self.dg400:
            its.calc_keq(c1, c2, c3, self.msat_normalization, rnaMax)

        corr_kbt, pvals_kbt = self.correlateMeasure_PY(rnaRange, measure='AvgKbt', dset='dg400')

        # get PY and PY standard deviation
        PY = [i.PY for i in self.dg400]
        PYstd = [i.PY_std for i in self.dg400]

        # Get the AvgKbt
        values = [np.nanmean(i.keq[:rnaMax-1]) for i in self.dg400]

        # write all of this to an output file
        write_py_SE15(self.dg400)

        # Piling onto the output. Calculate the average number of purines
        for its in self.dg400:
            its.calc_purines()

        corr_purine, pvals_purine = self.correlateMeasure_PY(rnaRange, measure='SumPurines',
                                                             dset='dg400')

        return self.dg400, rnaRange, values, PY, PYstd, corr_kbt, pvals_kbt, corr_purine, pvals_purine

    def PYvsAvgKbt(self):
        """
        Correlate PY with AvgKbt and return corr, pval, and indx for plotting

        topNuc is the 'best' correlating nucleotide position
        """

        #set the range of values for which you wish to calculate correlation
        # some ITSs have msat at 20, so go up to that

        # start with 2-nt RNA, from which the first translocation step is taken
        # XXX For MSAT-calculations, use rnaMin=rnaMax=20. This gives you the
        # coeffs you're after. Then, use those coeffs with rnaMin=2,rnaMax=20
        # to create the ladder plot.

        rnaMin = 2
        rnaMax = 20
        #rnaMax = 15
        rnaRange = range(rnaMin, rnaMax+1)

        # XXX Be very aware of the distinction between rnaRange and
        # rnaRangeSmallest which is used for msat parameter estimation.
        if self.msat_param_estimate:
            rnaRangeSmallestPossible = range(20, 21)
        else:
            rnaRangeSmallestPossible = rnaRange

        # With MSAT on, 20 is last scatter plot
        if self.msat_normalization:
            scatter_plot_nt_pos = 20  # msat
        else:
            scatter_plot_nt_pos = 13  # max without msat

        if scatter_plot_nt_pos not in rnaRange:
            return 'XXX: scatter plot could not be made'

        # if coefficients have been provided, do not calculate from scratch
        if self.coeffs:
            c1, c2, c3 = self.coeffs
        else:
            analysis = 'Normal'
            onlySignCoeff = 0.05
            variable_combo = {'DNA': True, 'RNA': True, '3N': True}
            result, grid_size, aInfo = optimizeParam(self.ITSs,
                    rnaRangeSmallestPossible,
                    self.testing, analysis, variable_combo, 'AvgKbt',
                    self.msat_normalization, self.msat_param_estimate)

            c1, c2, c3 = get_print_parameterValue_output(result,
                    rnaRangeSmallestPossible,
                    onlySignCoeff, analysis, self.name,
                    grid_size, self.msat_normalization, self.type_of_average,
                    self.msat_param_estimate, aInfo=aInfo)

            self.coeffs = c1, c2, c3

            print ('c1: {0}, c2: {1}, c3: {2}'.format(c1, c2, c3))

        # add the constants to the ITS objects and calculate Keq
        for its in self.ITSs:
            its.calc_keq(c1, c2, c3, self.msat_normalization, rnaMax)
        # Ensure to use full rnaRange here, even if a smaller RNA range has
        # been used to obtain the weight coefficients themselves
        corr, pvals = self.correlateMeasure_PY(rnaRange, measure='AvgKbt')
        indx = rnaRange

        PYs = [i.PY for i in self.ITSs]
        PYstd = [i.PY_std for i in self.ITSs]

        tr_steps = float(max(indx)-1)

        # now dividing by ntMax to calculate average Kbt
        values_max = [np.nanmean(i.keq[:tr_steps]) for i in self.ITSs]
        values_best = [np.nanmean(i.keq[:self.topNuc]) for i in self.ITSs]
        values_choice = [np.nanmean(i.keq[:scatter_plot_nt_pos-1]) for i in self.ITSs]

        value_pack = (values_max, values_best, values_choice, scatter_plot_nt_pos)

        # also calculate the correlation obtained with 1-1-1 (baseline case) to
        # show what the result would have been when comparing with others.
        for its in self.ITSs:
            its.calc_keq(1.0, 1.0, 1.0, self.msat_normalization, rnaMax)
        corr111, pvals111 = self.correlateMeasure_PY(rnaRange, measure='AvgKbt')

        # Piling onto the output. Calculate the average number of purines
        for its in self.ITSs:
            its.calc_purines()
        corr_purine, pvals_purine = self.correlateMeasure_PY(rnaRange, measure='SumPurines')

        # topNuc and values_best were used to make the inset plot
        return self.topNuc, indx, corr, pvals, PYs, PYstd, value_pack, corr111, pvals111, corr_purine, pvals_purine

    def Shuffler(self):
        """
        Obtain optimal correlation for a shuffled ITS-set where MSAT and PY
        remain the same, but the sequence is randomly shuffled.
        """
        rnaMin = 2
        rnaMax = 20
        rnaRange = range(rnaMin, rnaMax+1)

        # XXX Be very aware of the distinction between rnaRange and
        # rnaRangeSmallest which is used for msat parameter estimation.
        if self.msat_param_estimate:
            rnaRangeSmallestPossible = range(20, 21)
        else:
            rnaRangeSmallestPossible = rnaRange

        variable_combo = {'DNA': True, 'RNA': True, '3N': True}

        analysis = 'Random'

        # copy-waste, now passing 'shuffle' keyword
        result, grid_size, aInfo = optimizeParam(self.ITSs,
                rnaRangeSmallestPossible, self.testing, analysis,
                variable_combo, 'AvgKbt', self.msat_normalization,
                self.msat_param_estimate, randomize_method='shuffle')

        corr_std = [r[1].corr_std for r in sorted(result.items())]

        if self.msat_param_estimate:
            corr = result[20].all_corr_for_msat_mean
            corr_std = result[20].all_corr_for_msat_std
            pvals = [np.nan for _ in corr]  # you don't use these

        else:
            corr = [r[1].corr_mean for r in sorted(result.items())]
            pvals = [r[1].pvals_mean for r in sorted(result.items())]

        # NOTE: you don't care about the c1,c2,c3, but you want to
        # write output ...
        onlySignCoeff = 0.05
        c1, c2, c3 = get_print_parameterValue_output(result,
                        rnaRangeSmallestPossible,
                        onlySignCoeff, analysis, self.name, grid_size,
                        self.msat_normalization, self.type_of_average,
                        self.msat_param_estimate, aInfo=aInfo)

        shuffler = {analysis: [rnaRange, corr, corr_std, pvals]}

        return shuffler

    def crossCorrRandom(self, testing=True):
        """
        Cross-correlation and randomization of ITS sequences: effect on SE_n-PY
        correlation.
        """

        # set the range of values for which you wish to calculate correlation
        rnaMin = 2
        rnaMax = 20
        rnaRange = range(rnaMin, rnaMax+1)

        # XXX Be very aware of the distinction between rnaRange and
        # rnaRangeSmallest which is used for msat parameter estimation.
        if self.msat_param_estimate:
            rnaRangeSmallestPossible = range(20, 21)
        else:
            rnaRangeSmallestPossible = rnaRange

        analysis2stats = {}

        variable_combo = {'DNA': True, 'RNA': True, '3N': True}

        for analysis in ['Normal', 'Random', 'Cross Correlation']:
        #for analysis in ['Cross Correlation']:
        #for analysis in ['Random']:

            result, grid_size, aInfo = optimizeParam(self.ITSs,
                    rnaRangeSmallestPossible, self.testing, analysis,
                    variable_combo, 'AvgKbt', self.msat_normalization,
                    self.msat_param_estimate)

            corr_std = [r[1].corr_std for r in sorted(result.items())]

            # if normal, pick mean (no-msat_param_estimate) or optimal for
            # rna=20=msat parameter values
            if analysis == 'Normal':
                onlySignCoeff = 0.05

                c1, c2, c3 = get_print_parameterValue_output(result,
                        rnaRangeSmallestPossible,
                        onlySignCoeff, analysis, self.name, grid_size,
                        self.msat_normalization, self.type_of_average,
                        self.msat_param_estimate, aInfo=aInfo)

                for its in self.ITSs:
                    its.calc_keq(c1, c2, c3, self.msat_normalization, rnaMax)

                corr, pvals = self.correlateMeasure_PY(rnaRange, measure='AvgKbt')

            # if random or cross-correlation, pick the average of the best
            # results for each nucleotide
            else:
                if self.msat_param_estimate:
                    corr = result[20].all_corr_for_msat_mean
                    corr_std = result[20].all_corr_for_msat_std
                    pvals = [np.nan for _ in corr]  # you don't use these

                else:
                    corr = [r[1].corr_mean for r in sorted(result.items())]
                    pvals = [r[1].pvals_mean for r in sorted(result.items())]

                # NOTE: you don't care about the c1,c2,c3, but you want to
                # write output ...
                onlySignCoeff = 0.05
                c1, c2, c3 = get_print_parameterValue_output(result,
                                rnaRangeSmallestPossible,
                                onlySignCoeff, analysis, self.name, grid_size,
                                self.msat_normalization, self.type_of_average,
                                self.msat_param_estimate, aInfo=aInfo)

            analysis2stats[analysis] = [rnaRange, corr, corr_std, pvals]

        return analysis2stats


class Plotter(object):
    """
    Object that deals with plotting. Knows about the plotting parameters
    (ticks, number of axes) and the figure and axes objects.

    Contains routines for plotting onto the axes. These routines are the plots
    that enter the paper.
    """

    def __init__(self, YaxNr=1, XaxNr=1, shareX=False, shareY=False,
                    plotName='MyPlot', p_line=True, labSize=10.5, tickLabelSize=10.5,
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

        widthInch = cm2inch(xCm)
        heightInch = cm2inch(yCm)

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
            self._nextXaxNr += 1

        return nextAx

    def addLetters(self, letters=('A', 'B'), positions=('UL', 'UL'), shiftX=0, shiftY=0):
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
                ax.bar(range(2, len(data)+2), data, yerr=data_std, align='center')
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
                    ax.set_ylabel('Keq')

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

        #colors = benjami_colors(pvals, Q=Q, nonsigncol=brew_gray, signcol=brew_orange)
        colors, sign_indx = benjami_colors(pvals, Q=Q, nonsigncol=brew_gray, signcol=brew_gray)

        def autolabel(rects, sign_indx):
            # attach some text labels
            for rect, sign in zip(rects, sign_indx):
                height = rect.get_height()
                if sign == 1:
                    ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '$*$',
                            ha='center', va='bottom')

        # plotit
        ax = self.getNextAxes()  # for the local figure
        rectangles = ax.bar(left=xmers, height=bar_height, align='center', color=colors, width=1.0)
        ax.set_xticks(xmers)
        ax.set_xlim(xmers[0]-1, xmers[-1]+1)

        # add a start for the significant ones
        autolabel(rectangles, sign_indx)

        # labels
        if xlab:
            ax.set_xlabel('ITS position $i$', size=self.labSize)
        ax.set_ylabel('Correlation: AP and $K_{bt,i}$', size=self.labSize)

        # ticks
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength, width=self.tickWidth)

        # X: show only positions with significant positions or 5, 10, 15, 20
        # XXX The start here must be in sync with the range for which
        # calculations have been done!
        xIndx = []
        for inx, colr in enumerate(colors, start=3):
            #if (inx%2 == 0):
            if (inx%2 != 0):
                xIndx.append(str(inx))
            else:
                xIndx.append('')

        if xticks:
            ax.set_xticklabels(xIndx)
        else:
            ax.set_xticklabels([])

        # XXX uncommented while testing
        # Y: show every other tick with a label
        yIndx = [g for g in np.arange(-0.2, 0.8, 0.2)]
        ax.set_yticks(yIndx)
        ax.set_ylim(-0.05, 0.73)

        return ax

    def averageAP_SEXX(self, results, norm=False, xlab=True, xticks=True):
        """
        Each library has a high correlation coefficient alone -- but not when
        combined??? Something must be wrong in the ordering, or some systemic
        bias in avgkbt?
        """

        ax = self.getNextAxes()

        keq = results['dg100']['avgkbt'] + results['dg400']['avgkbt']
        #keq = results['dg400']['avgkbt']
        #keq = results['dg100']['avgkbt']

        averageAP = results['dg100']['averageAP'] + results['dg400']['averageAP']

        averageAP400 = results['dg400']['averageAP']
        averageAP100 = results['dg100']['averageAP']

        PY400 = results['dg400']['PY']
        PY100 = results['dg100']['PY']

        print('Correlation between PY and average AP for 400 and 100:')
        print spearmanr(PY400, averageAP400)
        print spearmanr(PY100, averageAP100)
        print('---')
        1/0

        # make into fake percentage
        averageAP = [s*100 for s in averageAP]

        ax.scatter(keq, averageAP, c='gray')
        print('Spearmanr avgkbt and average AP')
        if norm:
            print('Normalized')
        else:
            print('Not normalized')
        #print pearsonr(keq, averageAP)
        print spearmanr(keq, averageAP)

        # labels
        if xlab:
            ax.set_xlabel('$\overline{K}_{bt,\mathrm{MSAT}}$', size=self.labSize)

        ax.set_ylabel('Average AP ($\%$)', size=self.labSize)

        # ticks
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength, width=self.tickWidth)

        if not xticks:
            ax.set_xticklabels([])

        if norm:
            yrang = np.arange(0.996, 1.004, 0.002)
        else:
            yrang = np.arange(0, 36, 5)
            ax.set_yticks(yrang)
            ax.set_ylim(0, 36)

        #ax.set_xlim(0.45, 1.2)
        #debug()

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
        axL.set_xlabel('SE$_{20}$')
        axL.set_ylabel('Sum AP')

        axR.scatter(keq100, sumAP100norm, c='b')
        axR.scatter(keq400, sumAP400norm, c='b')
        axL.set_xlabel('SE$_{20}$')
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
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.grid(True, which='major', color='lightgrey',
                      alpha=0.5, dashes=[1,1,1,1,1,1])

        colors = ['g', 'b', 'k', 'c']
        colors = [brew_blue, brew_blue, brew_green, brew_purple]
        lstyle = ['-', '-', '-', '-']

        # create labels from the combinations
        #enOrder = ['$\Delta${DNA-DNA}', '$\Delta{RNA-DNA}$', '$\Delta{3N}$']
        enOrder = ['$\Delta$DNA-DNA', '$\Delta$RNA-DNA', '$\Delta${3N}']

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

            # skip the one with all -- you're showing it in too many plots
            if key == 'True_True_True':
                continue

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

            indxMax = indx[-1] + 1
            indMin = 2

            xticklabels = [str(integer) for integer in range(indMin, indxMax)]

            #Make sure ymin has only one value behind the comma
            ymin = float(format(ymin, '.1f'))
            yticklabels = [format(i, '.1f') for i in np.arange(-ymin, -ymax, -0.1)]

            # legend
            ax.legend(loc='upper left', prop={'size': 5}, handlelength=3.3)

            # xticks
            ax.set_xticks(range(indMin, indxMax))
            #ax.set_xticklabels(xticklabels)
            ax.set_xticklabels(odd_even_spacer(xticklabels, oddeven='even'))
            ax.set_xlim(indMin-1, indxMax)
            ax.set_xlabel("$n$", size=self.labSize)

            #  setting the tick font sizes
            ax.tick_params(labelsize=self.tickLabelSize)

            ax.set_ylabel("Correlation: PY and average $\overline{K}_{bt,n}$", size=self.labSize)

            ax.set_yticks(np.arange(ymin, ymax, 0.1))
            ax.set_yticklabels(odd_even_spacer(yticklabels, oddeven='odd'))
            ax.set_ylim(ymin, ymax)

    def delineatorPlot(self, delineateResults, inst_change=False):
        """
        delineateResults is a double dictionary
        Double dictionary [FalseFalseTrue][name] -> [indx, corr, pvals]
        where the the three lists are the indices correlation and pvaleus for
        correlation between PY and SE_indx

        Where the falsetrues are in the order DNA RNA 3N
        So the above example would be the keq values from optimizing only on 3N

        inst_change flag enables plotting of instantaneous increase in
        correlation for each step
        """

        ymin = -0.3  # correlation is always high
        ymax = 1.1

        ax = self.getNextAxes()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.grid(True, which='major', color='lightgrey',
                      alpha=0.5, dashes=[1,1,1,1,1,1])

        colors = [brew_green, brew_purple, brew_blue]
        #dashes = [(1,1,1,1,1,1), (1,3,1,3,1,3), False]
        dashes = [False, False, False]

        # convert from dict-keys to plot labels
        key2label = {
                'False_False_True': '$\Delta$3N',
                'True_False_False': '$\Delta$DNA-DNA',
                'False_True_False': '$\Delta$RNA-DNA'}

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

        # also plot the instantaneous increase in correlation for each
        # nucleotide addition
        if inst_change:
            debug()

        # interpolate pvalues (x, must increase) with correlation (y) and
        # obtain the correlation for p = 0.05 to plot as a black
        p_line = False
        if p_line and (label == '$\Delta{3N}$'):
            # hack to get pvals and corr coeffs sorted
            pv, co = zip(*sorted(zip(pvals, corr)))
            f = interpolate(pv, co, k=1)
            ax.axhline(y=f(0.05), ls='--', color='r', linewidth=3)

        indxMax = indx[-1] + 1
        indMin = 2

        #Make sure ymin has only one value behind the comma
        ymin = float(format(ymin, '.1f'))
        yticklabels = [format(i, '.1f') for i in np.arange(-ymin, -ymax, -0.1)]

        # legend
        ax.legend(loc='upper left', prop={'size' :6}, handlelength=3)

        # xticks
        ax.set_xticks(range(indMin, indxMax))

        xtickLabels = []
        for i in range(indMin, indxMax):
            if i%2 == 0:
                xtickLabels.append(str(i))
            else:
                xtickLabels.append('')

        ax.set_xticklabels(xtickLabels)

        ax.set_xlim(indMin-1, indxMax)
        ax.set_xlabel("$n$", size=self.labSize)

        # setting the tick font sizes
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength, width=self.tickWidth)

        ax.set_ylabel("Correlation: PY and $\overline{K}_{bt,n}$", size=self.labSize)

        ax.set_yticks(np.arange(ymin, ymax, 0.1))
        ax.set_yticklabels(odd_even_spacer(yticklabels, oddeven='odd'))
        ax.set_ylim(ymin, ymax)

    def dg400validater(self, dg400, values, PY, PYstd, fit_function=False):

        ax = self.getNextAxes()

        print("DG400 correlation values PY")
        print spearmanr(values, PY)

        # multiply by 100 for plot
        PY = [p*100 for p in PY]
        PYstd = [p*100 for p in PYstd]

        # simple average for now, just for the effect
        ax.errorbar(values, PY, yerr=PYstd, fmt=None, ecolor='gray', zorder=1, elinewidth=0.3)
        ax.scatter(values, PY, c=brew_blue, s=12, zorder=2, lw=0)

        ########### Set figure and axis properties ############
        ax.set_xlabel('$\overline{K}_{bt,\mathrm{15}}$', size=self.labSize)

        ax.set_ylabel('Productive yield ($\%$)', size=self.labSize)

        #xmin, xmax = min(SE15), max(SE15)
        #xscale = (xmax-xmin)*0.1
        #ax.set_xlim(xmin-xscale, xmax+xscale)
        #ax.set_xlim(0, 20)

        ymin, ymax = min(PY), max(PY)
        yscale_low = (ymax-ymin)*0.2
        yscale_high = (ymax-ymin)*0.3
        ax.set_ylim(ymin-yscale_low, ymax+yscale_high)
        ax.set_yticks(np.linspace(0, 30, 4))

        # FINALLY!!! A clean way to set which tick labels to show.
        # Cry pain for the hacks you've used to get this effect
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

        # tick parameters
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength, width=self.tickWidth)

        # add a polynomial or sigmoid fit
        if fit_function:
            self.add_fitted_function(ax, PY, values)

        dg400dict = dict(((o.name, o) for o in dg400))

        specials = ['N25', 'N25/A1anti', 'DG115a', 'DG133']

        t_steps = 14.

        for name in specials:

            # maybe you've already skipped these
            if name not in dg400dict:
                continue

            SE15 = sum(dg400dict[name].keq[:t_steps])/t_steps
            PY = dg400dict[name].PY*100

            if name == 'N25':
                box_x = 20
                box_y = 10

            if name == 'N25_A1anti' or name == 'N25/A1anti':
                name = 'N25/A1anti'
                box_x = -1
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
        return ax

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

    def PYvsAvgKbtscatter(self, PYs, PYstd, values, rna_len_choice):
        """
        Plotting a scatter between PY and SE at position choice

        Means=True means plotting the average reverse translocation equilibrium
        value and not the sum.
        """

        # the index for which corr has the highest value
        print("PYvsAvgKbtscatter: Spearman, Pearson for avgkbt{0} and PY"
                .format(rna_len_choice))
        print(str(spearmanr(values, PYs)) + 'Spearmanr')
        print(str(pearsonr(values, PYs)) + 'Pearsonr')

        ax = self.getNextAxes()

        # multiply by 100 for plot
        PYs = [p*100 for p in PYs]
        PYstd = [p*100 for p in PYstd]

        ax.errorbar(values, PYs, yerr=PYstd, fmt=None, ecolor='gray', zorder=1,
                elinewidth=0.3)
        ax.scatter(values, PYs, c=brew_blue, s=12, zorder=2, lw=0)
        # XXX get the errorbars to be gray too!!!!!!!!

        ax.set_ylabel("Productive yield ($\%$)", size=self.labSize)
        ax.set_xlabel("$\overline{K}_{bt,\mathrm{MSAT}}$", size=self.labSize)

        ymin = -0.1

        # setting the tick number sizes
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)

        xmin, xmax = min(values), max(values)
        step = 0.2
        tick_positions = np.arange(xmin, xmax+step, step)
        tick_labels = [format(v, '.1f') for v in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        xscale = (xmax-xmin)*0.08
        ax.set_xlim(min(xmin-xscale, tick_positions[0]),
                    max(xmax+xscale, tick_positions[-1]))

        ymin, ymax = min(PYs), max(PYs)
        #yscale_low = (ymax-ymin)*0.03
        yscale_high = (ymax-ymin)*0.34
        ax.set_ylim(0, ymax+yscale_high)

        return ax

    def instantaneousCorrelation(self, corr):
        """
        Plot the increase in correlation at each reaction step
        """
        #from scipy.stats import lognorm, norm
        #from numpy.polynomial import Polynomial

        # use the current axis
        mainAx = self.axes[self._thisYaxNr, self._thisXaxNr]
        instantAx = mainAx.twinx()  # make a twin axis

        xlim = mainAx.get_xlim()

        instCorr = [abs(corr[i]) - abs(corr[i-1]) for i in range(1, len(corr))]

        instCorrZerod = []
        for val in instCorr:
            if val > 0:
                instCorrZerod.append(val)
            else:
                instCorrZerod.append(0)

        # float -> int
        xlim = (int(xlim[0]), int(xlim[1]+1))
        rna_lenghts = range(*xlim)

        instantAx.bar(rna_lenghts, instCorrZerod, width=0.8, color='g',
                alpha=0.4, align='center')

        # Fit data to normal and lognormal distributions
        # Add that 2-mers have 0 correlation too; for the sake of the plot.
        #F = Fit()
        #rna_lenghts_Z = [2] + rna_lenghts
        #instCorrZerod_Z = [0] + instCorrZerod
        #params_normal = optimize.curve_fit(F.normal_distribution, rna_lenghts_Z,
                #instCorrZerod_Z)
        #normal_fitted_values = [F.normal_distribution(l, *params_normal[0]) for l
                #in rna_lenghts_Z]
        #params_lognormal = optimize.curve_fit(F.lognormal_distribution, rna_lenghts_Z,
                #instCorrZerod_Z)
        #lognormal_fitted_values = [F.lognormal_distribution(l, *params_lognormal[0]) for l
                #in rna_lenghts_Z]

        #pp = Polynomial.fit(rna_lenghts_Z, instCorrZerod_Z, 4)

        # XXX don't show this fit; it's nice, but you'd need to devote more
        # results/discussion to it.
        #instantAx.plot(*pp.linspace(), color='k')

        # XXX opt for polyfit instead of fitting to lognorm
        #instantAx.plot(rna_lenghts_Z, lognormal_fitted_values, color='k')

        instantAx.set_ylabel('Increase in correlation', size=self.labSize-1, color='green')

        # tick parameters
        instantAx.tick_params(axis='y', labelsize=self.tickLabelSize,
                length=self.tickLength, width=self.tickWidth, colors='green')

        yrange = np.arange(0,1.1,0.1)
        yticklabels_skip = odd_even_spacer(yrange)

        instantAx.set_yticks(yrange)
        instantAx.set_yticklabels(yticklabels_skip)
        instantAx.set_ylim(0, instantAx.get_ylim()[1])

        rna_max = xlim[-1]
        # override
        #its_max = 20
        instantAx.set_xticks(range(2, rna_max + 1))

        xtickLabels = []
        for i in range(2, rna_max+1):
            if i%5 == 0:
                xtickLabels.append(str(i))
            else:
                xtickLabels.append('')

        instantAx.set_xticklabels(xtickLabels)

        instantAx.set_xlim(2-0.3, rna_max + 0.3)

        return

    def cumulativeAbortive(self, cumulRaw, cumulAp, corr):
        """
        Plot the cumulative abortive probability on top of the correlation
        ladder.
        """

        # use the current axis
        mainAx = self.axes[self._thisYaxNr, self._thisXaxNr]

        cumulRawNorm = np.array(cumulRaw)/cumulRaw[-1]
        cumulRawNormMinusFirst = (np.array(cumulRaw)-cumulRaw[0])/cumulRaw[-1]
        cumulApNorm = np.array(cumulAp)/cumulAp[-1]

        #choice = 'AP'
        choice = 'raw'

        xlim = mainAx.get_xlim()
        cumlAx = mainAx.twinx()  # make a twin axis

        if choice == 'AP':
            # cumulApNorm starts with 2. make it start with 3 using [1:]
            cumlAx.plot(np.arange(xlim[0]-1, xlim[1]+1), cumulApNorm, ls='-',
                    linewidth=2, color='g')

            cumlAx.set_ylabel('Cumulative abortive probability',
                    size=self.labSize, color='green')

        elif choice == 'raw':
            #cumlAx.plot(np.arange(xlim[0]-1, xlim[1]+1), cumulRawNorm, ls='-',
                    #linewidth=1, color='g')
                    #linewidth=1, color='g', marker='D', markersize=3)
            cumlAx.plot(np.arange(xlim[0]-1, xlim[1]+1), cumulRawNormMinusFirst, ls='-',
                    linewidth=1, color='g')

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

    def PYvsAvgKbtLadder111(self, existing_ax, indx, corr111, pvals111, PYs):
        """
        Show what correlation looks like with weights equal to 111
        """
        # check for nan in corr (make it 0)
        corr, pvals = remove_nan(corr111, pvals111)

        ## the ladder plot
        colr = brew_gray
        revCorr = [c*(-1) for c in corr]  # to get an increasing correlation
        existing_ax.plot(indx, revCorr, linewidth=self.lineSize, color=colr, marker='s',
                markersize=2, ls='-', label='$c_{1}=c_{2}=c_{3}=1$', markeredgecolor='none')

    def PYvsAvgKbtLadderPurines(self, existing_ax, indx, corr_purines, pvals_purines, PYs):
        """
        Show what correlation looks like with weights equal to _purines
        """
        main_ax = self.axes[self._thisYaxNr, self._thisXaxNr]
        purine_ax = main_ax.twinx()  # make a twin axis

        # check for nan in corr (make it 0)
        corr, pvals = remove_nan(corr_purines, pvals_purines)

        ## the ladder plot
        colr = brew_green
        #rev_corr = [c*(-1) for c in corr_purines]  # to get an increasing correlation
        purine_ax.plot(indx, corr, linewidth=self.lineSize, color=colr, marker='s',
                markersize=2, ls='-', label='Number of purines', markeredgecolor='none')

        purine_ax.set_ylabel('Correlation: PY and \#purines\n up to RNA length $n$',
                size=self.labSize-1, color='green', multialignment='center')

        # Set axis parameters the same as for other axis
        yticks = [t for t in main_ax.get_yticks()]

        yticklabels = [tl.get_text() for tl in main_ax.get_yticklabels()]
        yticklabels = [-1*float(tl) if tl != ' ' else ' ' for tl in yticklabels]  # positive now
        yticklabels[1] = '0.0'  # sigh
        ylim = [l for l in main_ax.get_ylim()]

        purine_ax.set_yticks(yticks)
        purine_ax.set_yticklabels(yticklabels)
        purine_ax.set_ylim(ylim)

        #purine_ax.set_xticklabels(xticklabels)
        purine_ax.set_xlim(2-0.5, indx[-1]+1)
        #purine_ax.set_xticks(xticks)

        purine_ax.tick_params(axis='y', labelsize=self.tickLabelSize,
                length=self.tickLength, width=self.tickWidth, colors='green')

    def PYvsAvgKbtladder(self, indx, corr, pvals, PYs, pval_pos='high'):
        """
        The classic ladder plot, optinally including inset scatterplot
        """

        ax = self.getNextAxes()

        ax.yaxis.grid(True, which='major', color='lightgrey',
                      alpha=0.5, dashes=[1,1,1,1,1,1])

        # check for nan in corr (make it 0)
        corr, pvals = remove_nan(corr, pvals)

        its_max = indx[-1]
        #its_max = 15

        ## the ladder plot
        colr = brew_blue
        revCorr = [c*(-1) for c in corr]  # to get an increasing correlation
        #ax.plot(indx, revCorr, linewidth=self.lineSize, color=colr, marker='s',
                #markersize=3, label='Optimal $c_1, c_2, c_3$')
        ax.plot(indx, revCorr, linewidth=self.lineSize, color=colr, marker='s',
                markersize=2, markeredgecolor='none')

        if self.p_line:
            # sort pvals and corr and interpolate
            pv, co = zip(*sorted(zip(pvals, corr)))
            # interpolate pvalues (x, must increase) with correlation
            # (y) and obtain the threshold for p = 0.05
            #f = interpolate(pv[:-2], co[:-2], k=1)
            # other approach: just count till you get past 0.05
            pval_line_pos = np.argmax(np.asarray(pv[::-1]) < 0.05)
            #ax.axhline(y=(-1)*f(0.05), ls=':', color='r',
                                #label='p = 0.05 threshold', linewidth=2)
            ax.axvspan(0, pval_line_pos-1, color='darksalmon')

        ymin = -0.1
        #ymin = 0
        yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -1.1, -0.1)]
        yticklabels_skip = odd_even_spacer(yticklabels, oddeven='odd')

        ax.set_yticks(np.arange(ymin, 1.1, 0.1))
        ax.set_yticklabels(yticklabels_skip)
        ylim0 = -0.05
        ax.set_ylim(ylim0, 1.001)

        #ax.set_ylabel("Correlation: PY and $\overline{K}_{bt,i}$", size=self.labSize,
                     #color='blue')
        ax.set_ylabel("Correlation: PY and $\overline{K}_{bt,n}$", size=self.labSize)

        # set tick label size
        ax.tick_params(labelsize=self.tickLabelSize, length=self.tickLength,
                width=self.tickWidth)
        #ax.tick_params(axis='y', colors='blue')
        ax.tick_params(axis='y')

        # xticks
        xticks = range(2, its_max+1)
        xtickLabels = []
        for i in xticks:
            if i%2 == 0:
                xtickLabels.append(str(i))
            else:
                xtickLabels.append('')

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtickLabels)
        #ax.set_xticklabels([])
        ax.set_xlim(2-0.5, its_max)
        ax.set_xlabel("$n$", size=self.labSize)

        # bbox_to_anchor= x, y, width, height
        #if pval_pos == 'high':
            #yLoc = 0.85
        #elif pval_pos == 'low':
            #yLoc = 0.15

        #debug()

        #ax.legend(bbox_to_anchor=(0.8, yLoc, 0.2, 0.1), loc='best',
                    #prop={'size':4.5})

        return ax

    def randomShuffledITSPlot(self, shuffle_results):
        """
        Plot that shows that when randomly shuffling ITSs and optimizing for PY
        one does not get significant correlation coefficients.
        """

        ax = self.getNextAxes()

        indx, corr, corr_std, pvals = shuffle_results['Random']

        # change sign for correlation to get an 'upward' plot
        corr = [-c for c in corr]  # change sign

        # check for nan in corr (make it 0)
        corr, pvals = remove_nan(corr, pvals)

        ax.errorbar(indx, corr, yerr=corr_std,
                    linewidth=self.lineSize,
                    marker='*', markersize=3)

        indxMax = indx[-1] + 1
        indxMin = 2

        ymin = -0.5
        ymax = 1.1
        # yticks
        yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -ymax, -0.1)]
        yticklabels_skip = odd_even_spacer(yticklabels, oddeven='odd')

        yticks = np.arange(ymin, ymax, 0.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels_skip)
        ax.set_ylim(ymin, 1.1)
        ax.set_ylabel("Correlation: PY and $\overline{K}_{bt,n}$", size=self.labSize)

        # xticks
        xticklabels = [str(integer) for integer in range(indxMin, indxMax+1)]
        xticklabels_skip = odd_even_spacer(xticklabels, oddeven='even')

        ax.set_xticks(range(indxMin, indxMax))
        ax.set_xticklabels(xticklabels_skip)
        ax.set_xlim(indxMin-1, indxMax)
        ax.set_xlabel("$n$", size=self.labSize)

        # setting the tick font sizes
        ax.tick_params(labelsize=self.tickLabelSize)

        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)
        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

    def crossCorrPlot(self, analysis2stats):
        """
        Ladder-plot with random ITS and cross correlation
        """

        ax = self.getNextAxes()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.grid(True, which='major', color='lightgrey',
                      alpha=0.5, dashes=[1,1,1,1,1,1])

        colors = [brew_blue, brew_green, brew_purple]

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
                        color=colr)

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
                                     linewidth=self.lineSize, color=colr)

            elif analysis == 'Cross Correlation':
                lab = 'cross-validation'
                ax.errorbar(incrX, corr, yerr=corr_stds, label=lab,
                                     linewidth=self.lineSize, color=colr)

        indxMax = incrX[-1] + 1
        indxMin = 2

        ymin = -0.5
        ymax = 1.1
        # yticks
        yticklabels = [format(i,'.1f') for i in np.arange(-ymin, -ymax, -0.1)]
        yticklabels_skip = odd_even_spacer(yticklabels, oddeven='odd')

        yticks = np.arange(ymin, ymax, 0.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels_skip)
        ax.set_ylim(ymin, 1.1)
        ax.set_ylabel("Correlation: PY and $\overline{K}_{bt,n}$", size=self.labSize)

        # xticks
        xticklabels = [str(integer) for integer in range(indxMin, indxMax+1)]
        xticklabels_skip = odd_even_spacer(xticklabels, oddeven='even')

        ax.set_xticks(range(indxMin, indxMax))
        ax.set_xticklabels(xticklabels_skip)
        ax.set_xlim(indxMin-1, indxMax)
        ax.set_xlabel("$n$", size=self.labSize)

        ax.legend(loc='upper left', prop={'size':6})

        # setting the tick font sizes
        ax.tick_params(labelsize=self.tickLabelSize)

        #ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      #alpha=0.5)
        #ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      #alpha=0.5)


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


def cm2inch(cm):
    return float(cm)/2.54


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


def optimizeParam(ITSs, rna_range, testing, analysis, variableCombo, measure,
                  msat_normalization, msat_param_estimate,
                  randomize_method='random_gatc'):
    """
    Optimize c1, c2, c3 to obtain max correlation with PY/FL/TA/TR

    # equation:
    k1 = exp(-(c1*rna_dna_i + c2*dna_dna_{i+1} + c3*Dinuc_{i-1}) * 1/RT)

    PY = Productive Yield
    FL = Full Length
    TA = Total Abortive
    TR = Total RNA

    Default is to calculate the average Kbt up to msat, but the entire sequence
    can be taken into account of msat_normalization == false
    """

    # When using msat_param_estimate, only do parameter estimation for one
    # nucleotide.
    if msat_param_estimate:
        rna_range = range(20, 21)

    # grid size parameter value search space
    # XXX use 21/11 instead of 20/10 to evenly space your values!
    grid_size = 21
    if testing:
        #grid_size = 11
        grid_size = 6

    # Information about which analysis to run, and for CrossCorr and Random how
    # many iterations
    analysisInfo = {
            'Normal':{'run':False},
            'CrossCorr': {'run':False, 'nr_iterations':0},
            'Random': {'run':False, 'nr_iterations':0},
            }

    if analysis == 'Normal':
        analysisInfo['Normal']['run'] = True

    if analysis == 'Random':
        # nr of random sequences tested for each parameter combination
        analysisInfo['Random']['run'] = True
        #analysisInfo['Random']['nr_iterations'] = 100
        analysisInfo['Random']['nr_iterations'] = 40
        if testing:
            #analysisInfo['Random']['nr_iterations'] = 30
            analysisInfo['Random']['nr_iterations'] = 10

    elif analysis == 'Cross Correlation':
        # nr of samplings of 50% of ITSs for each parameter combination
        analysisInfo['CrossCorr']['run'] = True
        analysisInfo['CrossCorr']['nr_iterations'] = 40
        if testing:
            analysisInfo['CrossCorr']['nr_iterations'] = 10

    elif analysis != 'Normal':
        print('Give correct analysis parameter name')
        1/0

    # Parameter ranges you want to test out
    # NOTE: Using negative values works to nullify the bias toward positive
    # correlation in random DNA.
    # NOTE: the best would have been to sample randomly instead of going
    # through the hoops like that. You'd get a better representation of the
    # sample space :S
    if variableCombo['RNA']:
        c1 = np.linspace(0, 1.0, grid_size)
        #c1 = np.linspace(-1.0, 1.0, grid_size)
    else:
        c1 = np.array([0])

    if variableCombo['DNA']:
        c2 = np.linspace(0, 1.0, grid_size)
        #c2 = np.linspace(-1.0, 1.0, grid_size)
    else:
        c2 = np.array([0])

    if variableCombo['3N']:
        c3 = np.linspace(0, 1.0, grid_size)
        #c3 = np.linspace(-1.0, 1.0, grid_size)
    else:
        c3 = np.array([0])

    par_ranges = (c1, c2, c3)

    all_results = optim.main_optim(rna_range, ITSs, par_ranges, analysisInfo,
                                    measure, msat_normalization,
                                    msat_param_estimate, randomize_method)

    # return results for requested analysis type ('normal' by default)
    return all_results[analysis], grid_size, analysisInfo


def get_print_parameterValue_output(results, rna_range, onlySignCoeff,
        analysis, callingFunctionName, grid_size, msat_normalization,
        type_of_average, msat_param_estimate, aInfo=False, combo=''):
    """
    Analyze the weight coefficients to select three values from a set of
    contestants. Current method is the mean of the statistically significant
    values.

    There is an unhelathy coupling between writing output and getting parameter
    values for Keq-calculation. Should be separated.
    """

    logDir = 'outputLog'
    if not os.path.isdir(logDir):
        os.mkdir(logDir)

    analysisNoGap = analysis.replace(' ', '_')
    logFilePath = os.path.join('outputLog',
            '_'.join(['log', 'msat:'+str(msat_normalization), analysisNoGap,
                            callingFunctionName, 'grid:'+str(grid_size),
                            str(aInfo).replace(' ', '')]))

    # sometimes the same function calls, but with different combinations of the
    # free enregy variables
    if combo != '':
        logFilePath += '_' + str(combo)

    logFilePath += '.log'

    logFileHandle = open(logFilePath, 'wb')
    logFileHandle.write('calling function: ' + callingFunctionName + '\n')
    logFileHandle.write('combination of free energy variables: ' + str(combo) + '\n')
    logFileHandle.write('grid_size: ' + str(grid_size) + '\n')
    logFileHandle.write('msat_normalization: ' + str(msat_normalization) + '\n')
    logFileHandle.write('Range of RNA lengths: ' + str(rna_range) + '\n')
    logFileHandle.write('Type of average: ' + str(type_of_average) + '\n')
    logFileHandle.write('Significant coeff limit: ' + str(onlySignCoeff) + '\n')

    if aInfo:
        for asis, settings in aInfo.items():
            runThis = settings['run']
            logFileHandle.write('analysis: ' + asis + ' ' + str(runThis))
            if 'nr_iterations' in settings:
                howMany = settings['nr_iterations']
                logFileHandle.write(' nr: ' + str(howMany) + '\n')
    # output
    outp = {}

    minpvalues = [results[pos].pvals_min for pos in rna_range]
    meanpvalues = [results[pos].pvals_mean for pos in rna_range]
    medianpvalues = [results[pos].pvals_median for pos in rna_range]
    maxcorr = [results[pos].corr_max for pos in rna_range]
    meancorr = [results[pos].corr_mean for pos in rna_range]
    mediancorr = [results[pos].corr_median for pos in rna_range]

    # print the mean and std of the estimated parameters
    for param in ['c1', 'c2', 'c3']:
        # is params_best always what I want? For Normal, yes, but Random and
        # Cross-correlation? Look into params_best bearing in mind that this is
        # the value which is output

        # for cross-corr, params_best is not the best of the params! Neither is
        # it what you want to output !!!

        msat_estimate = 0

        # When using msat parameter estimation, use only the parameters from
        # full rna length, which is the maximum MSAT.
        # this means that there is no need for parameter estimation for
        # 3,4,...20-nt values. Just estimate for 20 and then use that value to
        # calculate all the other results. Will that require a lot of
        # refactoring?

        if msat_param_estimate:
            if analysis == 'Normal':
                msat_estimate = results[20].params_best[param]

            elif analysis in ['Cross Correlation', 'Random']:
                msat_estimate = results[20].params_mean[param]

        # for random you again want the median value ... this is
        if analysis == 'Normal':
            parvals = [results[pos].params_best[param] for pos in rna_range]

        elif analysis in ['Cross Correlation', 'Random']:
            parvals = [results[pos].params_mean[param] for pos in rna_range]

        if analysis == 'Normal':
            pvalues = minpvalues  # min of 20 best
        else:
            pvalues = medianpvalues  # median of X cross-corr, random

        # For calculation you want to keep
        # Issue: you screen out non-significant values, but that means that
        # you can't plot for combination with nonsignificant stuffs!
        # compromise: when the median pvalue is too high, use a median of all
        # values; when there is some significant stuff in there, do the
        # filtering. This is the only way to show on plots the effect of zero
        # correlation for some parameter combinations.
        medianAllPvalues = nanmedian(pvalues)
        ParameterValuesFiltered = []

        indx = 0
        for rna_length, pvalue in zip(rna_range, pvalues):
            # in this case, there are significant values, so we choose only
            # parameter values associated with significant correlation
            if (medianAllPvalues < onlySignCoeff) and (pvalue < onlySignCoeff):
                # don't consider rna-dna until after full hybrid length is reached
                if param == 'c1' and rna_length < 8:
                    ParameterValuesFiltered.append(0)
                else:
                    ParameterValuesFiltered.append(parvals[indx])
            # in this case, there are no significant values anyway, but we need
            # something to make a plot
            else:
                if param == 'c1' and rna_length < 8:
                    ParameterValuesFiltered.append(np.nan)
                else:
                    ParameterValuesFiltered.append(parvals[indx])

            indx += 1

        print param, ParameterValuesFiltered

        # ignore nan in mean and std calculations
        mean = nanmean(ParameterValuesFiltered)
        median = nanmedian(ParameterValuesFiltered)
        normal_std = nanstd(ParameterValuesFiltered)

        if ParameterValuesFiltered == []:
            madStd = np.nan
        else:
            madStd = mad_std(ParameterValuesFiltered)  # using the mad as an estimator
            if madStd.mask:
                madStd = np.nan
            else:
                madStd = float(madStd)

        outpString = '{0}: {1:.2f} (mean) +/- {2:.2f} or {0}: {3:.2f}'\
                ' (median) +/- {4:.2f}; msat(20): {5:.2f} '\
                        .format(param, mean, normal_std, median, madStd,
                                msat_estimate)
        print(outpString)
        logFileHandle.write(analysis + '\n')
        logFileHandle.write(outpString + '\n')

        # This is what is going into the keq calculation and plotting! Better
        # keep the median if that's what you're reporting in the figures.
        outp[param + '_mean'] = mean
        outp[param + '_median'] = median

    # print correlations
    if analysis == 'Normal':
        infoStr = "Normal analysis: max correlation and corresponding p-value"
        print(infoStr)
        logFileHandle.write(infoStr + '\n')
        print analysis

        for nt, c, p in zip(rna_range, maxcorr, pvalues):
            output = (nt, c, p)
            #print(output)
            logFileHandle.write(str(output) + '\n')
            print nt, c, p
    else:
        infoStr = "Random or cross-correlation: mean (median) correlation and p-values"
        print(infoStr)
        logFileHandle.write(infoStr + '\n')
        print analysis
        for nt, c, p, cm, pm in zip(rna_range, meancorr, meanpvalues, mediancorr,
                medianpvalues):
            output = (nt, c, p, cm, pm)
            logFileHandle.write(str(output) + '\n')
            print output

    logFileHandle.close()

    if type_of_average == 'mean':
        outp_param = outp['c1_mean'], outp['c2_mean'], outp['c3_mean']
    elif type_of_average == 'median':
        outp_param = outp['c1_median'], outp['c2_median'], outp['c3_median']
    else:
        print 'this is not right'
        1/0

    return outp_param


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
    For DG100 you need sum(AP[:10]) before you reach any decent correlation.
    That is up to 11-mer!!!! And the 14-mer also makes the difference.

    The 6-7 mer makes a big difference too.

    However!! Excluding the 2-mer and the 3-mer improves this drastically. The
    early AP do matter, but the 2-mer and 3-mer confound things. It could be
    that they are 'cheap' abortive products, because they happen very fast.

    It seems that sum(AP) early is almost positively correlated with PY! It's
    only when considering the late AP that the correlation between sum(AP) and
    PY becomes negative, like you'd expect.

    XXX
    High PY variants are likely to abort early, but unlikely to abort late.
    XXX

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

    XXXX ------- XXXX

    Update: you should plot all the basic correlations available in the
    dataserts: G, A, T, C with AP, raw abortive raw FL, combinations, throw at
    it all you've got.
    """

    py = [i.PY for i in ITSs]

    # XXX RESULT: the "A" part contributes more to correlation with PY
    # than the "G" part of the pyrimidines for dg100, and much more for dg400.
    # for DG400 correlation is significant starting at +4, dips at +5, and then
    # increases gradually to +11. Correlation with G isn't significant at all
    # until + 10 or so! But gradually increases from +8 all the way up to + 15.
    # for DG400 G is actually significantly ANTI correlated with PY at +3 and +4!
    # this is also true for DG100, but not significant, and -0.18 or so.
    # The DG400 stuff can be an artefact from biasing toward the 3rd and 4th
    # nucleotides by forcing the dinucleotide structure.
    # Still, i'd say that for DG100 your conclusion still stands. How big is
    # the AG difference in Hein and Malinen?
    # Considering the U > C > A > G for going to the backtracked state, for
    # DG100 U has a stronger anti-correlation with PY than
    # There is an assymetry here though:
    #   A is more strongly correlated than G -> together they are stronger
    #   C and T are similarly anti-correlated -> together they are stronger (or
    #   just the negative of the AG correlation)

    # At the same time, there is an interesting effect where there is no
    # correlation between the number of As and the number of Gs. This can
    # indicate that these two purines have a different
    # that these two purine values have

    # This can be result 1 in your work. Strong correlation with number of
    # purines. Has been shown ... G..., butstronger with A than with G.
    # Further, no correlation between A and G; if A and G were equally
    # beneficial to avoid paused and backtracked state, one would expect these
    # two values to have some correlation since they would be equally likely to
    # appear in the high-PY variants, and equally likely not to appear in the
    # low-PY variants.

    # What about abortive probabilities. Are they totally random ORWTF?

    #ladder_pre = [[1 if nuc in ['G', 'A'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    ladder_pre = [[1 if nuc in ['A'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder = [sum(ladder_pre[:i])]
    #xx=15
    #ladder_G = [[1 if nuc in ['G'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_A = [[1 if nuc in ['A'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_C = [[1 if nuc in ['C'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_T = [[1 if nuc in ['T'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_G = [[1 if nuc in ['G'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #ladder_A = [[1 if nuc in ['A'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #ladder_C = [[1 if nuc in ['C'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #ladder_T = [[1 if nuc in ['T'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #lad_np_G = [sum(ladder_G[i][:xx]) for i in range(len(ladder_pre))]
    #lad_np_A = [sum(ladder_A[i][:xx]) for i in range(len(ladder_pre))]
    #lad_np_C = [sum(ladder_C[i][:xx]) for i in range(len(ladder_pre))]
    #lad_np_T = [sum(ladder_T[i][:xx]) for i in range(len(ladder_pre))]
    #msats = [i.msat for i in ITSs]

    #nts = ('G', 'A', 'C', 'T')
    #nr_nts = (lad_np_G, lad_np_A, lad_np_C, lad_np_T)
    #for nt, nr_nt in zip(nts, nr_nts):
        #print nt, spearmanr(msats, nr_nt)

    #debug()
    avg_corr = []
    np_corr = []

    coeff = [0.25, 0.0, 0.55]
    for i in ITSs:
        i.calc_keq(*coeff, msat_normalization=True, rna_len=20)

    #rna_lenghts = range(2,10)

    #for x in rna_lenghts:
        #lad_np = [sum(ladder_pre[i][:x]) for i in range(len(ladder_pre))]
        ##avg_kbt = [np.mean(i.keq[:x]) for i in ITSs]
        #measure_values = [np.nanmean(i.keq[:x-1]) for i in ITSs]
        #np_corr.append(spearmanr(py, lad_np)[0])
        #avg_corr.append(spearmanr(py, measure_values)[0])

    rna_lenghts = range(12,16)
    for x in rna_lenghts:
        lad_np = [sum(ladder_pre[i][12:x]) for i in range(len(ladder_pre))]
        #avg_kbt = [np.mean(i.keq[:x]) for i in ITSs]
        measure_values = [np.nanmean(i.keq[12:x-1]) for i in ITSs]
        np_corr.append(spearmanr(py, lad_np)[0])
        avg_corr.append(spearmanr(py, measure_values)[0])

    fig, ax = plt.subplots()
    ax.plot(rna_lenghts, [-i for i in avg_corr], label='Average $K_{bt}$ up to RNA length')
    ax.plot(rna_lenghts, np_corr, label='Number of purines up to RNA length')

    ax.set_xlabel('RNA length')
    ax.set_ylabel('Spearman correlation with PY')
    ax.grid()

    ax.legend(loc='best')
    plt.show()


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
    indx = []

    # recall that 0 implies that no test was done.
    # you'll need to avoid this: remove all 0s
    no_zero_pvals = [p for p in pvals if p != 0]

    no_zero_pvals_sorted = sorted(no_zero_pvals)

    nr_tests = len(no_zero_pvals)

    for pvl in pvals:
        if pvl == 0:
            colors.append('k')
            indx.append(-1)
        else:
            # two index operations: first the index in the sorted
            if pvl < ((no_zero_pvals_sorted.index(pvl)+1)/nr_tests)*Q:
                colors.append(signcol)
                indx.append(1)
            else:
                colors.append(nonsigncol)
                indx.append(0)

    return colors, indx


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


def doFigureCalculations(fig2calc, pickDir, figures, coeffs, calculateAgain,
                        testing, dg100, dg400, msat_normalization,
                        average_for_plots_and_output, msat_param_estimate):
    """
    Perform all calculations necessary for plotting. Each figure can have
    several 'plot' functions.
    """

    def calcWrapper(name, msat_normalization, average_for_plots_and_output, dset=False,
            coeff=False, topNuc=False):
        """
        Wrapper for calculating and returning plot data. Pickling if necessary.
        """

        filePath = os.path.join(pickDir, name)

        if not os.path.isfile(filePath) or calculateAgain:

            # when recalculating, do not reuse any coefficients
            #if calculateAgain:
                #coeff = False

            calcr = Calculator(dg100, dg400, name, testing, dset, coeff,
                                topNuc, msat_normalization,
                                average_for_plots_and_output,
                                msat_param_estimate)
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
            # XXX is the topNuc used for something important? lets hope not.
            calcResults[subCalcName] = calcWrapper(subCalcName,
                    msat_normalization, average_for_plots_and_output,
                    topNuc=13, coeff=coeffs)
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


def raw_data_distribution(dg100, dg400):
    """
    Distribution of raw radiactive intensity across all ITSs
    """
    from pandas import DataFrame as df

    for dset_name, ITSs in [('DG100', dg100), ('DG400', dg400)]:
        #if dset_name == 'DG400':
            #for its in ITSs:
                #its.abortiveProb = its.abortiveProb*2
        dsetmean = np.nanmean([i.PY for i in ITSs])
        if dset_name == 'DG100':
            rna_range = range(2,21)
            ymax = 2*10**7
        else:
            rna_range = range(2,16)
            ymax = 0.2*10**7
        for division in ['low PY', 'high PY', 'all']:
            my_df = df()
            for rna_len in rna_range:
                if division == 'low PY':
                    my_df[rna_len] = [i.rawDataMean[rna_len-2] for i in ITSs
                            if i.PY < dsetmean]
                if division == 'high PY':
                    my_df[rna_len] = [i.rawDataMean[rna_len-2] for i in ITSs
                            if i.PY > dsetmean]
                if division == 'all':
                    my_df[rna_len] = [i.rawDataMean[rna_len-2] for i in ITSs]

            # -99 is nan here.
            my_df.replace('-99.0', value=np.nan, inplace=True)
            ax = my_df.plot(kind='box', ylim=(0, ymax))
            #ax = my_df.plot(kind='box')
            fig = ax.get_figure()
            fig.suptitle('Intensity distributions for {0} {1}'.
                    format(dset_name, division))
            dirname = 'Intensity_distributions'
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            filepath = os.path.join(dirname, dset_name + '_' +division + '.pdf')
            fig.savefig(filepath, format='pdf', size_inches=(9,15))


def ap_distribution(dg100, dg400):
    """
    Distribution of AP across all ITSs
    """

    from pandas import DataFrame as df

    for dset_name, ITSs in [('DG100', dg100), ('DG400', dg400)]:
        #if dset_name == 'DG400':
            #for its in ITSs:
                #its.abortiveProb = its.abortiveProb*2
        dsetmean = np.nanmean([i.PY for i in ITSs])
        if dset_name == 'DG100':
            rna_range = range(2, 21)
        else:
            rna_range = range(2, 16)

        for partition in ['low PY', 'high PY', 'all']:
            my_df = df()
            for rna_len in rna_range:
                if partition == 'low PY':
                    my_df[rna_len] = [i.abortiveProb[rna_len-2] for i in ITSs
                                      if i.PY < dsetmean]
                if partition == 'high PY':
                    my_df[rna_len] = [i.abortiveProb[rna_len-2] for i in ITSs
                                      if i.PY > dsetmean]
                if partition == 'all':
                    my_df[rna_len] = [i.abortiveProb[rna_len-2] for i in ITSs]

            # replace no signal with Nan. This way, you average those abortive
            # probabilities which actually exist (at position 20, 90% of itss
            # do not abort, but take the average AP of those who actually
            # reach)
            my_df = my_df.replace(0.0, np.nan)
            ax = my_df.plot(kind='box', ylim=(0, 0.8))
            fig = ax.get_figure()
            fig.suptitle('AP distributions for {0} {1}'.format(dset_name, partition))
            filepath = os.path.join('AP_distributions', dset_name + '_' + partition + '.pdf')
            fig.savefig(filepath, format='pdf', size_inches=(9, 15))


def dataset_to_file(dg100, name, write_AP=False):
    if write_AP:
        filepath = name + '_parameters_AP.txt'
    else:
        filepath = name + '_parameters.txt'

    filehandle = open(filepath, 'wb')
    for its in dg100:
        line = []
        line.append(its.name)
        line.append(its.sequence)
        line.append(str(its.PY))

        if write_AP:
            for ap in its.abortiveProb:
                line.append(str(ap))

        filehandle.write('\t'.join(line) + '\n')

    filehandle.close()


def consensus_pause_sites(dg100, dg400):
    """
    Consensus pause signal is -10G, -1Y, +1G. Assume it's because of the RNA-DNA
    hybrid, can you find this signal in the ITSs?

    'G[GATC]{8}[T,C]G' ?

    Result: vague preference for low to medium PY ITSs, but one example in DG100 of a
    high PY ITS. Conclusion: does not have explanatory power. A lot more
    sequences have 2 out of 3 matching, but neither here is there a clear
    pattern.
    """
    import re

    p = re.compile('G[GATC]{8}[T,C]G')
    #matches = p.findall('GGGGGGGGGCG')
    #nr_matches = len(matches)

    for name, dset in [('DG100', dg100), ('DG400', dg400)]:

        ma = []
        PY = []

        print('\n {0}\n'.format(name))

        for its in sortITS(dset):
            matches = p.findall(its.sequence[:15])
            nr_matches = len(matches)
            print its.PY, nr_matches

            ma.append(nr_matches)
            PY.append(its.PY)

        print(spearmanr(ma, PY))


def purine_vs_PY_scatter(dg100, dg400):
    """
    Reviewers want to see purine vs PY scatterplot. Do it for purines up to
    +15.
    """

    rcParams['figure.figsize'] = 8, 4  # This was the only way I could get this respected
    fig, axes = plt.subplots(ncols=2)
    ax_index = 0
    for dset_name, dset in [('DG100', dg100), ('DG400', dg400)]:
        purines = []
        PYs = []
        PYstd = []
        for its in dset:
            its.calc_purines()
            purines_15 = sum(its.purines[:15])

            purines.append(purines_15)
            PYs.append(its.PY)
            PYstd.append(its.PY_std)

        ax = axes[ax_index]
        ax.errorbar(purines, PYs, yerr=PYstd, fmt=None, ecolor=brew_gray, zorder=1,
                elinewidth=0.3)
        ax.scatter(purines, PYs, c='black', s=14, zorder=2, lw=0)
        ax.set_xlabel('Number of purines from +1 to +15')
        ax.set_ylabel('PY')
        ax.set_title(dset_name + ' library')

        if ax_index == 0:
            label = 'A'
        elif ax_index == 1:
            label = 'B'

        xpos = 0.03
        ypos = 0.97
        ax.text(xpos, ypos,
                label, transform=ax.transAxes, fontsize=18,
                fontweight='bold', va='top')

        ax_index += 1

    plt.tight_layout()
    #fig.savefig('figures/purine_py_scatter.pdf', format='pdf')
    fig.savefig('../../The-Tome/my_papers/rna-dna-paper/supplementary/figures/purine_py_scatter.pdf', format='pdf')


def greB_analysis(dg100):
    """
    Analyse the GreB+/- data from 2006 paper.
    """
    # These values are pretty different from the PY from quantitative data. How are they
    # obtained?
    table2_py_data = {
            'N25/A1': {'plus':     16.3, 'minus': 7.1},
            'N25': {'plus':        18.5, 'minus': 9.0},
            'DG115a': {'plus':     15.2, 'minus': 5.1},
            'DG127': {'plus':      13.4, 'minus': 3.5},
            'DG133': {'plus':      15.1, 'minus': 4.3},
            'N25anti': {'plus':    20.1, 'minus': 3.8},
            'DG137a': {'plus':     5.4,  'minus': 1.5},
            'DG154a': {'plus':     6.0,  'minus': 1.8},
            'N25/A1anti': {'plus': 6.0,  'minus': 1.5}}

    plus = [16.3, 18.5, 15.2, 13.4, 15.1, 20.1, 5.4, 6.0, 6.0]
    minus = [7.1, 9.0, 5.1, 3.5, 4.3, 3.8, 1.5, 1.8, 1.5]

    # XXX For the values in the table, the coefficient of variation is 30%
    # higher for GreB minus (statistically significant?). This
    # indicates that when using GreB, PY values become more similar. This
    # enforces the idea that abortive cycling drives low PY.
    # NOTE: the PY values given in Table 2 are high compared to the values
    # given in Table 1 of the paper. Using the values in Table 1, the GreB+
    # experiments have half the spread in PY compared to Table2.
    # You can mention this to Lilian. This is an argument that rate of
    # backtracking is not greatly sequence dependent.

    mean_plus = np.mean(plus)
    std_plus = np.std(plus)
    mean_minus = np.mean(minus)
    std_minus = np.std(minus)
    print(std_plus / mean_plus)
    print(std_minus / mean_minus)
    #return
    py_table1 = [i.PY for i in dg100 if i.name in table2_py_data]
    mean_table1 = np.mean(py_table1)
    std_table1 = np.std(py_table1)
    print(std_table1 / mean_table1)

    # Print PY for the ones you have GreB PY for
    purines = []
    for its in dg100:
        its.calc_purines()
        purines.append(sum(its.purines))
    for its in dg100:
        if its.name in table2_py_data:
            tabPy = table2_py_data[its.name]['minus']
            tabPyplus = table2_py_data[its.name]['plus']
            print(its.name + ' ', its.PY * 100, tabPy, tabPyplus)
            print(its.sequence)


def main():
    #remove_controls = True
    remove_controls = False

    dg100 = data_handler.ReadData('dg100-new')
    dg400 = data_handler.ReadData('dg400')

    if remove_controls:
        controls = ['DG133', 'DG115a', 'N25', 'N25anti', 'N25/A1anti']
        dg400 = [i for i in dg400 if i.name not in controls]
        #dg100 = [i for i in dg100 if not i.name in controls]

    # Normalize the AP -- removes correlation between sum(PY) and sum(AP)
    #normalize_AP(ITSs)

    # basic correlations
    #basic_info(dg100)
    #basic_info(dg400)
    ap_distribution(dg100, dg400)
    return
    #raw_data_distribution(dg100, dg400)
    #greB_analysis(dg100)

    #dataset_to_file(dg100, name='dg100', write_AP=True)
    #dataset_to_file(dg400, name='dg400')
    #return

    #consensus_pause_sites(dg100, dg400)

    # XXX: start here: create this figure, add to supplementary, build
    # supplementary, build new version, make a list of all your changes, and
    # send to Lilian et. al
    #purine_vs_PY_scatter(dg100, dg400)

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

    # Do not recalculate but use saved values (for tweaking plots)
    calculateAgain = False
    #calculateAgain = True

    # XXX warning if coeffs are set they are used instead of recalculating!
    # XXX afaik, these are only used for Figure2 (and 400-library). It would
    # not make sense to use them on the cross-corr and delineate plots :S
    #coeffs = [0.12, 0.14, 0.66]
    #coeffs = [0.14, 0.14, 0.69]
    #coeffs = [0.0, 0.0, 0.95]
    coeffs = [0.26, 0.0, 0.58]
    #coeffs = [0.25, 0.0, 0.55]
    #coeffs = [1.0, 1.0, 1.0]
    #coeffs = False  # False means coeffs will be calculated

    # when testing, run fewer iterations when estimating parameters
    #testing = True
    testing = False

    # Specify if the parameter values should be medians or means
    average_for_plots_and_output = 'mean'
    #average_for_plots_and_output = 'median'

    msat_normalization = True
    #msat_normalization = False

    # With this option set, only use optimal coeffs for rna_len=20; this
    # assumes MSAT has been enabled.
    msat_param_estimate = True
    #msat_param_estimate = False

    figures = [
            'Figure_PYvsAvgKbt',  # AvgKbt vs PY (in Paper)
            #'Figure_AP_Kbt',  # 1x2 Keq and AvgKbt vs AP (in Paper)
            #'CrossRandomDelineateSuppl',  # (in Paper)
            #'Figure_DG400_corr_ladder',  # DG400 scatter plot and ladder (in Paper)
            #'Figure_DinucleotideOrder',
            #'Figure_DG400_corr',  # DG400 scatter plot (in Paper)
            #'Figure3',  # Delineate + DG400
            #'Figure3_instantaneous_change',  # Delineate w/increase in correlation + DG400
            #'Figure4',  # delineate + cumul abortive, + keq-AP correlation
            #'Test',   # Random and cross-corrleation (supplementary)
            ############ Below: not figures, just single calculations
            #'DeLineate',   # Just to do the delineate calculation
            ]

    # Dependency between figure and calculations.
    # Commented out figures are not in paper currently
    fig2calc = {
            'Figure_PYvsAvgKbt': ['PYvsAvgKbt'],
            'Figure_AP_Kbt': ['AP_vs_Keq'],
            'Figure_DG400_corr_ladder': ['dg400_scatter_ladder'],
            'CrossRandomDelineateSuppl': ['crossCorrRandom', 'delineate', 'delineateCombo'],
            #'Figure_DG400_corr': ['dg400_validation'],
            #'Figure_DinucleotideOrder': ['Shuffler']
            #'Figure3': ['delineate', 'dg400_validation'],
            #'Figure3_instantaneous_change': ['delineate'],
            #'Figure4': ['AP_vs_Keq', 'PYvsAvgKbt', 'cumulativeAbortive'],
            #'Test':  ['crossCorrRandom'],
            #'DeLineate':  ['delineate', 'delineateCombo']
            }

    # for collecting figures you want to save to file
    saveMe = {}

    # save your calculations here for reuse
    # these data are reused for plotting when not recalculating results
    pickDir = 'pickledData'

    #plt.ion()
    plt.ioff()

    # Do calculations
    calcResults = doFigureCalculations(fig2calc, pickDir, figures, coeffs,
                                       calculateAgain, testing, dg100, dg400,
                                       msat_normalization,
                                       average_for_plots_and_output,
                                       msat_param_estimate)

    # Do plotting
    for fig in figures:

        plotr = None

        ##################### FIGURE SE vs PY ########################
        if fig == 'Figure_PYvsAvgKbt':

            topNuc, indx, corr, pvals, PYs, PYstd, value_pack, corr111, pvals111, corr_purine, pvals_purine = calcResults['PYvsAvgKbt']
            values_max, values_best, values_choice, choice = value_pack

            #debug()

            plotr = Plotter(YaxNr=1, XaxNr=2, plotName='values_15 vs PY',
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            # Set position for scatter plot
            ax_scatr = plotr.PYvsAvgKbtscatter(PYs, PYstd, values_choice, choice)
            ax_lad = plotr.PYvsAvgKbtladder(indx, corr, pvals, PYs)

            ax_scatr.yaxis.set_ticks_position('left')
            ax_scatr.xaxis.set_ticks_position('bottom')
            ax_lad.xaxis.set_ticks_position('bottom')

            # adding a visualization of what 1-1-1 looks like
            plotr.PYvsAvgKbtLadder111(ax_lad, indx, corr111, pvals111, PYs)

            # adding a visualization of the correlation with # of purines
            plotr.PYvsAvgKbtLadderPurines(ax_lad, indx, corr_purine, pvals_purine, PYs)

            # Try to leave out the legend
            #ax_lad.legend(prop={'size':5}, loc='lower right')

            # Set fig size
            plotr.setFigSize(current_journal_width, 4.5)

            #plt.tight_layout()
            plotr.figure.subplots_adjust(left=0.08, top=0.95, right=0.87,
                    bottom=0.17, wspace=0.35)
            #plotr.figure.subplots_adjust(wspace=0.45)
            # make the correlation label come closer
            ax_lad.yaxis.labelpad = 1
            ax_scatr.yaxis.labelpad = 1

            # Only keep bottom and left ticks and axes
            # Hide the right and top spines
            #ax_scatr.spines['right'].set_visible(False)
            #ax_scatr.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines

            plotr.addLetters(shiftX=0)

            saveMe['PYvsAvgKbt'] = plotr.figure

        if fig == 'Figure3_instantaneous_change':

            # delineation of effects of different energies
            delinResults = calcResults['delineate']

            plotr = Plotter(plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=2, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            plotr.delineatorPlot(delinResults, inst_change=True)  # plot the delineate plot

            # Should be 8.7 cm
            plotr.setFigSize(current_journal_width, 4.7)
            plotr.figure.subplots_adjust(left=0.13, top=0.98, right=0.99,
                    bottom=0.18, wspace=0.4)

            saveMe['Delineate_Instantaneous_change'] = plotr.figure

        ###################### FIGURE DELINEATE + DG400 ########################
        # A figure that combines the 'delineate' and scatter plot for DG400 figures
        if fig == 'Figure3':

            # delineation of effects of different energies
            delinResults = calcResults['delineate']
            # the DG400 scatter plot
            calcdDG400, SE15, PY, PYstd = calcResults['dg400_validation']

            plotr = Plotter(YaxNr=1, XaxNr=2, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            plotr.delineatorPlot(delinResults)  # plot the delineate plot
            plotr.dg400validater(calcdDG400, SE15, PY, PYstd)

            # Should be 8.7 cm
            plotr.setFigSize(current_journal_width, 4.7)
            plotr.figure.subplots_adjust(left=0.13, top=0.98, right=0.99,
                    bottom=0.18, wspace=0.4)

            plotr.addLetters()

            saveMe['Delineate_And_DG400'] = plotr.figure

        ###################### FIGURE  DG400 ########################
        # A figure with scatter plot for DG400 figures
        if fig == 'Figure_DG400_corr_ladder':

            dg400, rna_range, values, PY, PYstd, corr_kbt, pvals_kbt, corr_purine, pvals_purine = calcResults['dg400_scatter_ladder']

            plotr = Plotter(YaxNr=1, XaxNr=2, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            ax_sct = plotr.dg400validater(dg400, values, PY, PYstd)
            ax_lad = plotr.PYvsAvgKbtladder(rna_range, corr_kbt, pvals_kbt, PY)

            ax_sct.yaxis.set_ticks_position('left')
            ax_sct.xaxis.set_ticks_position('bottom')
            ax_lad.xaxis.set_ticks_position('bottom')

            plotr.PYvsAvgKbtLadderPurines(ax_lad, rna_range, corr_purine, pvals_purine, PY)

            # Should be 8.7 cm
            plotr.setFigSize(current_journal_width, 4.5)
            plotr.figure.subplots_adjust(left=0.09, top=0.98, right=0.88,
                    bottom=0.18, wspace=0.4)

            ax_lad.yaxis.labelpad = 1
            ax_sct.yaxis.labelpad = 1

            plotr.addLetters(shiftX=0)

            saveMe['DG400_scatr_ladr'] = plotr.figure

        ###################### FIGURE  DG400 ########################
        # A figure with scatter plot for DG400 figures
        if fig == 'Figure_DG400_corr':

            # the DG400 scatter plot
            calcdDG400, SE15, PY, PYstd = calcResults['dg400_validation']

            plotr = Plotter(YaxNr=1, XaxNr=1, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            plotr.dg400validater(calcdDG400, SE15, PY, PYstd)

            # Should be 8.7 cm
            plotr.setFigSize(5.7, 4.2)
            plotr.figure.subplots_adjust(left=0.15, top=0.98, right=0.99,
                    bottom=0.18, wspace=0.4)

            saveMe['DG400'] = plotr.figure

        if fig == 'Figure4':

            """
            Display the cumulative AP (minus +2) and delineate as well as the
            nt-2-nt correlation (which doesn't match as well).
            """
            results2 = calcResults['cumulativeAbortive']
            topNuc, indx, corr, pvals, PYs, PYstd, SEmax, SEbest = calcResults['PYvsAvgKbt']

            #plotr = Plotter(YaxNr=1, XaxNr=2, plotName=fig,
                    #p_line=True, labSize=6, tickLabelSize=6, lineSize=2,
                    #tickLength=2, tickWidth=0.5)
            plotr = Plotter(YaxNr=1, XaxNr=2, plotName=fig,
                    p_line=True, labSize=8, tickLabelSize=8, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            # ladder plut with cumul abortive
            ax_lad = plotr.PYvsAvgKbtladder(indx, corr, pvals, PYs, SEbest, topNuc,
                                        inset=False, pval_pos='low')
            plotr.cumulativeAbortive(*results2, corr=corr)

            # bar plot
            resAPvsKeq = calcResults['AP_vs_Keq']['Non-Normalized']
            axM1 = plotr.moving_average_ap_keq(resAPvsKeq)

            #plotr.setFigSize(current_journal_width, 4.5)
            plotr.setFigSize(17.7, 6.5)
            #plt.tight_layout()
            #plotr.figure.subplots_adjust(left=0.11, top=0.97, right=0.995,
                    #bottom=0.18, wspace=0.33)
            plotr.figure.subplots_adjust(left=0.06, top=0.97, right=0.995,
                    bottom=0.15, wspace=0.33)

            # make the correlation label come closer
            axM1.yaxis.labelpad = 0.4
            ax_lad.yaxis.labelpad = 1

            letters = ('A', 'B')
            positions = ['UL', 'UR']
            plotr.addLetters(letters, positions)

            # Add a little grid

            saveMe[fig] = plotr.figure

        if fig == 'Figure_AP_Kbt':

            """
            Display the sum(AP)-avgKbt correlation as well as the nt-2-nt
            correlation between AP and Keq.
            """
            plotr = Plotter(YaxNr=1, XaxNr=1, plotName=fig,
                    p_line=True, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            # bar plot
            resAPvsKeq = calcResults['AP_vs_Keq']['Non-Normalized']
            ax = plotr.moving_average_ap_keq(resAPvsKeq)

            plotr.setFigSize(current_journal_width-2.5, 4.2)
            #plt.tight_layout()
            plotr.figure.subplots_adjust(left=0.14, top=0.97, right=0.995,
                    bottom=0.19, wspace=0.33)

            ax.yaxis.labelpad = 0.4

            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey',
                          alpha=0.5, dashes=[1,1,1,1,1,1])
            ax.set_axisbelow(True)

            saveMe[fig] = plotr.figure

        ###################### FIGURE Cross-corr and Random ########################
        # Used to be in the main paper, now supplementary
        if fig == 'Figure_DinucleotideOrder':

            plotr = Plotter(YaxNr=1, XaxNr=1, plotName=fig,
                    p_line=False, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            shuffle_results = calcResults['Shuffler']

            plotr.randomShuffledITSPlot(shuffle_results)

            plotr.setFigSize(current_journal_width, 5)

            plotr.figure.subplots_adjust(left=0.1, top=0.97, right=0.98,
                    bottom=0.12, wspace=0.35, hspace=0.1)

            saveMe[fig] = plotr.figure

        if fig == 'CrossRandomDelineateSuppl':

            plotr = Plotter(YaxNr=1, XaxNr=3, plotName=fig,
                    p_line=False, labSize=6, tickLabelSize=6, lineSize=1,
                    tickLength=2, tickWidth=0.5)

            # delineation of effects of different energies
            delinResults = calcResults['delineate']
            # cross corrrlation and random
            crossCorrResults = calcResults['crossCorrRandom']
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

