"""
New, class based attempt at getting some rigour into creating figures for your
papers. You need more flexibility in terms of subfiguring and not re-running
simulations just to change figure data.

You also need to do some new calculations based on DG3D and DGbubble and DG
hybrid.
"""
from __future__ import division

########### Make a debug function ##########
from ipdb import set_trace as debug  # NOQA

import data_handler
import numpy as np
import matplotlib.pyplot as plt
import optim
import os
from scipy.stats import spearmanr, pearsonr, nanmean, nanstd, nanmedian  # NOQA
from operator import attrgetter

# Global variables :)
#here = os.getcwd()  # where this script is executed from! Beware! :)
here = os.path.abspath(os.path.dirname(__file__))  # where this script is located
fig_dir1 = os.path.join(here, 'figures')
fig_dir2 = '/home/jorgsk/Dropbox/The-Tome/my_papers/rna-dna-paper/figures'
fig_dirs = (fig_dir1, fig_dir2)


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


class PlotObject(object):
    """
    Holds the the information needed to plot
    """

    def __init__(self, x_ticks):
        self.plotName = ''
        self.xTicks = x_ticks
        self.yTicks = 0
        self.xTickFontSize = 0
        self.xTickFontSize = 0

        self.nrSubolots = 1

        self.imageSize = [1, 4]


class AP(object):
    def __init__(self, keq, ap, dg3d=-1, dgDna=-1, dgRna=-1):
        self.keq = keq
        self.dg3d = dg3d
        self.ap = ap
        self.dgDna = dgDna
        self.dgRna = dgRna


def visual_inspection(ITSs, variable='AP'):
    """
    1) Bar plot of the abortive probabilities 3 by 3
    """
    plt.ion()

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


def optimize(ITSs, testing=True, target='PY', analysis='Normal'):
    """
    Optimize c1, c2, c3 to obtain max correlation with PY/FL/TA/TR

    As of right now, you only have PY for DG100 ... would be interesting to get
    FL in there as well. Do you have this information? Maybe verify with Lilian.

    PY = Productive Yield
    FL = Full Length
    TA = Total Abortive
    TR = Total RNA
    """

    # grid size
    grid_size = 20

    if testing:
        grid_size = 6

    # assume normal and no randomization or cros correlation analysis
    randize = 0
    crosscorr = 0

    if analysis == 'Normal':
        normal = True
    elif analysis == 'Random':
        randize = grid_size
        normal = False
    elif analysis == 'Cross Correlation':
        crosscorr = grid_size
        normal = False
    else:
        print('Give correct analysis parameter name')
        1/0

    # only use PY for correlation kthnkx
    #targetAttr = {'PY': 'PY',
                  #'FL': 'FL',
                  #'TA': 'totAbort',
                  #'TR': 'totRNA'
                  #}

    # Parameter ranges you want to test out
    c1 = np.array([1])  # insensitive to variation here
    c2 = np.linspace(0, 1.0, grid_size)
    c3 = np.linspace(0, 1.0, grid_size)
    c4 = np.linspace(0, 1.0, grid_size)

    par_ranges = (c1, c2, c3, c4)

    its_min = 2
    its_max = 21
    #if testing:
        #its_max = 15

    its_range = range(its_min, its_max)
    #its_range = [its_min, 5, 10, 15, its_max]

    all_results = optim.main_optim(its_range, ITSs, par_ranges, randize,
            crosscorr, normal)

    # extract the results where the full dataset has been used
    if analysis == 'Normal':
        results = all_results[0]
    elif analysis == 'Random':
        results = all_results[1]
    elif analysis == 'Cross Correlation':
        results = all_results[2]
    else:
        print('WTF mate')

    outp = {}

    # print the mean and std of the estimated parameters
    for param in ['c1', 'c2', 'c3', 'c4']:
        parvals = [results[pos].params_best[param] for pos in its_range]
        pvals = [results[pos].pvals_min for pos in its_range]
        meanpvals = [results[pos].pvals_mean for pos in its_range]
        maxcorr = [results[pos].corr_max for pos in its_range]
        meancorr = [results[pos].corr_mean for pos in its_range]

        # only report parameter values which correspond to significant correlation
        significantParvals = []
        for ix, pval in enumerate(pvals):
            if pval > 0.05:
                continue

            # don't consider rna-dna until after full hybrid length is reached
            if param == 'c2' and ix < 8:
                significantParvals.append(0)
            else:
                significantParvals.append(parvals[ix])

        print param, significantParvals

        # ignore nan in mean and std calculations
        mean = nanmean(significantParvals)
        median = nanmedian(significantParvals)
        std = nanstd(significantParvals)

        print('{0}: {1:.2f} (mean) or {0}: {2:.2f} (median) +/- '
                '{3:.2f}'.format(param, mean, median, std))

        outp[param] = mean

    # print correlations
    if analysis == 'Normal':
        print("Normal analysis: max correlation and corresponding p-value")

        for nt, c, p in zip(its_range, maxcorr, pvals):
            print nt, c, p
    else:
        print("Random or cross-correlation: mean correlation and pvalues")
        for nt, c, p in zip(its_range, meancorr, meanpvals):
            print nt, c, p

    return outp['c2'], outp['c3'], outp['c4']


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

    Return arrays of AP and DG3D values in different batches. For example, the
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

    Return arrays of AP and DG3D values in different batches. For example, the
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
    plt.ion()

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

    plt.ion()

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
    plt.ion()

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
    plt.ion()

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
    window with the PY/FL/TA
    """

    #dset = dg100
    dset = dg400

    movSize = 1

    py = [i.PY for i in dset]
    fl = [i.fullLengthMean for i in dset]
    ta = [i.totAbortMean for i in dset]

    xmers = range(2, 21)
    attr = 'abortiveProb'

    # get a moving average window for each center_position
    # calculating from an x-mer point of view
    plim = 0.05
    # Use a less conservative test for the paper

    for label, meas in [('PY', py), ('FL', fl), ('TotalAbort', ta)]:
    #for label, meas in [('FL', fl), ('TotalAbort', ta)]:

        # the bar height will be the correlation with the above three values for
        # each moving average center position
        bar_height = []
        pvals = []

        nr_tests = 0

        for mer_center_pos in xmers:
            movArr = get_movAv_array(dset, center=mer_center_pos,
                    movSize=movSize, attr=attr)

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

        ax.set_xlabel('Center x-mer for moving average')
        ax.set_ylabel('{0} -- {1}: window size: '
                        '{2}'.format(label, attr, movSize))


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


def moving_average_ap_keq(dg100, dg400):

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
    plt.ion()

    #dset = dg100
    #dset = dg400
    dset = dg100 + dg400

    normalize_AP(dset)

    movSize = 0

    xmers = range(2, 21)

    # get a moving average window for each center_position
    # calculating from an x-mer point of view
    Q = 0.05  # minimum false discovery rate
    # Use a less conservative test for the paper

    # the bar height will be the correlation with the above three values for
    # each moving average center position
    bar_height = []
    pvals = []

    for mer_center_pos in xmers:

        attr = 'abortiveProb'
        #movAP = get_movAv_array(dset, center=mer_center_pos, movSize=movSize, attr=attr)
        movAP = get_movAv_array(dset, center=mer_center_pos, movSize=0,
                attr=attr, prePost='both')

        attr = 'keq'
        movKeq = get_movAv_array(dset, center=mer_center_pos, movSize=movSize,
                attr=attr, prePost='both')
                #attr=attr, prePost='post')

        corr, pval = spearmanr(movAP, movKeq)

        bar_height.append(corr)
        pvals.append(pval)

        print mer_center_pos, corr, pval

    # Add colors to the bar-plots according to the benjamini hochberg method
    # sort the p-values from high to low; test against plim/

    colors = benjami_colors(pvals, Q=Q, nonsigncol='black', signcol='gray')

    # plotit
    fig, ax = plt.subplots()
    ax.bar(left=xmers, height=bar_height, align='center', color=colors)
    ax.set_xticks(xmers)
    ax.set_xlim(xmers[0]-1, xmers[-1])

    ax.set_xlabel('ITS position', size=10)
    ax.set_ylabel('Correlation coefficient'.format(movSize), size=10)

    for l in ax.get_xticklabels():
        l.set_fontsize(9)
    for l in ax.get_yticklabels():
        l.set_fontsize(9)

    filename = 'AP_vs_Keq'

    def mm2inch(mm):
        return float(mm)/25.4

    # Should be 8.7 cm
    # 1 inch = 25.4 mm
    width = mm2inch(127)
    height = mm2inch(97)
    fig.set_size_inches(width, height)

    for fdir in fig_dirs:
        #for formt in ['pdf', 'eps', 'png']:
        for formt in ['pdf']:

            name = filename + '.' + formt
            odir = os.path.join(fdir, formt)

            if not os.path.isdir(odir):
                os.makedirs(odir)

            fig.savefig(os.path.join(odir, name), transparent=True, format=formt)


def main():

    remove_controls = True

    dg100 = data_handler.ReadData('dg100-new')  # read the DG100 data
    dg400 = data_handler.ReadData('dg400')  # read the DG100 data

    ITSs = dg100

    if remove_controls:
        controls = ['DG133', 'DG115a', 'N25', 'N25anti']
        dg400 = [i for i in dg400 if not i.name in controls]

    # lower resolution while testing
    #testing = False
    #testing = True

    # Add keq-values by first calculating c1, c2, c3
    for ITSs in [dg100, dg400]:
        #c1, c2, c3 = optimize(ITSs, testing, target='PY', analysis='Normal')
        c1, c2, c3 = [0.23, 0.07, 0.95]

         #add the constants to the ITS objects and calculate Keq
        for its in ITSs:
            its.calc_keq(c1, c2, c3)

    # Return c1, c2, c3 values obtained with cross-validation
    #optimize(dg100, testing, target='PY', analysis='Cross Correlation')
    #optimize(dg100, testing, target='PY', analysis='Normal')

    #ITSs = dg100
    #ITSs = dg400

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

    # XXX ap keq for all positions
    #plus_minus_keq_ap(dg100, dg400)

    # XXX ap keq at each position
    #position_wise_correlation_shifted(dg100, dg400)

    # XXX bar plot of the sum of AP and the sum of raw
    #apRawSum(dg100, dg400)

    # XXX moving average of AP vs PY/FL/TA
    #moving_average_ap(dg100, dg400)

    # XXX moving average between AP and Keq
    moving_average_ap_keq(dg100, dg400)

    return ITSs


if __name__ == '__main__':
    ITSs = main()
    ga = ITSs[0]  # just for testing attributes in the interpreter...
