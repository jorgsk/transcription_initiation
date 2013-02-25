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
from scipy.stats import spearmanr


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


def optimize(ITSs, testing=True, target='PY'):
    """
    Optimize c1, c2, c3 to obtain max correlation with PY/FL/TA/TR

    As of right now, you only have PY for DG100 ... would be interesting to get
    FL in there as well. Do you have this information? Maybe verify with
    Lilian.

    PY = Productive Yield
    FL = Full Length
    TA = Total Abortive
    TR = Total RNA
    """

    # grid size
    grid_size = 15

    if testing:
        grid_size = 6

    targetAttr = {'PY': 'PY',
                  'FL': 'FL',
                  'TA': 'totAbort',
                  'TR': 'totRNA'
                  }

    # -> you depend on the order .. maybe you can fix this now once and for
    # all? :) or just do what works ...
    y = np.array([getattr(its, targetAttr[target]) for its in ITSs])*0.01

    # Parameter ranges you want to test out
    c1 = np.array([1])  # insensitive to variation here
    c2 = np.linspace(0, 1.0, grid_size)
    c3 = np.linspace(0, 1.0, grid_size)
    c4 = np.linspace(0, 1.0, grid_size)

    #c2 = np.array([0.33]) # 15 runs
    #c3 = np.array([0]) # 15 runs
    #c4 = np.array([0.95]) # 15 runs

    par_ranges = (c1, c2, c3, c4)

    its_max = 21
    #if testing:
        #its_max = 15

    its_range = range(2, its_max)
    #its_range = [5,10, 15, 20]

    all_results = optim.main_optim(y, its_range, ITSs, par_ranges)

    # extract the results where the full dataset has been used
    results = all_results[0]

    outp = {}

    # print the mean and std of the estimated parameters
    for param in ['c1', 'c2', 'c3', 'c4']:
        #get the parameter values from pos 6 to 21
        # TODO should get the ones for which there is significant correlation
        par_vals = [results[pos].params_best[param] for pos in range(6, its_max)]
        print param, par_vals

        if param == 'c2':
            mean = np.mean(par_vals[8:])
            std = np.std(par_vals[8:])
        else:
            mean = np.mean(par_vals)
            std = np.std(par_vals)

        print('{0}: {1:.2f} +/- {2:.2f}'.format(param, mean, std))

        outp[param] = mean

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
    from operator import attrgetter

    ITSs = sorted(ITSs, key=attrgetter(attribute))

    return ITSs


def add_keq(ITSs):
    """
    Calculate keq for the different ITSs
    """

    #c1, c2, c3 = optimize(dg100, testing, target='PY')
    c1, c2, c3 = [0.33, 0.07, 0.82]
    c1, c2, c3 = [0.13, 0.07, 0.92]

    # add the constants to the ITS objects and calculate Keq
    for its in ITSs:
        its.calc_keq(c1, c2, c3)

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

    scrunches = separateByScrunch(ITSs, 'DgDNA15', nr=nr, upto=upto)

    arranged = {}

    for lab, ITSs in scrunches.items():

        aProbs = []
        keqs = []

        fro = 1
        to = 10

        # build arrays
        for its in ITSs:
            for (ap, keq) in zip(its.abortiveProb[fro:to], its.keq[fro:to]):

                aProbs.append(ap)
                keqs.append(keq)

        arranged[lab] = (aProbs, keqs)

    return arranged


def getDinosaur(ITSs, upto=15):
    """
    Sort values with the DNA DNA value up to that point

    Optionally subtract the RNA-DNA value (length 10 RNA-DNA)

    How to do it? Make a basic class for AP! This makes it super-easy to change
    sortings. Just loop through a keyword.
    """
    class AP(object):
        def __init__(self, ap, dgDna, dgRna, dgDnaNorm, dgRnaNorm):
            self.ap = ap
            self.dgDna = dgDna
            self.dgRna = dgRna
            self.dgDnaNorm = dgDnaNorm
            self.dgRnaNorm = dgRnaNorm

            self.dgControl = dgDna - dgRna
            self.dgControlNorm = dgDnaNorm - dgRnaNorm

    arranged = {}
    label = 'dinosaur'

    # add objects here and sort by the dgdna-score
    objects = []

    for its in ITSs:
        for pos in range(1, upto):

            # correct for a max 10 nt rna-dna hybrid
            rnaBeg = min(pos-10, 0)

            apObj = AP(its.abortiveProb[pos],
                       sum(its.dna_dna_di[:pos]),
                       sum(its.rna_dna_di[rnaBeg:pos]),
                       sum(its.dna_dna_di[:pos])/float(pos),
                       sum(its.rna_dna_di[rnaBeg:pos])/float(pos)),

            objects.append(apObj)

    # TODO! sort objects by one of your metrices and offYago!




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

    """
    plt.ion()

    #fig, ax = plt.subplots()

    if method == 'initial':
        arranged = getITSdnaSep(ITSs, nr='two', upto=upto)

    elif method == 'dinosaur':
        arranged = getDinosaur(ITSS, upto=upto)

    for (labl, (aprobs, keqs)) in arranged.items():

        # separate between a series of plots and just a single value?

        corr, pvals = spearmanr(aprobs, keqs)

        print(labl)
        print('Corr: {0}, pval: {1}\n'.format(corr, pvals))


def ap_other(ITSs):
    """
    Correlate the AP with whatever you can come up with DGDNA/RNA G, A, T,C and
    so on. Basically showing that the AP don't depend on a lot of shait.
    """


def main():

    remove_controls = True

    dg100 = data_handler.ReadData('dg100-new')  # read the DG100 data
    #dg100_Old = data_handler.ReadData('dg100')  # read the DG100 data
    dg400 = data_handler.ReadData('dg400')  # read the DG100 data

    if remove_controls:
        controls = ['DG133', 'DG115a', 'N25', 'N25anti']
        dg400 = [i for i in dg400 if not i.name in controls]

    ITSs = dg100
    #ITSs = dg400

    add_keq(ITSs)

    #ITSs = sortITS(ITSs, 'PY')

    #print('Sorted by PY:')
    #fold_difference(ITSs)
    ## Plot basic info (FL, PY, total RNA) for the different quantitations
    #data_overview(ITSs)
    #print()

    ## do the same but now sorted by Keq
    #ITSs = sortITS(ITSs, 'SE')
    ##ITSs = sortITS(ITSs, 'name')

    #print('Sorted by SE:')
    #fold_difference(ITSs)
    ## Plot basic info (FL, PY, total RNA) for the different quantitations
    #data_overview(ITSs)

    # plot correlation between Keq and AP and Raw
    # add possibility for screening for high/low dnadna dnarna values
    keq_ap_raw(ITSs)

    # try to correlate the AP to other values, such as the DGDNA up to that
    # point, and/or the DGDNA + DGRNA up to that point
    ap_other(ITSs)

    # When you have these two plots -- what is your story? First you need to
    # find out if your theory about dgdna dng rna scrunched complex works.
    # in general XXX try to introduce energy in scrunched complex, but be
    # explicit that there could somehow be other iniosyncrhic effects of
    # scrunching DNA.

    # OK is this a dead end? there is more fold variation within abortive RNA
    # than within FL. However, variation in PY correlates just as much with FL
    # as abortive RNA. We can say howver: by varying the ITS, one obtains a
    # higher fold-variation in abortive product than in full length product,
    # indicating that the nucleotide variation leads to greatest variation in
    # overall aborted product. (Can I show this graphically? how important will
    # it be?)

    # RESULT: shit, it looks like variation in FL is driving variation in SE15
    # TODO: find another way of quantifying the linear fold variation?
    # It seems that with your current way of calculating Keq (c3=1 etc) you
    # don't get the same order of the DG4xx promoters. That's one thing.
    # Another is that after re-ordering according to Keq, it seems that
    # there is more variation in FL along the Keq-dimension than in TA.
    # This goes contrary to what you had hypothesized. It's a bit of a blow for
    # your theory.

    # TODO make this agnostic to sizes in the ITS list
    #visual_inspection(ITSs, variable='Raw')

    # RESULT: variation in abortive RNA drives variation in PY

    # RESULTS: some observations (obs: they are sensitive to the first/second
    # quantitation -- they are valid for the first and the mean(first, second):
    # * The outlier DG434 has unusually much abortive transcript and very
    #   little full length
    # * The "first" 4 predicted itss have pretty high FL -- but they also have
    #   high A, which gives them an overall low PY
    # * After DG435 total A transcript remains more or less the same,
    #   while the amount of FL transcript increases -- this causes the increase
    #   in PY. Any model would actually need this raw information... lol

    # Print the mean and std of FL, PY, total RNA for the different quants
    #quant_stats(ITSs, sortby='Keq')

    # Check if the 3D positions with high AP/Raw data have different DGDNA or
    # DGRNA than the 3D positions with low AP/Raw

    # Select 3D positions from check DG energies -0 -1 and -2 from the site --
    # more minus the longer the x-mer?

    # If nothing works... what is your exit strategy? Show that Keq, and
    # neither DGBUBBle nor DGHYBRID correlate with the abortive probabilities

    # 1) Use the DG100 series to get the weights
    #testing = True  # Fast calculations
    # c1: dna, c2: rna, c3: 3'dinuc

    # 2) Use the DG400 series to correlate Keq with AP and Raw
    # 3) Do those with low AP have DNA-bubble RNA-DNA energy up to that point?

    return ITSs

    """
    i) How does the different delta G values correlate with AP (-2, -1, 0, +1,
    +2) around the

    You need to present a story right? So let's start there. The first step

    1) You found that PY correlates with Keq through optimization

    2) You reason that this is because occupation of the pre-translocated state
    increases the chance of backtracking (cite several papers, incl. 2012),
    leading to increased abortive initiation (backed up by the fact that)
    variation in abortive RNA drives the variation in PY (can I give a
    percentage?)

    3) Therefore, you check out the abortive probabilities. However, comparing
    the Keq values to AP at each position gives zero correlation.

    4) Therefore, you postulate that while the DG 3'end sequence may predispose
    for the initial backtracking step, the subsequence steps of bubble collapse
    might be sequence dependent in some different way, like through DG-DNA or
    DG-RNADNA

    5) Hey -- these variants have the same promoter -- yet their total RNA,
    total aborted, and total FL vary between the variants. Depending on the
    speed of the aborted rate, it takes more time to synthesize a FL transcript
    than to abort. If transcript abortivity happens fast, you'd expect more
    total RNA for the high-aborters. If transcript abortivity is slow, you'd
    expect less total RNA for the high-aborters.

    Can you classify them in other interesting cases? What does it mean if five
    promoters have the same FL but different total abortive? I think it means
    that bubble collapse happens more easily for those that have high abortive?
    But if bubble collapse happens more easily, these promoters also
    re-initiate faster, which may increase the FL as well.

    It would be nice to have a state-driven model (Xie et al...) where one
    could try to tweak these different values.

    The AP share this assumption: that bubble collapse happens at the same rate
    given a backtracking event. If bubble collapse is reduced in speed at a
    given position (like for a 13 nt compared to a 4 nt -- or the other way
    around!) then the AP is not valid any more. There may be a lot of
    backtracking happening that is not recorded.

    Can information about this be had from the GreB+ experiments? What is the
    expected effect of GreB -- cleave to restart elongation. This means that
    only the quickest backtracking events will pass through. If backtracking
    was super-duper quick, GreB could not have any effect. Hence, there must be
    some intermediary state between backtracking and bubble collapse where GreB
    has the time to act. The half-life of this intermediary state can make a
    big contribution on the AP.

    Compare two backtracking events: both will reduce the # of FL transcript by
    1. However, if one backtracked complex dwells long in an intermedate state,
    the abortive transcript of length x will be less for this ITS ->
    paradoxically giving this variant a higher PY! :S --> however, the other
    one who collapses his bubble fast will more quicker have a new shot at
    producing another FL, which may increase the final FL value, thereby
    equalizing this? All this is not certain and a rigorous mathematical
    analysis must be employed to understand it.

    Can you anyway try to say something about the dwell times of the
    backtracked complexes? Is it likely that the dwell-time depends on the free
    energy of the RNA-DNA hybrid and the DNA-DNA bubble? A strong hybrid
    prolongs the dwell time and a weak bubble shortens it?

    There are several unknowns: does backtracking happen in a step-by-step
    manner until collapse? Or is it with a P(distance-untill-dwell-spot)
    distribution that varies for each position, depending on the length of the
    position and the bubble and RNADNA energies? And from that initial
    distance, there is yet another P(collapse) which can be possion-distributed
    with different parameters, depending partly on the distance and partly on
    the energies of the RNA-DNA and DNA-DNA.

    Another aspect is if there is an intrinsic difference between backtracking
    when before and after full hybrid length is reached -- full hybrid length
    could be stabilizing, making further backtracking and bubble collapse
    less likely -- more than can be accounted for by the free energy alone;
    the full hybrid can act structurally stabilizing in ways unrelated to free
    energy. But most backtracking happens before the 10-mark.. right?

    Give yourself the freedom to optimize for FL, TAbortive, and TRNA, does
    that help?

    AP has adjusted for the fact that most abortive production happens early in
    the ITS.

    Let's say that for a given ITS, for the same P(backtrack), if AP varies
    greatly, this is because of a difference in the collapsability.

    To start with something, can you calculate the following:
    """

if __name__ == '__main__':
    ITSs = main()
