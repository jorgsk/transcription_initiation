"""
New, class based attempt at getting some rigour into creating figures for your
papers. You need more flexibility in terms of subfiguring and not re-running
simulations just to change figure data.

You also need to do some new calculations based on DG3D and DGbubble and DG
hybrid.
"""
from __future__ import division

########### Make a debug function ##########
from ipdb import set_trace as debug

import data_handler
import numpy as np
import matplotlib.pyplot as plt
import optim


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
        grid_size = 8

    targetAttr = {'PY': 'PY',
                  'FL': 'FL',
                  'TA': '_totAbort',
                  'TR': '_totRNA'
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

    return outp


def main():

    # Where Figures are saved

    dg100 = data_handler.ReadData('dg100')  # read the DG100 data
    dg400 = data_handler.ReadData('dg400')  # read the DG100 data

    #ITSs = dg400

    # remove data; remove the controls and sort by name again
    #omgno = ['N25', 'N25/A1anti', 'DG115a', 'DG133']
    #from operator import attrgetter as atrib
    #ITSs = [its for its in ITSs if its.name not in omgno]
    #ITSs = sorted(ITSs, key=atrib('name'))
    # Plot
    #visual_inspection(ITSs, variable='Raw')

    # Plot basic info (FL, PY, total RNA) for the different quantitations
    #data_overview(ITSs)

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
    #quant_stats(ITSs)

    # What did lilian say about the different quantitations?

    # Check if the 3D positions with high AP/Raw data have different DGDNA or
    # DGRNA than the 3D positions with low AP/Raw

    # Select 3D positions from check DG energies -0 -1 and -2 from the site --
    # more minus the longer the x-mer?

    # If nothing works... what is your exit strategy? Show that Keq, and
    # neither DGBUBBle nor DGHYBRID correlate with the abortive probabilities

    # 1) Use the DG100 series to get the weights
    testing = True  # Fast calculations
    # c1: dna, c2: rna, c3: 3'dinuc
    c1, c2, c3 = optimize(dg100, testing, target='PY')

    # 2) Use the DG400 series to correlate Keq with AP and Raw
    # 3) Do those with low AP have DNA-bubble RNA-DNA energy up to that point?

    """
    i) How does the different delta G values correlate with AP (-2, -1, 0, +1,
    +2) around the

    You need to present a story right? So let's start there. The first step

    1) You found that PY correlates with Keq through optimization

    2) You reason that this is because occupation of the pre-translocated state
    increases the chance of backtracking (cite several papers, incl. 2012)

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

    Assume that 
    """

if __name__ == '__main__':
    main()
