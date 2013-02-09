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

    def __init__(self):
        self.plotName = ''
        self.xTicks = x_ticks
        self.yTicks = 0
        self.xTickFontSize = 0
        self.xTickFontSize = 0

        self.nrSubolots = 1

        self.imageSize = [1,4]


def visual_inspection(ITSs):
    """
    1) Bar plot of the abortive probabilities 3 by 3
    """
    plt.ion()

    rowPlotNr = 3
    colPlotNr = 3
    plotPerFig = rowPlotNr*colPlotNr

    # make a map from nucleotide to color
    nuc2clr = {'G': 'b', 'A':'g', 'T':'k', 'C':'y'}

    # nr of figures
    nr_figs = int(np.ceil(len(ITSs)/plotPerFig))

    # partition the ITSs into as many groups as there are figures
    ITSgroups = [ITSs[i*plotPerFig:i*plotPerFig+plotPerFig] for i in range(nr_figs)]

    for fig_nr, ITSgrp in enumerate(ITSgroups):

        fig, axes = plt.subplots(rowPlotNr, colPlotNr)
        rowNr = 0
        colNr = 0

        for oITS in ITSgrp:

            # make a bar plot for the 
            seq = list(oITS.sequence[1:21])
            colors = [nuc2clr[n] for n in seq]
            left_corners = range(1,21)

            # assigxn a plot and advance counter
            ax = axes[rowNr, colNr]
            ax.bar(left_corners, oITS.rawDataMean, yerr=oITS.rawDataStd,
                    color=colors)
            #ax.bar(left_corners, oITS.abortiveProb, yerr=oITS.abortiveProb_std,
                    #color=colors)
            ax.set_xticks([])
            ax.set_title(oITS.name)

            # reset the counter if at end of row
            if rowNr == 2:
                rowNr = 0
                colNr += 1
            else:
                rowNr += 1

    plt.show()


def main():
    # Where Figures are saved
    outpDirs = ['']

    #ITS100 = data_handler.ReadData('dg100') # read the DG100 data
    ITS400 = data_handler.ReadData('dg400') # read the DG100 data

    # Plot
    visual_inspection(ITS400)

    # Do some correlation

    # Check if the 3D positions with high AP/Raw data have different DGDNA or
    # DGRNA than the 3D positions with low AP/Raw

    # Select 3D positions from check DG energies -0 -1 and -2 from the site --
    # more minus the longer the x-mer?
    
    # If nothing works... what is your exit strategy? Show that Keq, and
    # neither DGBUBBle nor DGHYBRID correlate with the abortive probabilities

    #See if you can calculate AP!
    # You should employ another optimizer me thinks, for the balance between
    # RNADNA and DNABUBBLE energies. Use DG3D to separate 
    # and LowKeq
    # 1) Print AP!


if __name__ == '__main__':
    main()
