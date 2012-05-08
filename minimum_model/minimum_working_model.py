# Standard library
#import os
import time
import multiprocessing
import csv
import random
from operator import itemgetter
from itertools import product

# My library
import Energycalc as Ec

# Dependencies : numpy, scipy, Energycalc, dinuc_values.py, bio-python, matplotlib
# sudo apt-get install python-numpy
# sudo apt-get install python-scipy
# sudo apt-get install python-biopython
# sudo apt-get install python-matplotlib (minimum version 1.0)

import numpy as np
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# only for debugging purposes in ipython
from IPython.Debugger import Tracer
debug = Tracer()

class ITS(object):
    """ Storing the ITSs in a class for better handling. """

    def __init__(self, sequence, name='noname', PY=-1, PY_std=-1, apr=-1, msat=-1):
        # Set the initial important data.
        self.name = name
        self.sequence = sequence
        self.PY = PY
        self.PY_std = PY_std
        self.APR = apr
        self.msat = int(msat)
        self.rna_dna1_15 = Ec.RNA_DNAenergy(self.sequence[:15])
        self.rna_dna1_20 = Ec.RNA_DNAenergy(self.sequence)

        # Redlisted sequences 
        self.redlist = []
        self.redlisted = False

        __indiv = list(sequence)
        __dinucs = [__indiv[c] + __indiv[c+1] for c in range(20-1)]

        # Make di-nucleotide vectors for all the energy parameters
        self.rna_dna_di = [Ec.NNRD[di] for di in __dinucs]
        self.dna_dna_di = [Ec.NNDD[di] for di in __dinucs]
        self.keq_di = [Ec.Keq_EC8_EC9[di] for di in __dinucs]
        self.keq_delta_di = [Ec.delta_keq[di] for di in __dinucs]
        self.k1_di = [Ec.k1[di] for di in __dinucs]
        self.kminus1_di = [Ec.kminus1[di] for di in __dinucs]

class Result(object):
    """
    Placeholder for results for plotting
    These are results from the simulation for a given its_length; they are the
    top 20 correlation coefficients, pvals, fitted parameters (c1, c2, c3, c4)
    and the final values

    You get in slightly different things here for Normal and Random results.

    Normal results are the (max) 20 best for each its-length. Random results are
    the X*20 best. For random, mean and std are what's interesting (the mean and
    std of the best for each random sample)
    """
    def __init__(self, corr=np.nan, pvals=np.nan, params=np.nan,
                 finals=np.nan, time_series=False):

        self.corr = corr
        self.pvals = pvals
        self.params = params
        self.finals = finals

        # calculate the mean, std, max, and min values for correlation
        # coefficients, pvalues, and finals simulation results (concentration in
        # its_len)
        self.corr_mean = np.mean(corr)
        self.corr_std = np.std(corr)
        self.corr_max = np.max(corr)
        self.corr_min = np.min(corr)

        self.pvals_mean = np.mean(pvals)
        self.pvals_std = np.std(pvals)
        self.pvals_max = np.max(pvals)
        self.pvals_min = np.min(pvals)

        self.finals_mean = np.mean(finals)
        self.finals_std = np.std(finals)
        self.finals_max = np.max(finals)
        self.finals_min = np.min(finals)

        # The parameters are not so easy to do something with; the should be
        # supplied in a dictionray [c1,c2,c3,c4] = arrays; so make mean, std,
        # min, max for them as well

        # handle nan
        if params is np.nan:
            params = {'c1':[np.nan],
                      'c2':[np.nan],
                      'c3':[np.nan],
                      'c4':[np.nan]}

        self.params_mean = dict((k, np.mean(v)) for k,v in params.items())
        self.params_std = dict((k, np.std(v)) for k,v in params.items())
        self.params_max = dict((k, np.max(v)) for k,v in params.items())
        self.params_min = dict((k, np.min(v)) for k,v in params.items())

        # You need the parameters for the highest correlation coefficient, since
        # that's what you're going to plot (but also plot the mean and std)
        # You rely on that v is sorted in the same order as corr
        self.params_best = dict((k, v[0]) for k,v in params.items())

        if time_series:
            # OK, how are you going to treat this? supposedly they are sorted in
            # the its-fashion. There's no problem sending in a class method to
            # pickle I think... it's if you're multiprocessing from WITHIN a
            # class it's getting mucky.
            self.time_series = time_series

def StringOrFloat(incoming):
    """ Return float if element is float, otherwise return unmodified. """
    datatype = type(incoming)
    if datatype == str:
        try:
            fl00t = float(incoming)
            return fl00t
        except:
            return incoming

def ReadAndFixData(data_path):
    """ Read Hsu csv-file with PY etc data. """

    f = open(data_path, 'rt')
    a = csv.reader(f, delimiter='\t')
    loaded = [[StringOrFloat(v) for v in row] for row in a]
    f.close()

    # Selecting the columns I want from Hsu. Removing all st except PYst.
    # list=[Name 0, Sequence 1, PY 2, PYstd 3, RPY 4, RIFT 5, APR 6, MSAT 7, R 8]
    lizt = [[row[0], row[1], row[2], row[3], row[4], row[6], row[8], row[10],
             row[11]] for row in loaded]

    # Making a list of instances of the ITS class. Each instance is an itr.
    # Storing Name, sequence, and PY.
    ITSs = [ITS(r[1], r[0], r[2], r[3], r[6], r[7]) for r in lizt]

    return ITSs

def ITS_generator(nrseq, length=18, ATstart=True):
    """Generate list of nrseq RNA random sequences. Seq with length. """

    gatc = list('GATC')

    if ATstart:
        beg = 'AT'
        length = length-2
    else:
        beg = ''

    return [beg + ''.join([random.choice(gatc) for dummy1 in range(length)])
            for dummy2 in range(nrseq)]

def model_wrapper(ITSs, testing, p_line):
    """
    arguments
    testing: solve the model with a reduced grid
    p-line: make a 0.05 probability threshold line
    """

    # grid size
    grid_size = 15
    if testing:
        grid_size = 5

    # random and cross-validation
    control = 15
    if testing:
        control = 2

    # Compare with the PY percentages in this notation
    PYs = np.array([itr.PY for itr in ITSs])*0.01

    # Set the parameter ranges you want to test out
    c1 = np.array([20]) # insensitive to variation here
    c2 = np.linspace(0.001, 0.2, grid_size)
    c3 = np.linspace(0.001, 0.2, grid_size)
    c4 = np.linspace(0.1, 0.4, grid_size)

    par_ranges = (c1, c2, c3, c4)

    # Time-grid
    t = np.linspace(0, 1., 100)

    # XXX you should get it to work with 21
    its_range = range(3, 20)

    optim = False # GRID

    randomize = control # here 0 = False (or randomize 0 times)

    initial_bubble = True

    # Fit with 50% of ITS and apply the parameters to the remaining 50%
    retrofit = control

    all_results = scrunch_runner(PYs, its_range, ITSs, par_ranges, optim,
                                 randomize, retrofit, t, initial_bubble)

    # extract the specific results
    results, rand_results, retrof_results = all_results

    # ladder plot
    fig_lad, ax_lad = print_scrunch_ladder(results, rand_results,
                                           retrof_results, optim, randomize,
                                           par_ranges, initial_bubble, p_line)

    fig_sct = print_scrunch_scatter(results, rand_results, optim, randomize, par_ranges,
                  initial_bubble, PYs)

    return fig_lad, fig_sct

def print_scrunch_ladder(results, rand_results, retrof_results, optimize,
                         randomize, par_ranges, initial_bubble, p_line,
                         description=False, print_params=True, in_axes=False,
                         ax_nr=0):
    """
    Alternative print-scrunch.
    [0][0] is the pearson for the real and the control
    [0][1] is the parameters for the pearson fit
    """
    if print_params:
        fig, axes = plt.subplots(1,2)
    else:
        fig, ax = plt.subplots()
        axes = np.array([ax])

    if in_axes:
        axes = in_axes

    colors_params = ['r', 'c', 'm', 'k']

    all_params = ('c1', 'c2', 'c3', 'c4')
    # only plot parameters that are variable 
    plot_params = [p for p, r in zip(all_params, par_ranges) if len(r) > 1]

    # assign colors to parameters
    par2col = dict(zip(all_params, colors_params))

    # go over both the real and random results
    # This loop covers [0][0] and [0][1]
    for (ddict, name, colr) in [(results, 'real', 'b'),
                                (rand_results, 'random', 'g'),
                               (retrof_results, 'cross-validated', 'k')]:

        # don't process 'random' or 'retrofit' if not evaluated
        if False in ddict.values():
            continue

        # get its_index and corr-coeff from sorted dict
        if name == 'real':
            indx, corr, pvals = zip(*[(r[0], r[1].corr_max, r[1].pvals_max)
                               for r in sorted(ddict.items())])

        elif name == 'random':
            indx, corr = zip(*[(r[0], r[1].corr_mean)
                               for r in sorted(ddict.items())])

            stds = [r[1].corr_std for r in sorted(ddict.items())]

        elif name == 'cross-validated':
            indx, corr = zip(*[(r[0], r[1].corr_mean)
                               for r in sorted(ddict.items())])

            stds = [r[1].corr_std for r in sorted(ddict.items())]

        # make x-axis
        incrX = range(indx[0], indx[-1]+1)

        # if random, plot with errorbars
        if name == 'real':
            axes[ax_nr].plot(incrX, corr, label=name, linewidth=2, color=colr)

            # interpolate pvalues (x, must increase) with correlation (y) and obtain the
            # correlation for p = 0.05 to plot as a black
            if p_line:
                pv = reversed(list(pvals))
                co = reversed(list(corr))
                f = scipy.interpolate.interp1d(list(pv), list(co))
                axes[ax_nr].axhline(y=f(0.05), ls='--', color='r', label='p = 0.05 threshold',
                                    linewidth=2)

        elif name == 'random':
            axes[ax_nr].errorbar(incrX, corr, yerr=stds, label=name, linewidth=2,
                             color=colr)

        elif name == 'cross-validated':
            axes[ax_nr].errorbar(incrX, corr, yerr=stds, label=name, linewidth=2,
                             color=colr)


        # skip if you're not printing parametes
        if not print_params:
            continue

        # get its_index parameter values (they are index-sorted)
        if name == 'real':
            paramz_best = [r[1].params_best for r in sorted(ddict.items())]
            paramz_mean = [r[1].params_mean for r in sorted(ddict.items())]
            paramz_std = [r[1].params_std for r in sorted(ddict.items())]

            # each parameter should be plotted with its best (solid) and mean
            # (striped) values (mean should have std)
            for parameter in plot_params:

                # print the best parameters
                best_par_vals = [d[parameter] for d in paramz_best]
                axes[ax_nr+1].plot(incrX, best_par_vals, label=parameter,
                                   linewidth=2, color=par2col[parameter])
                # mean
                mean_par_vals = [d[parameter] for d in paramz_mean]

                # std
                # print the mean and std of the top 20 parameters
                std_par_vals = [d[parameter] for d in paramz_std]
                axes[ax_nr+1].errorbar(incrX, mean_par_vals, yerr=std_par_vals,
                                       color=par2col[parameter], linestyle='--')


    xticklabels = [str(integer) for integer in range(3,21)]
    yticklabels = [str(integer) for integer in np.arange(-1, 1.1, 0.1)]
    #yticklabels = [str(integer) for integer in np.arange(0, 1, 0.1)]
    #yticklabels_1 = [str(integer) for integer in np.arange(-0.05, 0.5, 0.05)]

    # make the almost-zero into a zero if going from -1 to 1
    yticklabels[10] = '0'

    for ax in axes.flatten():
        # legend
        ax.legend(loc='lower left')

        # xticks
        ax.set_xticks(range(3,21))
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(3,21)
        ax.set_xlabel("Nucleotide from transcription start", size=20)

        # awkward way of setting the tick font sizes
        for l in ax.get_xticklabels():
            l.set_fontsize(12)
        for l in ax.get_yticklabels():
            l.set_fontsize(12)

    axes[0].set_ylabel("Correlation coefficient, $r$", size=20)

    axes[0].set_yticks(np.arange(-1, 1.1, 0.1))
    #axes[0].set_yticks(np.arange(0, 1, 0.1))
    axes[0].set_yticklabels(yticklabels)
    # you need a grid to see your awsome 0.8 + correlation coefficient
    axes[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

    if print_params:
        axes[1].set_ylabel("Model parameter values", size=20)
        #axes[1].set_yticks(np.arange(-0.05, 0.5, 0.05))
        #axes[1].set_yticklabels(yticklabels_1)
        #axes[1].set_ylim(-0.05, 0.5)

    fig.set_figwidth(15)
    fig.set_figheight(10)

    if optimize:
        approach = 'Least squares optimizer'
    else:
        approach = 'Grid'

    for_join = []

    for let, par in zip(('a', 'b', 'c', 'd'), par_ranges):
        if len(par) == 1:
            for_join.append(let + ':{0}'.format(par[0]))
        else:
            mi, ma = (format(min(par), '.3f'), format(max(par), '.3f'))
            for_join.append(let + ':({0}--{1})'.format(mi, ma))

    if description:
        descr = ', '.join(for_join) + '\n' + description
    else:
        descr = ', '.join(for_join)

    hedr = 'Approach: {0}. Nr random samples: {1}\n\n{2}\n'.format(approach,
                                                               randomize, descr)
    fig.suptitle(hedr)

    return fig, axes
def print_scrunch_scatter(results, rand_results, optim, randomize, par_ranges,
                          initial_bubble, PYs):
    """
    Print scatter plots at 5, 10, 15, and 20 if available. If not, leave blank.

    RESULT this has proven to be a problem. While the correlation itself is good
    for any c1, only a specific c1 value will give similar value ranges for PYs
    and finals for a given its_len. This is to be expected, and is a trade-off
    between c1 and t -- the time of integration.

    Isn't it then marvelous that you got such a good match between PY and the
    'abortive propensity' initially? A little bit too marvelous for my flavor.
    """
    #plt.ion()

    PYs = PYs*100

    #rows = [5, 10, 15, 20]
    rows = [15]

    if len(rows) > 1:
        fig, axes = plt.subplots(len(rows), 1, sharey=True)
    else:
        fig, ax = plt.subplots()
        axes = [ax]

    # for each row, pick the results 
    fmts = ['k', 'g', 'b', 'c']

    for row_nr, maxnuc in enumerate(rows):
        name = '1_{0}_scatter_comparison'.format(maxnuc)

        if maxnuc-1 in results:
            finals = results[maxnuc-1].finals # you were comparing to PY/100
        else:
            continue

        ax = axes[row_nr]

        ax.scatter(finals, PYs, color= fmts[row_nr])

        corrs = pearsonr(finals, PYs)

        ax.set_ylabel("PY ({0} nt of ITS)".format(maxnuc), size=15)

        if row_nr == 0:
            #ax.set_xlabel("Abortive propensity", size=20)
            #header = '{0}\nr = {1:.2f}, p = {2:.1e}'.format(name, corrs[0], corrs[1])
            header = 'Pearson: r = {1:.2f}, p = {2:.1e}'.format(name, corrs[0], corrs[1])
            ax.set_title(header, size=15)
        else:
            header = 'r = {0:.2f}, p = {1:.1e}'.format(corrs[0], corrs[1])
            ax.set_title(header, size=15)

        # awkward way of setting the tick sizes
        for l in ax.get_xticklabels():
            l.set_fontsize(12)
            #l.set_fontsize(6)
        for l in ax.get_yticklabels():
            l.set_fontsize(12)
            #l.set_fontsize(6)

        fmin, fmax = min(finals), max(finals)

        scale = (fmax-fmin)*0.2

        ax.set_xlim(fmin-scale, fmax+scale)

    return fig

def scrunch_runner(PYs, its_range, ITSs, ranges, optimize, randize, retrofit, t,
                   init_bubble):
    """
    Wrapper around grid_scrunch and opt_scrunch.
    """

    # make 'int' into a 'list' containing just that int
    if type(its_range) is int:
        its_range = range(its_range, its_range +1)

    results = {}
    results_random = {}
    results_retrof = {}

    for its_len in its_range:

        # set initial states
        state_nr = its_len - 1
        # The second nt is then state 0
        # initial vales for the states
        y0 = [1] + [0 for i in range(state_nr-1)]

        # get the 'normal' results
        normal_obj = grid_scruncher(PYs, its_len, ITSs, ranges, t, y0,
                                    state_nr, initial_bubble=init_bubble)

        results[its_len] = normal_obj

        # Randomize
        if randize:
            # get the results for randomized ITS versions
            random_obj = grid_scruncher(PYs, its_len, ITSs, ranges, t, y0,
                                        state_nr, randomize=randize,
                                        initial_bubble=init_bubble)
        else:
            random_obj = False

        # Retrofit
        if retrofit:
            retrof_obj = grid_scruncher(PYs, its_len, ITSs, ranges, t, y0,
                                        state_nr, retrof=retrofit,
                                        initial_bubble=init_bubble)
        else:
            retrof_obj = False

        results_retrof[its_len] = retrof_obj
        results_random[its_len] = random_obj


    return results, results_random, results_retrof

def grid_scruncher(PYs, its_len, ITSs, ranges, t, y0, state_nr, randomize=0,
                   initial_bubble=True, retrof=0):
    """
    Separate scruch calls into randomize, retrofit, or neither.
    """

    # RETROFIT
    if retrof:
        retro_results = []
        for repeat in range(retrof):

            # Choose 50% of the ITS randomly; both energies and PYs
            retrovar = get_its_variables(ITSs, retrofit=True, PY=PYs)
            #extract the stuff
            ITS_fitting, ITS_compare, PYs_fitting, PYs_compare = retrovar

            arguments_fit = (y0, t, its_len, state_nr, ITS_fitting, PYs_fitting,
                           initial_bubble)

            fit_result = grid_scrunch(arguments_fit, ranges)

            # Skip if you don't get a result (should B OK with large ranges)
            if not fit_result:
                continue

            # update the ranges to the optimal for the fitting
            par_order = ('c1', 'c2', 'c3', 'c4')
            fit_ranges = [np.array([fit_result.params_best[p]])
                          for p in par_order]

            # rerun with new arguments
            arguments_compare = (y0, t, its_len, state_nr, ITS_compare, PYs_compare,
                         initial_bubble)

            # run the rest of the ITS with the optimal parameters
            control_result = grid_scrunch(arguments_compare, fit_ranges)

            # Skip if you don't get a result (should B OK with large ranges)
            if not control_result:
                continue

            else:
                retro_results.append(control_result)

        return average_rand_result(retro_results)

    # RANDOMIZE
    elif randomize:
        rand_results = []
        for repeat in range(randomize):

            # normalizing by testing random sequences. Can we fit the parameters
            # from random sequences to these ITS values?

            ITS_variables = get_its_variables(ITSs, randomize=True)

            arguments = (y0, t, its_len, state_nr, ITS_variables, PYs,
                         initial_bubble)

            rand_result = grid_scrunch(arguments, ranges)

            # skip if you don't get anything from this randomizer
            if not rand_result:
                continue

            else:
                rand_results.append(rand_result)


        return average_rand_result(rand_results)

    # NO RETROFIT AND NO RANDOMIZE
    else:
        #go from ITS object to dict to appease the impotent pickle
        ITS_variables = get_its_variables(ITSs)

        arguments = (y0, t, its_len, state_nr, ITS_variables, PYs,
                     initial_bubble)

        return grid_scrunch(arguments, ranges)

def ITSgenerator():
    """Generate a single random ITS """
    gatc = list('GATC')
    return 'AT'+ ''.join([random.choice(gatc) for dummy1 in range(18)])

def average_rand_result(rand_results):
    """make new corr, pvals, and params; just add them together; ignore the
    #finals

    # Here you must think anew. What do you want to show for the randomized
    # values in the final plot? There are, say, 10 random versions. At each
    # randomization, I pick the one with max correlation. I disregard the 20
    # best; just keep the max.
    #

    # For the parameters I want to show the average and std of those parameters
    # that were used to obtain the 'best' correlation coefficients.

    # What do I get in? Is it the same for the grid and the optimized? No, it's
    # not. For the Grid you get the 20 best.
    """

    new_corr, new_pvals, new_params = [], [], {'c1':[], 'c2':[],
                                               'c3':[], 'c4':[]}

    # Add the values from the X random versions
    # You don't want all 20, you just want the best ones

    # check if 

    for res_obj in rand_results:
        # check if this is a NaN object; skip it
        if res_obj.corr is np.nan:
            continue

        new_corr.append(res_obj.corr[0])
        new_pvals.append(res_obj.pvals[0])

        for k, v in res_obj.params.items():
            new_params[k].append(v[0])

    # Check if new_corr is empty (means all results were NaN)
    if new_corr == []:
        return Result()
    else:
        # make a new result object
        return Result(new_corr, new_pvals, new_params, res_obj.finals)

def get_its_variables(ITSs, randomize=False, retrofit=False, PY=False):
    """
    Get ITS objects and return a dict of their energy variables: RNA-DNA,
    DNA-DNA, Keq
    """
    new_ITSs = []

    for its in ITSs:

        # if randomize, create a new object ITS object from a random sequence
        if randomize:
            rand_seq = ITSgenerator()
            its = ITS(rand_seq)

        ITSs_dict = {'rna_dna_di': np.array(its.rna_dna_di),
                     'dna_dna_di': np.array(its.dna_dna_di),
                     'keq_di': np.array(its.keq_delta_di)}

        new_ITSs.append(ITSs_dict)

    # if retrofit, divide the list in 2 with random indices. ALso split the PYs
    # in the same way cause you need the its-> PY relationship
    if retrofit:
        its_nr = len(ITSs)
        pickme = np.random.randint(its_nr, size=np.ceil(its_nr/2.0))

        ITS_fitting = [new_ITSs[i] for i in pickme]
        ITS_compare = [new_ITSs[i] for i in range(43) if i not in pickme]

        PYs_fitting = [PY[i] for i in pickme]
        PYs_compare = [PY[i] for i in range(43) if i not in pickme]

        return ITS_fitting, ITS_compare, PYs_fitting, PYs_compare

    else:

        return new_ITSs

def grid_scrunch(arguments, ranges):
    """
    Wrapper around the multi-core solver. Return a Result object with
    correlations and pvalues.
    """

    # all possible compbinations
    params = [p for p in product(*ranges)]

    #divide the params into 4
    window = int(np.floor(len(params)/4.0))
    divide = [params[i*window:(i+1)*window] for i in range(3)]
    last = params[3*window:] # make sure you get all in the last one
    divide.append(last)

    t1 = time.time()

    # make a pool of workers for multicore action
    my_pool = multiprocessing.Pool(4)
    #my_pool = multiprocessing.Pool(2)
    results = [my_pool.apply_async(_multi_func, (p, arguments)) for p in divide]
    my_pool.close()
    my_pool.join()
    # flatten the output
    all_results = sum([r.get() for r in results], [])

    # All_results is a list of tuples on the following form:
    # ((finals, time_series, par, rp, pp))
    # Sort them on the pearson correlation pvalue, 'pp'

     #the non-parallell version for debugging purposes
    #all_results = sum([_multi_func(*(pa, arguments)) for pa in divide], [])

    # Filter and pick the 20 with smalles pval
    all_sorted = sorted(all_results, key=itemgetter(4))

    # pick top 20
    top_hits = all_sorted[:20]

    # if top_hits is empty, return NaNs (created by default)
    if top_hits == []:
        return Result() # all values are NaN

    # now make separate corr, pvals, params, and finals arrays from these
    finals = np.array(top_hits[0][0]) # only get finals for the top top
    time_series = top_hits[0][1] # only get time series for the top top
    pars = [c[2] for c in top_hits]
    corr = np.array([c[3] for c in top_hits])
    pvals = np.array([c[4] for c in top_hits])

    # params must be made into a dict
    # params[c1,c2,c3,c4] = [array]
    params = {}
    for par_nr in range(1, len(ranges)+1):
        params['c{0}'.format(par_nr)] = [p[par_nr-1] for p in pars]

    # make a Result object
    result_obj = Result(corr, pvals, params, finals, time_series)

    print time.time() - t1

    return result_obj

def _multi_func(paras, arguments):
    """
    Evaluate the model for all parameter combinations. Return the final values,
    the parameters, and the correlation coefficients.
    """
    all_hits = []
    PYs = arguments[-2]

    for par in paras:
        finals, time_series = cost_function_scruncher(par, *arguments, run_once=True)

        # XXX this one is easy to forget ... you don't want answers where the
        # final values are too small
        if sum(finals) < 0.00001:
            continue

        # correlation
        rp, pp = pearsonr(PYs, finals)

        all_hits.append((finals, time_series, par, rp, pp))

    return all_hits

def cost_function_scruncher(start_values, y0, t, its_len, state_nr, ITSs, PYs,
                            initial_bubble, run_once=False, const_par=False,
                            truth_table=False):
    """
    k1 = c1*exp(-(c2*rna_dna_i - c3*dna_dna_{i+1} + c4*Keq_{i-1}) * 1/RT)

    Dna-bubble one step ahead of the rna-dna hybrid; Keq one step behind.
    When the rna-dna reaches length 9, the you must subtract the hybrid for
    each step forward.

    """
    # get the tuning parameters and the rna-dna and k1, kminus1 values
    # RT is the gas constant * temperature
    RT = 1.9858775*(37 + 273.15)/1000   # divide by 1000 to get kcalories
    finals = []
    time_series = []

    # The energy of the initiation bubble
    # energy at 0 improves the correlation after +13 or so
    if initial_bubble:
        minus11_en = -9.95 # 'ATAATAGATTCAT'
    else:
        minus11_en = 0 # 'ATAATAGATTCAT' after nt 15, it seems that having initial

    for its_dict in ITSs:

        # must shift by minus 1 ('GATTA' example: GA, AT, TT, TA -> len 5, but 4
        # relevant dinucleotides)

        # to dictionary with the rna dna keq parameters in them
        dna_dna = its_dict['dna_dna_di'][:its_len-1]
        rna_dna = its_dict['rna_dna_di'][:its_len-1]
        keq = its_dict['keq_di'][:its_len-1]

        # For run_once and if all 4 parameters are used
        if len(start_values) == 4:
            (a, b, c, d) = start_values

        # If optimization and not all 4 are used, a, b, c, d will be a mix
        # between variable and fixed parameters
        elif len(start_values) < 4:
            (a, b, c, d) = get_mixed_params(truth_table, start_values, const_par)

        k1, proceed = calculate_k1(minus11_en, RT, its_len, keq, dna_dna,
                                   rna_dna, a, b, c, d)

        # you must abort if proceed is false and run_once is true
        if run_once and not proceed:
            finals.append(0)
        else:

            A = equlib_matrix(k1, state_nr)
            # pass the jacobian matrix to ease calculation
            soln, info = scipy.integrate.odeint(rnap_solver, y0, t, args = (A,),
                                                full_output=True, Dfun=jacob_second)
            finals.append(soln[-1][-1])

            steps = len(soln)

            time_series.append([soln[steps*0.1], soln[int(steps/2)], soln[-1]])

    if run_once:

        return np.array(finals), time_series

    # if not run once, you're called from an optimizer -> return the distance
    # vector; also include the distance from var(PY) with var(finals)
    else:

        # Retry optimizer with PYs as objective
        objective = np.array(PYs)
        result = np.array(finals)

        return objective - result

def jacob_second(t, y, A):
    """
    Return the jacobian matrix

    X = [X0, X1, X2, X3]
    A = [[-k0,   0 , 0,  0],
         [k0 , -k1 , 0,  0],
         [0  ,  k1, -k2, 0],
         [0  ,  0  , k2, 0]]

    AX = [[-k0*X0         ],
          [k0*X0 - k1*X1  ],
          [k1*X1 - k2*X2  ],
          [k2*X2         ]]


    J = [[df1/dX0, df1/dX1, df1/dX2, df1/dX3],
         [df2/dX0, ...,                     ]
            .
            .
         [dfN/dx0, ...,              dfN/dX3]

    J = [[-k0,   0, 0, 0],
         [k0 , -k1, 0, 0],
         [0  ,  k1]
         ...

         Waitaminute, here J = A! Linear function ...


    """

    return A

def rnap_solver(y, t, A):
    """
    Solver.
    """

    dx = np.dot(A, y)

    return dx

def equlib_matrix(rate, state_nr):
    """
    Assume equlibrium for the reaction:
    X_f_0 <=> X_e_0 -> X_f_1

    Then you get d(Xf_1)/dt = k1[Xf_0]

    The value of k1 is calculate outside

    Example:

    seq = 'GATTA'

    X0 = A,
    X1 = T,
    X2 = T,
    X3 = A

    X0 -> X1 , k0
    X1 -> X2 , k1
    X2 -> X3 , k2

    X = [X0, X1, X2, X3]
    A = [[-k0,   0 , 0,  0],
         [k0 , -k1 , 0,  0],
         [0  ,  k1, -k2, 0],
         [0  ,  0  , k2, 0]]
    """

    # initialize with the first row
    rows = [  [-rate[0]] + [0 for i in range(state_nr-1)] ]

    for r_nr in range(state_nr-2):
        row = []

        for col_nr in range(state_nr):

            if col_nr == r_nr:
                row.append(rate[r_nr])

            elif col_nr == r_nr+1:
                row.append(-rate[r_nr+1])

            else:
                row.append(0)

        rows.append(row)

    rows.append([0 for i in range(state_nr-2)] + [ rate[-1], 0 ])

    return np.array(rows)

def calculate_k1(minus11_en, RT, its_len, keq, dna_dna, rna_dna, a, b, c, d):
    """
    Make the reaction rate array
    """

    # initialize empty rates
    k1 = np.zeros(its_len-2)

    proceed = True  # don't proceed for bad exponenitals and run_once = True

    for i in range(1, its_len-1):
        KEQ = keq[i-1]
        DNA_DNA = minus11_en + sum(dna_dna[:i])
        if i < 9:
            RNA_DNA = sum(rna_dna[:i])
        else:
            RNA_DNA = sum(rna_dna[i-9:i])

        expo = (-b*RNA_DNA +c*DNA_DNA +d*KEQ)/RT

        # if expo is above 0, the exponential will be greater than 1, and
        # XXX why? have you given this a lot of thought?
        # this will in general not work
        #if run_once and expo > 0:
            #proceed = False
            #break

        # there should be way of introducing this constraint to the
        # optimizer. Yes, but you have to change optimizers and it's not
        # straightforward.

        rate = a*np.exp(expo)

        k1[i-2] = rate

    return k1, proceed

def get_mixed_params(truth_table, start_values, const_par):
    """
    You pass a variable numer of fixed and constant parameters for optimization.
    You join these into one parameter set.
    """

    s_values = start_values.tolist()

    # make a copy of const_par -- otherwise it seems to disappear from the outer
    # loop. I don't understand how that works. Try to recreate it.
    c_par = list(const_par)

    # reverse parameters and constant_parameters to be able to pop
    # front-first
    s_values.reverse()
    c_par.reverse()

    mix_param = []

    for ind, boo in enumerate(truth_table):
        if boo:
            mix_param.append(s_values.pop())
        else:
            mix_param.append(c_par.pop())

    return mix_param

def main():
    data_dir = 'sequence_data/Hsu/csvHsu'
    ITSs = ReadAndFixData(data_dir) # read raw data

    #p_line = True
    p_line = False # might fail if testing is True
    testing = True

    fig_ladder, fig_scatter = model_wrapper(ITSs, testing, p_line)

    for fig, name in [(fig_ladder, 'ladder'), (fig_scatter, 'scatter')]:

        fig.savefig(name+'.pdf', transparent=True, format='pdf')


# If launched externally, run main()
if __name__ == '__main__':
    main()

