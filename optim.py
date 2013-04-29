"""
Module for optimizing correlation between ITSs and PY

You are breaking compatibility with the code in DocumentationTranscription
because you're no longer considering the differential equation model.
"""

from data_handler import ITS
import itertools
import numpy as np
import multiprocessing
from scipy.stats import spearmanr
from operator import itemgetter
from ipdb import set_trace as debug  # NOQA
import time


class Result(object):
    """
    Placeholder for results for plotting
    These are results from the simulation for a given its_length; they are the
    top 20 correlation coefficients, pvals, fitted parameters (c1, c2, c3)
    and the final values

    You get in slightly different things here for Normal and Random results.

    Normal results are the (max) 20 best for each its-length. Random results are
    the X*20 best. For random, mean and std are what's interesting (the mean and
    std of the best for each random sample)
    """
    def __init__(self, corr=[np.nan], pvals=np.nan, params=np.nan,
                 SEn=np.nan):

        self.corr = corr
        self.pvals = pvals
        self.params = params
        self.SEn = SEn

        # calculate the mean, std, max, and min values for correlation
        # coefficients, pvalues, and SEn
        self.corr_mean = np.mean(corr)
        self.corr_std = np.std(corr)
        max_indx = np.abs(corr).argmax()  # correlation may be negative

        if len(corr) == 1 and np.isnan(corr[0]):
            self.corr_max = np.nan
        else:
            self.corr_max = corr[max_indx]

        self.pvals_mean = np.mean(pvals)
        self.pvals_std = np.std(pvals)
        self.pvals_min = np.min(pvals)

        self.SEn_mean = np.mean(SEn)
        self.SEn_std = np.std(SEn)
        self.SEn_max = np.max(SEn)
        self.SEn_min = np.min(SEn)

        # The parameters are not so easy to do something with; the should be
        # supplied in a dictionray [c1,c2,c3] = arrays; so make mean, std,
        # min, max for them as well

        # handle nan
        if params is np.nan:
            params = {'c1':[np.nan],
                      'c2':[np.nan],
                      'c3':[np.nan]}

        # extract mean, std etc
        self.params_mean = dict((k, np.mean(v)) for k,v in params.items())
        self.params_std = dict((k, np.std(v)) for k,v in params.items())
        self.params_max = dict((k, np.max(v)) for k,v in params.items())
        self.params_min = dict((k, np.min(v)) for k,v in params.items())

        # You need the parameters for the highest correlation coefficient, since
        # that's what you're going to plot (but also plot the mean and std)
        # You rely on that v is sorted in the same order as corr
        self.params_best = dict((k, v[0]) for k,v in params.items())


def main_optim(its_range, ITSs, ranges, randize, crosscorr, normal):
    """

    Call the core optimialization wrapper but treat different analysis cases
    differently.

    """

    # make 'int' into a 'list' containing just that int
    if type(its_range) is int:
        its_range = range(its_range, its_range+1)

    results = {}
    results_random = {}
    results_crosscorr = {}

    for its_len in its_range:

        # 'normal' results (full dataset)
        if normal:
            normal_obj = core_optim_wrapper(its_len, ITSs, ranges)
        else:
            normal_obj = False

        # Randomize
        if randize:
            random_obj = randomize_wrapper(its_len, ITSs, ranges, randize)
        else:
            random_obj = False

        # Cross correlation
        if crosscorr:
            crosscorr_obj = crosscorr_wrapper(its_len, ITSs, ranges, crosscorr)
        else:
            crosscorr_obj = False

        results[its_len] = normal_obj
        results_crosscorr[its_len] = crosscorr_obj
        results_random[its_len] = random_obj

    return results, results_random, results_crosscorr


def ITS_RandomSplit(ITSs):
    """
    Divide the list in 2 with random indices.
    """

    its_nr = len(ITSs)
    fit_indices = set([])

    # populate an array with random indices
    while len(fit_indices) < its_nr/2.0:
        fit_indices.add(np.random.randint(np.ceil(its_nr)))

    ITS_fit = [ITSs[i] for i in fit_indices]
    ITS_compare = [ITSs[i] for i in range(its_nr) if i not in fit_indices]

    return ITS_fit, ITS_compare


def randomize_ITS_sequence(ITSs):
    """
    Copy the ITS variables and then randomize their sequence.
    """

    copy_ITSs = []

    # keep only the PY and names of the ITS objects -> new sequence and energies
    for its in ITSs:
        random_its = ITSgenerator()
        copy_ITSs.append(ITS(random_its, name=its.name, PY=its.PY))

    return copy_ITSs


def ITSgenerator():
    """Generate a random ITS """
    import random
    gatc = list('GATC')
    return 'AT'+ ''.join([random.choice(gatc) for dummy1 in range(18)])


def randomize_wrapper(its_len, ITSs, ranges, randomize=0):

    rand_results = []
    for _ in range(randomize):

        ITS_random = randomize_ITS_sequence(ITSs)

        rand_result = core_optim_wrapper(its_len, ITS_random, ranges)

        # skip if you don't get anything from this randomizer
        if not rand_result:
            continue

        else:
            rand_results.append(rand_result)

    return average_rand_result(rand_results)


def crosscorr_wrapper(its_len, ITSs, ranges, crosscorr=0):
    """
    Wrapper around core_optim_wrapper that deals with cross-correlation.
    Optimialization is performed on one half of the ITSs and the correlation is
    obtained from the second half using the parameters obtained from the first
    half.
    """

    retro_results = []
    for _ in range(crosscorr):

        # Choose 50% of the ITS randomly; both energies and PYs
        ITS_fit, ITS_compare = ITS_RandomSplit(ITSs)

        fit_result = core_optim_wrapper(its_len, ITS_fit, ranges)

        # Skip if you don't get a result (should B OK with large ranges)
        if not fit_result:
            continue

        # Get the optimal parameter values for the fitted ITSs
        par_order = ('c1', 'c2', 'c3')
        fit_ranges = [np.array([fit_result.params_best[p]])
                      for p in par_order]

        # run the rest of the ITS with the optimal parameters from fitting-set
        control_result = core_optim_wrapper(its_len, ITS_compare, fit_ranges)

        # Skip if you don't get a result (should B OK with large ranges)
        if not control_result:
            continue
        else:
            retro_results.append(control_result)

    # average the results (keeping mean and std) and return
    return average_rand_result(retro_results)


def core_optim_wrapper(its_len, ITSs, ranges):
    """
    Wrapper around the multi-core solver. Return a Result object with
    correlations and pvalues.
    """

    t1 = time.time()

    # all possible combinations of c_i values
    params = [p for p in itertools.product(*ranges)]

    # divide the parameter space into subsets for processing
    window = int(np.floor(len(params)/4.0))
    divide = [params[i*window:(i+1)*window] for i in range(3)]
    last = params[3*window:]  # make sure you get all in the last one
    divide.append(last)

    #make a pool of workers for multicore action
    # this is only efficient if you have a range longer than 10 or so
    rmax = sum([len(r) for r in ranges])
    #rmax = 5  # uncomment for multiprocessing debugging.
    if rmax > 6:
        my_pool = multiprocessing.Pool()
        results = [my_pool.apply_async(_multi_calc, (p, its_len, ITSs))
                      for p in divide]
        my_pool.close()
        my_pool.join()
        # flatten the output
        all_results = sum([r.get() for r in results], [])
    else:  # the non-parallell version for debugging and no multi-range calculations
        all_results = sum([_multi_calc(*(p, its_len, ITSs))
                            for p in divide], [])

    # All_results is a list of tuples on the following form: ((SEn, par, rp, pp))
    # Sort them on the pearson/spearman correlation pvalue, 'pp' and pick top 20
    top_hits = sorted(all_results, key=itemgetter(3))[:20]

    # if top_hits is empty, return NaNs results (created by default)
    if top_hits == []:
        return Result()

    # now make separate corr, pvals, params, and SEn arrays from these
    SEn = np.array(top_hits[0][0])  # only get SEn for the top top
    pars = [c[1] for c in top_hits]
    corr = np.array([c[2] for c in top_hits])
    pvals = np.array([c[3] for c in top_hits])

    # Return parameters in a dictionary form
    # pars[c1, c2, c3] = [array]
    params = {}
    for par_nr in range(1, len(ranges)+1):
        params['c{0}'.format(par_nr)] = [p[par_nr-1] for p in pars]

    # make a Result object
    result_obj = Result(corr, pvals, params, SEn)

    print 'time: ', time.time() - t1

    return result_obj


def _multi_calc(paras, its_len, ITSs):
    """
    Evaluate the model for all parameter combinations. Return the final values,
    the parameters, and the correlation coefficients.
    """

    all_hits = []
    y = [its.PY for its in ITSs]

    for par in paras:
        SEn = keq_calc(par, its_len, ITSs)

        # correlation
        rp, pp = spearmanr(y, SEn)

        # ignore parameter correlation where the correlation is a nan
        if np.isnan(rp):
            continue

        all_hits.append((SEn, par, rp, pp))

    return all_hits


def keq_calc(start_values, its_len, ITSs):

    """
    k1 = exp(-(c1*rna_dna_i + c2*dna_dna_{i+1} + c3*Keq_{i-1}) * 1/RT)

    Dna-bubble one step ahead of the rna-dna hybrid; Keq one step behind.
    When the rna-dna reaches length 9, the you must subtract the hybrid for
    each step forward. The hybrid oscillates between a 8bp and a
    9bp hybrid during translocation.
    """
    # get the tuning parameters and the rna-dna and k1, kminus1 values
    # RT is the gas constant * temperature
    RT = 1.9858775*(37 + 273.15)/1000  # divide by 1000 to get kcalories
    ITS_SEn = []

    for its in ITSs:

        # must shift by minus 1 ('GATTA' example: GA, AT, TT, TA -> len 5, but 4
        # relevant dinucleotides)

        # first -1 because of python indexing. additional -1 because only (1,2)
        # information is needed for 3. ATG -> [(1,2), (2,3)]. Only DNA-DNA needs
        # both.

        dna_dna = its.dna_dna_di[:its_len-1]
        rna_dna = its.rna_dna_di[:its_len-2]
        dg3d = its.keq_delta_di_b[:its_len-2]

        # the c1, c2, c3 values
        (a, b, c) = start_values

        # equilibrium constants at each position
        keqs = keq_i(RT, its_len, dg3d, dna_dna, rna_dna, a, b, c)

        SEn = sum(keqs)

        ITS_SEn.append(SEn)

    return np.array(ITS_SEn)


def keq_i(RT, its_len, keq, dna_dna, rna_dna, a, b, c):
    """
    NOTE!! now calculating for the backward translocation

    Recall that the energies are in dinucleotide form. Thus ATG -> [0.4, 0.2]

    Tip for the future; start at +1 also for nucleotides, so all arrays have the
    same length. Just have the first values as 0. Makes it much easier to
    understand the code later. The 0s are placeholders.

    !!There is no RNA-DNA energy change before +10. FFS.

    Q: what do you get in? the keq, dna_dna etc?

    k1[0] should be for d(x_3)/dt

    -> keq = (1,2)
    -> RD = 0
    -> DD = (1,3) - (1,2)

    k1[1] should be for d(x_4)/dt

    -> keq = (2,3)
    -> RD = 0
    -> DD = (1,4) - (1,3)

    ...

    k1[8] is the last where RD is zero

    k1[7] is then d(x_10)/dt -- the first where RD gets subtracted

    -> keq = (8,9)
    -> RD = (2,9) - (1,9)
    -> DD = (1,10) - (1,9)

    k1[8] is then d(x_11)/dt

    -> keq = (9,10)
    -> RD = (3,10) - (2,10)
    -> DD = (1,11) - (1,10)
    ...

    until k1[17] which is d(x_20)/dt -- or do you want to include x_21?

    """

    # initialize empty rates. -2 because you start going from +2 to +3.
    # ATGCT -> [0.3, 0.5, 0.2, 0.8]
    # -> (1,2) = 0.3
    # -> (1,3) = 0.3 + 0.5
    # -> (2,3) = (0.5)
    k1 = np.zeros(its_len-2)

    # for writing to file the Delta G values
    keq_array = []
    dnadna_array = []
    rnadna_array = []

    for i in range(0, its_len-2):

        KEQ = keq[i]  # already formulated for back-translation
        keq_array.append(KEQ)

        DNA_DNA = -dna_dna[i+1]  # negative for back-translocation
        dnadna_array.append(DNA_DNA)

        # index 8 is the same as x_11. k[8] is for x_11
        if i < 8:
            #RNA_DNA = sum(rna_dna[:i])
            RNA_DNA = 0
        else:
            # the 7 and the 8 are because of starting at +3, dinucleotides, and
            # python indexing starting at 0
            # take the negative for back-translocation
            RNA_DNA = -(sum(rna_dna[i-7:i+1]) - sum(rna_dna[i-8:i+1]))

        rnadna_array.append(RNA_DNA)

        exponent = (a*RNA_DNA +b*DNA_DNA +c*KEQ)/RT

        rate = np.exp(-exponent)

        k1[i] = rate

    return k1


def average_rand_result(rand_results):
    """
    Keep only the top scores for each of the results in the sample. Those top
    scores will then be used to make a new object with the average and std of
    those scores. That object is then returned. This makes these objects
    different from the 'normal' result objects, where the std and mean are
    from sub-optimal results.
    """

    new_corr, new_pvals = [], []
    new_params = {'c1':[], 'c2':[], 'c3':[]}

    # Add the values from the X random versions

    # You don't want all 20 saved outcomes from each random version, you just
    # want the best ones
    for res_obj in rand_results:

        # check if this is a NaN object; skip it if so
        if res_obj.corr[0] is np.nan:
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
        new_res = Result(new_corr, new_pvals, new_params, res_obj.SEn)

        return new_res
