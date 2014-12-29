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
    These are results from the simulation for a given its_length; for Normal
    simulations, they are the top 20 correlation coefficients, pvals, fitted
    parameters (c1, c2, c3) and the final values. By top 20 is meant from the
    top 20 parameter combinations.

    You get in slightly different things here for Normal, Random, and
    cross-correlated results.

    Normal results are the (max) 20 best for each its-length (you present
    corr_max).

    Random results are the X*20 best, where X is the number of times random
    sequences are generated for each its-length. For Random results, mean and
    std are what's interesting (the mean and std of the best for each random
    sample)

    Cross-correlated results have len(corr)=X, where X is the number of
    re-samplings of half the dataset. Corr now represent the optimal
    correlations for each of the resamplings: each of these values are the
    result of sampling 50% of the dataset, getting optimal parameters, and
    correlating the remaining 50% using those parameters. The question
    becomes how to represent these 'X cross-correlated correlation scores, and
    their parameter values!'

    """
    def __init__(self, corr=[np.nan], pvals=np.nan, params=np.nan,
            msat_corr_mean=[np.nan], msat_corr_std=[np.nan]):

        # Create a placeholder list for msat parameter estimation
        self.all_corr_for_msat = []
        self.all_corr_for_msat_mean = msat_corr_mean
        self.all_corr_for_msat_std = msat_corr_std

        self.corr = corr
        self.pvals = pvals
        self.params = params

        # calculate the mean, std, max, and min values for correlation
        # coefficients, pvalues, and SEn
        self.corr_median = np.median(corr)
        self.corr_mean = np.mean(corr)
        self.corr_std = np.std(corr)
        max_indx = np.abs(corr).argmax()  # correlation may be negative

        # corr max may be nan
        if len(corr) == 1 and np.isnan(corr[0]):
            self.corr_max = np.nan
        else:
            self.corr_max = corr[max_indx]

        self.pvals_median = np.median(pvals)
        self.pvals_mean = np.mean(pvals)
        self.pvals_std = np.std(pvals)
        self.pvals_min = np.min(pvals)

        # The parameters are not so easy to do something with; the should be
        # supplied in a dictionray [c1,c2,c3] = arrays; so make mean, std,
        # min, max for them as well

        # handle nan
        if params is np.nan:
            params = {'c1':[np.nan],
                      'c2':[np.nan],
                      'c3':[np.nan]}

        # extract mean, std etc
        self.params_median = dict((k, np.median(v)) for k,v in params.items())
        self.params_mean = dict((k, np.mean(v)) for k,v in params.items())
        self.params_std = dict((k, np.std(v)) for k,v in params.items())
        self.params_max = dict((k, np.max(v)) for k,v in params.items())
        self.params_min = dict((k, np.min(v)) for k,v in params.items())

        # You need the parameters for the highest correlation coefficient, since
        # that's what you're going to plot for Normal results
        # You rely on that v is sorted in the same order as corr ... relying on
        # sorting, whachagot next for me.
        # This value is not relevant for cross corr or random.
        # XXX is this where the random bias is seeping in?
        self.params_best = dict((k, v[0]) for k,v in params.items())


def main_optim(rna_range, ITSs, ranges, analysisInfo, measure,
        msat_normalization, msat_param_estimate):
    """
    Call the core optimialization wrapper but treat different analysis cases
    differently.
    """

    # make 'int' into a 'list' containing just that int
    if type(rna_range) is int:
        rna_range = range(rna_range, rna_range+1)

    results = {'Normal':{}, 'Random':{}, 'Cross Correlation':{}}

    # Get the first iteration of results based on a wide parameter range
    for rna_length in rna_range:

        # subsequently set to true depending on input
        normal_obj = False
        random_obj = False
        crosscorr_obj = False

        # 'normal' results (full dataset)
        if analysisInfo['Normal']['run']:
            normal_obj = core_optim_wrapper(rna_length, ITSs, ranges, measure,
                    msat_normalization)

        # Randomize
        if analysisInfo['Random']['run']:
            iterations = analysisInfo['Random']['nr_iterations']
            random_obj = randomize_wrapper(rna_length, ITSs, ranges, measure,
                    msat_normalization, msat_param_estimate,
                    randomize=iterations)

        # Cross correlation
        if analysisInfo['CrossCorr']['run']:
            iterations = analysisInfo['CrossCorr']['nr_iterations']
            crosscorr_obj = crosscorr_wrapper(rna_length, ITSs, ranges,
                    measure, msat_normalization, msat_param_estimate,
                    crosscorr=iterations)

        results['Normal'][rna_length] = normal_obj
        results['Random'][rna_length] = random_obj
        results['Cross Correlation'][rna_length] = crosscorr_obj

    return results


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
        random_its = DNASequenceGenerator(length=19)
        copy_ITSs.append(ITS(random_its, name=its.name, PY=its.PY,
                            msat=its.msat))

    return copy_ITSs


def DNASequenceGenerator(length):
    """Generate a random DNA sequence. length is a positive integer > 0 """
    import random
    gatc = list('GATC')
    return 'AT'+ ''.join([random.choice(gatc) for dummy1 in range(length)])


def MsatParamEstimateCorrleation(ITSs, params, msat_normalization, measure):

    correlation = []

    PYs = [its.PY for its in ITSs]
    # Get the optimal parameter values for the fitted ITSs
    parameters = [np.array([params[p]]) for p in ('c1', 'c2', 'c3')]

    for max_rna_len in range(2,21):
        r, p, values = CorrelatePYandMeasure(parameters, max_rna_len, ITSs,
                msat_normalization, measure, PYs)

        correlation.append(r)

    return correlation


def randomize_wrapper(rna_len, ITSs, ranges, measure, msat_normalization,
                        msat_param_estimate, randomize=0):
    """
    Generate random ITS sequences and do the simulation with them
    Return the average of the best scores you obtained
    """

    randResults = []
    for dummy in range(randomize):

        ITS_random = randomize_ITS_sequence(ITSs)

        rand_result = core_optim_wrapper(rna_len, ITS_random, ranges,
                                            measure, msat_normalization)

        if rand_result and msat_param_estimate:

            corr = MsatParamEstimateCorrleation(ITS_random, rand_result.params_best,
                                                    msat_normalization, measure)
            rand_result.all_corr_for_msat = corr

        # skip if you don't get anything from this randomizer
        if not rand_result:
            continue
        else:
            randResults.append(rand_result)

    # select the average of the top correlations to report
    avrgdResults = pick_top_results(randResults, msat_param_estimate)

    return avrgdResults


def crosscorr_wrapper(rna_len, ITSs, ranges, measure, msat_normalization,
                      msat_param_estimate, crosscorr=0):
    """
    Wrapper around core_optim_wrapper that deals with cross-correlation.
    Optimialization is performed on one half of the ITSs and the correlation is
    obtained from the second half using the parameters obtained from the first
    half.

    This proceedure is done 'crosscor' times for each function call
    (presumeably being done for each its-length, and more importantly for each
    parameter combination).

    The point is to compare with normal parameter runs: you have to defend your
    method: oh, you're checking the whole parameter space and picking the best
    correlation! How impudent.

    This is one answer: the cross correlation approach: for each its-len,
    'crosscorr' times this is done: split your ITSs in half. for one half, find
    the parameter values that give optimal correlation. Then, use those
    parameter values to get a (one only) correlation and p-value for the
    correlation between the measure you're using and the PY.

    You thus end up with 'crosscorr' optimal values: how to select the best of
    these? Select the median, or the mean?

    """

    retro_results = []
    for _ in range(crosscorr):

        # Choose 50% of the ITS randomly; both energies and PYs
        ITS_fit, ITS_compare = ITS_RandomSplit(ITSs)

        fit_result = core_optim_wrapper(rna_len, ITS_fit, ranges, measure,
                                        msat_normalization)

        # Skip if you don't get a result (should not be a problem with large ranges)
        if not fit_result:
            continue

        # Get the optimal parameter values for the fitted ITSs
        par_order = ('c1', 'c2', 'c3')
        fitted_params = [np.array([fit_result.params_best[p]]) for p in par_order]

        # run the rest of the ITS with the optimal parameters from fitting-set
        control_result = core_optim_wrapper(rna_len, ITS_compare, fitted_params,
                                            measure, msat_normalization)

        # if using msat parameter estimation, save correlation for the full
        # rna_length range
        if control_result and msat_param_estimate:

            corr = MsatParamEstimateCorrleation(ITS_compare, control_result.params_best,
                                                    msat_normalization, measure)
            control_result.all_corr_for_msat = corr

        # Skip if you don't get a result (should B OK with large ranges)
        if not control_result:
            continue
        else:
            retro_results.append(control_result)

    # average the results (keeping mean and std) and return
    # cross corr results
    return pick_top_results(retro_results, msat_param_estimate)


def core_optim_wrapper(rna_len, ITSs, ranges, measure, msat_normalization):
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
        results = [my_pool.apply_async(_multi_calc, (p, rna_len, ITSs, measure,
                                            msat_normalization)) for p in divide]
        my_pool.close()
        my_pool.join()
        # flatten the output
        all_results = sum([r.get() for r in results], [])
    else:  # the non-parallell version for debugging and no multi-range calculations
        all_results = sum([_multi_calc(*(p, rna_len, ITSs, measure, msat_normalization))
                            for p in divide], [])

    # All_results is a list of tuples on the following form: ((measure_values, par, rp, pp))
    # Sort them on the pearson/spearman correlation pvalue, 'pp' and pick top 20
    top_hits = sorted(all_results, key=itemgetter(3))[:20]

    # if top_hits is empty, return NaNs results (created by default)
    if top_hits == []:
        return Result()

    # now make separate corr, pvals, params, and SEn arrays from these
    # these corr values represent corr(sum(SEn[:its_len], PY)) for the
    # different c1, c2 ,c3 combinations
    par = [c[1] for c in top_hits]
    corr = np.array([c[2] for c in top_hits])
    pvals = np.array([c[3] for c in top_hits])

    print 'mean top 20 corr', np.nanmean(corr)
    #print 'top 3 corr and pval'
    #itr=1
    #for cr, pvl in zip(corr, pvals):
        #print '{0}: corr: {1}, pval: {2}'.format(itr, cr, pvl)
        #if itr == 3:
            #break
    #print 'top 20 corr', corr[0]
    #debug()
    # XXX in here there is a weak bias toward positive correlation coefficients.
    # (= rna_length=msat=20). The bias is not because of multiple competing
    # pvalues. It seems that if correlation is positive, then all parameter
    # combinations give positive correlation, and vica versa. So it is the
    # sequence set that determines if correlation should be positive or
    # negative, not parameter estimation. This drags! Is there an explanation?
    # Should you construct the random sequences with the same nucleotide
    # frequency as found in the DG100 libraries?

    # Return parameters in a dictionary form
    # pars[c1, c2, c3] = [array]
    parameters = {}
    for par_nr in range(1, len(ranges)+1):
        parameters['c{0}'.format(par_nr)] = [p[par_nr-1] for p in par]

    # make a Result object
    result_obj = Result(corr, pvals, parameters)

    print 'time: ', time.time() - t1

    return result_obj


def CorrelatePYandMeasure(parameters, max_rna_length, ITSs, msat_normalization,
                            measure, y):

    all_keqs = keq_calc(parameters, max_rna_length, ITSs, msat_normalization)

    # correlation
    if measure == 'SE':
        values = [np.nansum(keqs) for keqs in all_keqs]

    elif measure == 'product':
        values = [np.prod(keqs) for keqs in all_keqs]

    elif measure == 'AvgKbt':
        values = [np.nanmean(keqs) for keqs in all_keqs]

    r, p = spearmanr(y, values)

    return (r, p, values)


def _multi_calc(param_combo, rna_len, ITSs, measure, msat_normalization):
    """
    Evaluate the model for all parameter combinations. Return the final values,
    the parameters, and the correlation coefficients.

    When using msat_normalization with avgkbt, don't go further than msat whe
    """

    all_hits = []
    PYs = [its.PY for its in ITSs]

    for parc in param_combo:
        r, p, values = CorrelatePYandMeasure(parc, rna_len, ITSs,
                                             msat_normalization, measure, PYs)

        # ignore parameter correlation where the correlation is a nan
        if np.isnan(r):
            continue

        all_hits.append((values, parc, r, p))

    return all_hits


def keq_calc(start_values, rna_len, ITSs, msat_normalization):

    """
    k1 = exp(-(c1*rna_dna_i + c2*dna_dna_{i+1} + c3*Keq_{i-1}) * 1/RT)

    Dna-bubble one step ahead of the rna-dna hybrid; Keq one step behind.
    When the rna-dna reaches length 9, the you must subtract the hybrid for
    each step forward. The hybrid oscillates between a 8bp and a
    9bp hybrid during translocation.

    For msat normalization, report Keq = NaN when exceeding msat as a first
    attempt.

    Initial transcription starts with a 2-nt RNA-dna hybrid. Then it performs
    the first translocation step.
    """

    all_keqs = []

    for its in ITSs:

        # must shift by minus 1 ('GATTA' example: GA, AT, TT, TA -> len 5, but 4
        # relevant dinucleotides)

        # first -1 because of python indexing. additional -1 because only (1,2)
        # information is needed for 3. ATG -> [(1,2), (2,3)]. Only DNA-DNA needs
        # both.
        (c1, c2, c3) = start_values
        its.calc_keq(c1, c2, c3, msat_normalization, rna_len)

        all_keqs.append(its.keq)

    return np.asarray(all_keqs)


def keq_i(RT, rna_len, dg3d, dna_dna, rna_dna, a, b, c):
    """
    Calculating backward translocation equilibrium constant.

    Recall that the energies are in dinucleotide form. Thus ATG -> [0.4, 0.2]

    Tip for the future; start at +1 also for nucleotides, so all arrays have the
    same length. Just have the first values as 0. Makes it much easier to
    understand the code later. The 0s are placeholders.

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
    # translocation step

    # ATGCT -> [0.3, 0.5, 0.2, 0.8]
    # -> (1,2) = 0.3
    # -> (1,3) = 0.3 + 0.5
    # -> (2,3) = (0.5)

    # initialize empty rates. -1 because rna_len=2 should have one
    k1 = np.zeros(rna_len-1)

    # for writing to file the Delta G values
    dgdinuc_array = []
    dnadna_array = []
    rnadna_array = []

    # i=0 corresponds to rna_len=2
    for i in range(0, rna_len-1):

        DGDINUC = dg3d[i]  # already formulated for back-translation
        dgdinuc_array.append(DGDINUC)

        DNA_DNA = -dna_dna[i+1]  # negative for back-translocation
        dnadna_array.append(DNA_DNA)

        # index 8 is the same as rna lenth 10
        if i < 8:
            #RNA_DNA = sum(rna_dna[:i])
            RNA_DNA = 0
        else:
            # take the negative for back-translocation
            RNA_DNA = -(sum(rna_dna[i-7:i+1]) - sum(rna_dna[i-8:i+1]))

        rnadna_array.append(RNA_DNA)

        exponent = (a*RNA_DNA +b*DNA_DNA +c*DGDINUC)/RT

        rate = np.exp(-exponent)

        k1[i] = rate

    return k1


def pick_top_results(several_results, msat_param_estimate):
    """
    Keep only the top scores for each of the results in the sample. Those top
    scores will then be used to make a new object with the average and std of
    those scores. That object is then returned. This makes these objects
    different from the 'normal' result objects, where the std and mean are
    from sub-optimal results.

    Note: when coming from crossCorr (that's bad design..) each res_obj in the
    rand_results will only have one (1) value in .corr and .pvals -- it's the
    correlation obtained with group1's optimal parameters on group2.

    Note: when comming from randomize, each res_obj will have the normal top 20
    correlations for all the parameter combinations. Here, we select from all X
    randizes the topp correlation coefficient, pvals, and parameter values.
    Intersting, what are the parameter values for random? Not 0 since we're
    going from 0 to 1 ... should it be 0.5? I dunno, might be a bias there.
    """

    new_corr, new_pvals = [], []
    new_params = {'c1':[], 'c2':[], 'c3':[]}

    # You don't want all 20 saved outcomes from each random version, you just
    # want the best ones. By best, I mean highest and lowest correlation.
    # for cross-corr, there is only one value -- you pass it through here
    # actually just to make a new 'results' object from many results objects.
    # It's convoluted, I know, but I'm not refactoring any more now.
    for res_obj in several_results:

        # check if this is a NaN object; skip it if so
        if res_obj.corr[0] is np.nan:
            continue

        new_corr.append(res_obj.corr[0])
        new_pvals.append(res_obj.pvals[0])

        for k, v in res_obj.params.items():
            new_params[k].append(v[0])

    # Check if new_corr is empty (means all results were NaN)
    if new_corr == []:
        result = Result()
    else:
        # making a new result object where the attributes take on NEW MEANINGS :((((
        # never do this again kthnx.
        if msat_param_estimate:
            all_full_corr = np.asarray([r.all_corr_for_msat for r in several_results])
            mean_full_corr = np.nanmean(all_full_corr, axis=0)
            std_full_corr = np.nanstd(all_full_corr, axis=0)
            result = Result(msat_corr_mean=mean_full_corr, msat_corr_std=std_full_corr)
        else:
            result = Result(new_corr, new_pvals, new_params)

    return result
