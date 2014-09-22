"""
The ITS class holds sequence and experimental information about an ITS.
"""
import numpy as np
import Energycalc as Ec


class ITS(object):

    def __init__(self, sequence, name='noname', PY=-1, PY_std=-1, apr=-1, msat=-1):
        # Set the initial important data.
        self.name = name
        # sequence is the non-template strand == the "same as RNA-strand"
        self.sequence = sequence + 'TAAATATGGC'
        self.PY = PY
        self.PY_std = PY_std
        self.APR = apr
        self.msat = int(msat)

        self.nr_purines = sum([1 for s in self.sequence[:20] if s in ['G','A']])

        # These are optional data which are used if the raw quantitations are
        # available
        self.quantitations = []
        # all data lists go from RNA-mer 2 to 21
        self.abortiveProb = []  # mean
        self.abortiveProb_std = []  # std

        # Raw quants
        self.rawData = {}  # one entry for each quantitation
        self.rawDataMean = -1  # one entry for each quantitation
        self.rawDataStd = -1   # one entry for each quantitation

        # FL
        self.fullLength = {}
        self.fullLengthMean = -1
        self.fullLengthStd = -1

        # helper arrays
        self._prctYield = {}
        self._RNAprodFromHere = {}

        # Redlisted sequences
        self.redlist = []
        self.redlisted = False

        # The SE sum of equilibrium constants (Keqs)
        self.SE = -1

        # make a dinucleotide list of the ITS sequence
        __seqlen = len(self.sequence)
        __indiv = list(self.sequence)
        __dinucs = [__indiv[c] + __indiv[c+1] for c in range(__seqlen-1)]

        self.dinucs = __dinucs

        # Make di-nucleotide vectors for all the energy parameters
        # goes from 2-mer till length of sequence
        self.rna_dna_di = [Ec.NNRD[di] for di in __dinucs]
        self.dna_dna_di = [Ec.NNDD[di] for di in __dinucs]
        self.delta_g_f = [Ec.deltaG_f[di] for di in __dinucs]
        self.delta_g_b = [Ec.deltaG_b[di] for di in __dinucs]

        # define the 15 dna-dna and rna-rna and keq values
        self.DgDNA15 = sum(self.dna_dna_di[:15])
        self.DgRNA15 = sum(self.rna_dna_di[:15])
        self.Dg3D15 = sum(self.delta_g_b[:15])

    def __repr__(self):
        return "{0}, PY: {1}".format(self.name, self.PY)

    def ScaleHeinRateConstants()

    def forward_transclocation(self, method='constant', constant=2):
        """
        Forward translocation is in the brownian ratched model assumed to be
        thermally driven, and not very different between nucleotides. The lack
        of strong nucleotide-specificity for forward translocation was found
        experimentally by Malinen et. al, and by Hein et. al indirectly through
        pyrophosphorolysis.
        """

        if method == 'constant':
            return np.array([constant for n in self.sequence])

        if method == 'hein_et_al':
            # /min
            pyro_forward = np.array([Ec.pyrophosphorolysis_reverse[di] for di in __dinucs])

            # scale to the malinen_et_al rate constants
            hein_scaled_forward_rate_constants = ScaleHeinRateConstants(pyro_forward)

            return hein_scaled_forward_rate_constants

        if method == 'malinen_et_al':
            # unit /s
            half_lives = np.array([[Ec.malinen_et_al_forward_translocation_halflives_ms[nt]
                                        for nt in self.sequence])
            rate_constants = np.array([Ec.HalfLife2RateConstant(hl) for hl in half_lives])

            return rate_constants


    def abortive_rates(self, method='constant', constant=3):
        """
        Rate constants for abortive product release, once the hybrid has
        become short enough.
        """
        if method == 'constant':
            return np.array([constant for n in self.sequence])

        if method == 'hybrid_energy_dependent':
            """
            After max hybrid length is reached, increase in abortive rate
            constant increases with a factor proportional to RNA-DNA free
            energy.
            """
            pass

        if method == 'hybrid_length_dependent_linear':
            """
            Increase in abortive rate constant increases linearly with
            decreasing RNA-DNA bonds.
            """
            pass

        if method == 'hybrid_length_dependent_exponential':
            """
            Increase in abortive rate constant increases linearly with
            decreasing RNA-DNA bonds. Less than linear down to length 5, then
            larger than linear.
            """
            pass

    def subsequent_backtracking_rate_constants(self, method='constant',
            constant=5):
        """
        The rate constants for further backtracking until a state with abortive
        initiation is reached.
        """
        if method == 'constant':
            return np.array([constant for n in self.sequence])

        if method == 'scrunch-dependent':
            """
            Let scale with scrunched DNA energy
            """
            pass

    def initial_backtracking_rate_constants(self, method='constant', constant=10):
        """
        The rate constant for backtracking out of a presumeably
        pre-translocated state.
        """
        if method == 'constant':
            return np.array([constant for n in self.sequence])

        if method == 'scrunch-dependent':
            """
            Let scale with scrunched DNA energy
            """
            pass

    def ntp_incorporation_coefficients(self, method='constant', constant=20):
        """
        Rate constants given in /s
        """

        if method == 'constant':
            return np.array([constant for n in self.sequence])

        if method == 'malinen_sequence_dependent':
            """
            Follow Malinen et. al's values. Though he says that the nucleotide
            incorporation rates should be considered preliminary.

            These values are half-life. From these you might have to calculate
            the rate constants. For a first order reaction half life is equal
            to 0.693/k.

            Transcription initiation is assumed to start with a 2-NMP RNA so
            this array may be used starting from the third element: [2]
            """
            half_lives = np.array([Ec.malinen_et_al_nucleotide_addition_halflives_ms[n] for
                                        n in self.sequence])
            rate_constants = np.array([Ec.HalfLife2RateConstant(hl) for hl in half_lives])

            return rate_constants

    def averageRawDataAndFL(self):
        """
        More averaging and stding
        """

        if not self.sane():
            return

        # average mean
        self.fullLengthMean = np.mean(self.fullLength.values())
        self.fullLengthStd = np.std(self.fullLength.values())

        # average raw data
        self.rawDataMean = np.mean(self.rawData.values(), axis=0)
        self.rawDataStd = np.std(self.rawData.values(), axis=0)

    def sane(self):
        """
        Verify that raw abortive data and full length data are in place
        """

        if self.rawData == []:
            print('No raw data to work with!')
            return False

        elif self.fullLength == {}:
            print('No full length (FL) to work with!')
            return False

        else:
            return True

    def calc_PY(self):
        """
        Use the raw data to calculate PY (mean and std)
        """

        if not self.sane():
            return

        # store the PY for each quantitation
        self._PYraw = {}

        for quant in self.quantitations:

            totalRNA = sum(self.rawData[quant]) + self.fullLength[quant]
            self._PYraw[quant] = self.fullLength[quant]/totalRNA

        self.PY = np.mean([py for py in self._PYraw.values()])
        self.PY_std = np.std([py for py in self._PYraw.values()])

    def calc_AP(self):
        """
        Use the raw data to calculate AP (mean and std)
        """

        if not self.sane():
            return

        # store the AP for each quantitation
        self._APraw = {}
        self.totAbort = {}
        self.totRNA = {}

        # go through each quantitation
        for quant in self.quantitations:

            # define tmp arrays to work with
            rawD = self.rawData[quant]
            fullL = self.fullLength[quant]

            totalAbortive = sum(rawD)
            totalRNA = totalAbortive + fullL

            # Save these two for posterity
            self.totAbort[quant] = totalAbortive
            self.totRNA[quant] = totalRNA

            rDRange = range(len(rawD))

            # percentage yield at this x-mer
            prctYield = [(rD/totalRNA)*100 for rD in rawD]

            # amount of RNA produced from here and out
            RNAprodFromHere = [sum(rawD[i:]) + fullL for i in rDRange]

            # percent of RNAP reaching this position
            prctRNAP = [(rpfh/totalRNA)*100 for rpfh in RNAprodFromHere]

            # probability to abort at this position
            # More accurately: probability that an abortive RNA of length i is
            # produced (it may backtrack and be slow to collapse, producing
            # little abortive product but reducing productive yield)
            self._APraw[quant] = [prctYield[i]/prctRNAP[i] for i in rDRange]

        self.abortiveProb = np.mean(self._APraw.values(), axis=0)
        self.abortiveProb_std = np.std(self._APraw.values(), axis=0)

        self.totAbortMean = np.mean(self.totAbort.values(), axis=0)
        self.totRNAMean = np.mean(self.totRNA.values(), axis=0)

    def calc_keq(self, c1, c2, c3):
        """
        Calculate Keq_i for each i in [2,20]

        """

        # hard-coded constants hello Fates!
        RT = 1.9858775*(37 + 273.15)/1000  # divide by 1000 to get kcalories

        its_len = 21
        dna_dna = self.dna_dna_di[:its_len-1]
        rna_dna = self.rna_dna_di[:its_len-2]
        dg3d = self.delta_g_b[:its_len-2]

        # equilibrium constants at each position
        import optim

        self.keq = optim.keq_i(RT, its_len, dg3d, dna_dna, rna_dna, c1, c2, c3)

    def calc_AbortiveYield(self):
        """
        Calculate abortive to productive ratio

        """
        if not self.sane():
            return

        # store the PY for each quantitation
        self._AYraw = {}

        for quant in self.quantitations:

            totalRNA = sum(self.rawData[quant]) + self.fullLength[quant]
            self._AYraw[quant] = self.rawData[quant]/totalRNA

        self.AY = np.mean([py for py in self._PYraw.values()])
        self.AY_std = np.std([py for py in self._PYraw.values()])

