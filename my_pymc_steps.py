__author__ = 'Brandon C. Kelly'
# Originally located at: https://github.com/kasperschmidt/pymc_steps

import numpy as np
import sys
# import pymc3 as pymc
import pymc
# from pymc.six import print
# from pymc.Node import ZeroProbability
from scipy.linalg import cholesky


class RobustAdaptiveMetro(object):
    """
    Robust Adaptive Metropolis Algorithm (RAM).

    Reference: Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate, M. Vihola, 2012, Statistics &
    Computing, 22, 997-1008

    Much of the code for this class is just copied from the PyMC code for the Metropolis class. I then just modified it
    to include the robust adaptation step.
    """

    def __init__(self, stochastic, target_rate, proposal_covar=None, proposal_distribution=None, verbose=-1,
                 tally=True, stop_adapting=sys.maxsize, tdof=8):
        # Metropolis class initialization

        # Initialize superclass
        pymc.step_methods.__init__(self, [stochastic], tally=tally)

        # Initialize hidden attributes
        self._accepted = 0.
        self._rejected = 0.
        self._current_iter = 0
        self._stop_adapting = stop_adapting  # Stop adapting the proposal covariance matrix after this many iterations
        self._decay_rate = 2.0 / 3.0  # Gamma in the notation of Vihola 2012.
        self._tdof = tdof  # Degrees of freedom when using a t-distribution for the proposals, otherwise ignored.
        self._id = 'RobustAdaptiveMetro_' + '_'.join([p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += ['_accepted', '_rejected', '_current_iter', '_cholesky_factor', '_proposal_distribution',
                        '_decay_rate', 'target_rate', '_stop_adapting', '_tdof']
        self._tuning_info = ['_cholesky_factor']

        # Set public attributes
        self.stochastic = stochastic
        self.target_rate = target_rate

        if verbose > -1:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose

        # Determine size of stochastic
        if isinstance(self.stochastic.value, np.ndarray):
            self._dim = len(self.stochastic.value.ravel())
        else:
            self._dim = 1

        if proposal_covar is None:
            # Automatically set the initial covariance matrix of the proposals.
            if self.stochastic.value != 0.:
                if self._dim == 1:
                    self._cholesky_factor = abs(self.stochastic.value) * 0.1
                else:
                    self._cholesky_factor = np.diag(abs(self.stochastic.value) * 0.1)
            else:
                if self._dim == 1:
                    self._cholesky_factor = 0.1
                else:
                    self._cholesky_factor = np.eye(np.shape(self.stochastic.value)) * 0.1
        else:
            # Compute the cholesky factor for the covariance matrix of the proposals. The algorithm adapts the
            # covariance matrix through computations on the cholesky factor. If the parameter is scalar-valued then
            # then this is ignored.
            if self._dim == 1:
                # Cholesky factor is just standard deviation for scalars
                self._cholesky_factor = np.sqrt(proposal_covar)
            else:
                # Cholesky factor is upper triangular, needed for the rank 1 update
                self._cholesky_factor = cholesky(proposal_covar)

        # Initialize proposal deviate with array of zeros
        self._proposal_deviate = np.zeros(np.shape(self.stochastic.value), dtype=float)

        # If no dist argument is provided, assign a proposal distribution automatically.
        if not proposal_distribution:

            # Pick Gaussian by default
            self._proposal_distribution = "Normal"

        else:

            if proposal_distribution.capitalize() in self._valid_proposals:
                self._proposal_distribution = proposal_distribution
            else:
                raise ValueError("Invalid proposal distribution '%s' specified for Metropolis sampler."
                                 % proposal_distribution)

    _valid_proposals = ['Normal', 'T']

    @staticmethod
    def competence(s):
        """
        The competence function for the Robust Adaptive Metropolis Algorithm
        """
        if s.dtype is None:
            return .5

        if not s.dtype in pymc.datatypes.float_dtypes:
            # If the stochastic's binary or discrete, I can't do it.
            return 0
        else:
            return 1

    def hastings_factor(self):
        """
        Proposal is symmetric so just return zero.
        """
        return 0.

    def step(self):
        """
        Perform the Robust Metropolis step.
        """

        self._current_iter += 1

        # Probability and likelihood for s's current value:

        if self.verbose > 2:
            print()
            print(self._id + ' getting initial logp.')

        logp = self.logp_plus_loglike

        if self.verbose > 2:
            print(self._id + ' proposing.')

        # Sample a candidate value
        unit_proposal = self.unit_proposal()  # First get unit proposal from a normal or t distribution.
        # Now scale this unit proposal
        if self._dim == 1:
            centered_proposal = self._cholesky_factor * unit_proposal
        else:
            centered_proposal = np.transpose(self._cholesky_factor).dot(unit_proposal)

        self.stochastic.value = self.stochastic.value + centered_proposal

        # Probability and likelihood for s's proposed value:
        try:
            logp_p = self.logp_plus_loglike

        except:# ZeroProbability:

            # Reject proposal
            if self.verbose > 2:
                print(self._id + ' rejecting due to ZeroProbability.')
            self.reject()

            # Increment rejected count
            self._rejected += 1

            if self.verbose > 2:
                print(self._id + ' returning.')

            return

        if self.verbose > 2:
            print('logp_p - logp: ', logp_p - logp)

        # Evaluate acceptance ratio
        alpha = min(1.0, np.exp(logp_p - logp))

        unif = np.random.uniform()

        if unif > alpha:

            # Revert s if fail
            self.reject()

            # Increment rejected count
            self._rejected += 1
            if self.verbose > 2:
                print(self._id + ' rejecting')
        else:
            # Increment accepted count
            self._accepted += 1
            if self.verbose > 2:
                print(self._id + ' accepting')

        if (self._current_iter < self._stop_adapting) & (np.isfinite(alpha)):
            # Update the scale matrix of the proposals
            self.update_covar(alpha, unit_proposal, centered_proposal)

        if self.verbose > 2:
            print(self._id + ' returning.')

    def update_covar(self, alpha, unit_proposal, centered_proposal):
        """
        Method to update the covariance matrix (actually, its Cholesky decomposition), based on the
        Metropolist ratio, the proposed value and the value of the unit proposal.

        :params alpha: The Metropolis ratio of the proposed and current values.
        :param proposed_value (array-like): The centered value of the proposed parameter, given by the product of the
            unit proposal and the Cholesky factor.
        :param unit_proposal (array-like):  The value generated by the proposal object.
        """
        # The step size sequence for the scale matrix update. This is eta_n in the notation
        # of Vihola (2012).
        step_size = min(1.0, self._dim / (self._current_iter + 1.0) ** self._decay_rate)

        if self._dim == 1:
            # Parameter is scalar-valued so the update is done analytically
            log_width = np.log(self._cholesky_factor) + 0.5 * np.log(1.0 + step_size * (alpha - self.target_rate))
            self._cholesky_factor = np.exp(log_width)
        else:
            # Parameter is vector-valued, so do rank-1 Cholesky update/downdate
            unit_norm = np.sqrt(unit_proposal.dot(unit_proposal))  # L2 norm of the vector

            # Rescale the proposal vector for updating the scale matrix cholesky factor
            scaled_proposal = np.sqrt(step_size * abs(alpha - self.target_rate)) / unit_norm * centered_proposal

            # Update or downdate?
            downdate = alpha < self.target_rate

            # Perform the rank-1 update (or downdate) of the scale matrix cholesky factor
            self.CholUpdateR1(scaled_proposal, downdate=downdate)

    def CholUpdateR1(self, v, downdate=False):
        """
        Perform the rank-1 Cholesky update (or downdate). Suppose we have the Cholesky decomposition for a matrix, A.
        The rank-1 update computes the Cholesky decomposition of a new matrix B, where B = A + v * v.transpose(). The
        downdate corresponds to B = A - v * v.transpose(). The input array will be overwritten.

        :param v (array-like): A vector describing the update or downdate.
        :param downdate: A boolean variable describing whether to perform the downdate (downdate=True).
        """
        sign = 1.0
        if downdate:
            sign = -1.0

        for k in range(self._cholesky_factor.shape[0]):
            r = np.sqrt(self._cholesky_factor[k, k] * self._cholesky_factor[k, k] + sign * v[k] * v[k])
            c = r / self._cholesky_factor[k, k]
            s = v[k] / self._cholesky_factor[k, k]
            self._cholesky_factor[k, k] = r
            if k < self._cholesky_factor.shape[0] - 1:
                self._cholesky_factor[k, k + 1:] = (self._cholesky_factor[k, k + 1:] + sign * s * v[k + 1:]) / c
                v[k + 1:] = c * v[k + 1:] - s * self._cholesky_factor[k, k + 1:]

    def tune(self, verbose=0):
        """Tuning is done during the entire run, independently from the Sampler
        tuning specifications. """
        return False

    def reject(self):
        # Sets current s value to the last accepted value
        # self.stochastic.value = self.stochastic.last_value
        self.stochastic.revert()

    def unit_proposal(self):
        """
        This method is called by step() to generate unit proposal values. The actual proposed values are constructed
        from these via an Affine transformation based on the current value of the proposal Cholesky factor and the
        current parameter value.
        """
        if self._proposal_distribution == "Normal":
            if self._dim == 1:
                unit_deviate = np.random.standard_normal()
            else:
                unit_deviate = np.random.standard_normal(self._dim)
        else:
            if self._dim == 1:
                unit_deviate = np.random.standard_t(self._tdof)
            else:
                unit_deviate = np.random.standard_t(self._tdof, self._dim)

        return unit_deviate
