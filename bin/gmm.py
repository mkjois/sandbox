#!/usr/bin/env python
"""
A module (and tool) to train and evaluate Gaussian Mixture Models.
"""

USAGE="""
A simple tool for training and evaluating Gaussian Mixture Models.

Observation data vectors or a list of files containing sufficient
statistics may be read from standard input.  Depending on the options
invoked, a parameterized model or log likelihoods or sufficient
statistics or logging information may be written to standard output.
By default, logging information is written to standard error.

Basic usage:

# Initialize a model with the k-means++ algorithm
gmm.py -k K < DATA > MODEL_INIT

# An iteration of the EM algorithm
gmm.py -em MODEL < DATA > MODEL_NEW

# Evaluate data likelihood
gmm.py MODEL < DATA > LOG_LIKELIHOODS


Advanced usage:

# Batched processing: map E-step
gmm.py -e MODEL < DATA.partI > STATISTICS.partI

# Batched processing: reduce M-step
find . -name "STATISTICS.part*" | gmm.py -m > MODEL

# Use up to N data vectors, chosen uniformly at random
gmm.py -k K -n N < DATA > MODEL_INIT

# Repeat EM for I iterations, caching all data in memory
gmm.py -emi I MODEL < DATA > MODEL_NEW

# Train start-to-finish, caching some data in memory
gmm.py -k K -n N -emi I < DATA > MODEL

# Set the variance floor (scaled relative to global variance)
gmm.py -em -f F MODEL < DATA > MODEL_NEW

# Read input as raw floats, write model as binary (cPickle)
gmm.py -w W -b -k K < DATA_BINARY > MODEL_BINARY

# Use specified number of CPUs (0 --> no multiprocessing)
gmm.py -c C -k K -em < DATA > MODEL

# Overwrite the input model (and log to stdout)
gmm.py -em -o MODEL MODEL < DATA > LOG

# Output mixture likelihoods for each observation, as text
gmm.py -l mix MODEL < DATA > MIXTURE_LIKELIHOODS

# Output total data likelihood, as raw float
gmm.py -b -s MODEL < DATA > DATA_LIKELIHOOD_BINARY


Formats:

The default input is a newline-delimited sequence of text, each line
representing a vector of floating point numbers formatted as
whitespace-delimited text.  Use -w to read little-endian floats of a
specified feature width.

By default the model is output in human-readable JSON format as:

{"weights": [...], "means": [[...],...], "variances": [[...],...]}

By default, sufficient statistics are output in JSON format:

{"counts"=[...], "sums": [[...],...], "sums_squares": [[...],...],
 "data_ll"=...}

Note that this also includes the data log likelihood, which isn't
strictly speaking a sufficient statistic, but is very useful
information to retain, e.g., for terminating EM.

By default, log likelihoods for each observation (e.g. frame) are
output as newline-delimited text.  If "-l mix" is specified,
newline-delimited vectors of per-mixture likelihoods are output as
space-delimited text floating point representations.  If the -s
option is specified, a single summation is provided for all of the
data, rather than one line per observation.

Specifying the -b option will output a model or sufficient statistics
as an implementation-dependent (cPickle) serialization; for log
likelihoods, -b outputs a stream of little-endian floats.

TODO: serialize GMM._matrix rather than pickling the entire object
TODO: remove the multiprocessing pool
TODO: split/merge for decrementing/incrementing number of mixtures.
TODO: remove defunct mixtures?
TODO: full covariance matrix instead of diagonal variance vector.
"""

import bisect
from itertools import imap, izip
import json
import logging
import multiprocessing
import random
import scipy
from scipy.linalg import norm
from scipy.misc import logsumexp
import time

# Global variables for multiprocessing
POOL = None
CHUNKSIZE = 1000

# Global variables as in HTK
MINMIX = 10e-5
LZERO = -1.0e10
LSMALL = -0.5e10
MINLARG = 2.45e-308

class GMM(object):
    """
    A Gaussian Mixture Model with diagonal covariances.
    """
    def __init__(self, parameters=None, data=None, k=1):
        """
        Initialize a GMM in any of the following ways:

        1. Default: object with empty parameters

        2. If an attribute dictionary is provided, load and convert to
        internal representation: store log weights, inverse variance,
        and Gaussian constants for efficient computation.

        3. If data are provided, means are set via the k-means++
        initialization; mixture variances are set to the global
        variance; mixture weights are set uniformly.

        A GMM's pool attribute may be used to specify multiprocessing
        """
        if data is None:
            if parameters is None:
                self.weights = None
                self.means = None
                self.variances = None
            else:
                self.weights = scipy.array(parameters['weights'])
                self.means = map(scipy.array, parameters['means'])
                self.variances = map(scipy.array, parameters['variances'])
        else:
            data = list(data) # Ensure data can be iterated multiple times
            logging.debug("caching %d data samples" % len(data))

            # Compute global variance and set uniform weights
            global_variance = scipy.var(scipy.array(data), 0)
            logging.debug("Global variance:\n%s" % global_variance)
            self.variances = [global_variance] * k
            self.weights = [1.0 / k] * k

            # Sample from distributions proportional to distances of nearest mean
            means = []
            means.append(choice(data)) # First mean uniformly at random
            if POOL:
                nearest_dists = POOL.map(_l2norm, ((means[0], x) for x in data), CHUNKSIZE)
            else:
                nearest_dists = [_l2norm((means[0], x)) for x in data]
            for i in range(1, k):
                logging.debug('Sampling mean ' + str(i))
                means.append(choice(data, nearest_dists))
                if POOL:
                    new_dists = POOL.imap(_l2norm, ((means[-1], x) for x in data), CHUNKSIZE)
                else:
                    new_dists = (_l2norm((means[-1], x)) for x in data)
                nearest_dists = map(min, izip(nearest_dists, new_dists))
            self.means = means

    def __repr__(self):
        return repr(self.parameters)

    @property
    def parameters(self):
        """
        Converted internal scipy arrays into JSON-serializable lists.
        """
        parameters = {}
        parameters['weights'] = list(self.weights)
        parameters['means'] = map(list, self.means)
        parameters['variances'] = map(list, self.variances)
        return parameters

    @property
    def weights(self):
        """
        Store log weights internally
        """
        if self._logweights is None:
            return None
        else:
            return scipy.exp(self._logweights)

    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._logweights = None
        else:
            logweights = []
            for weight in weights:
                if weight < MINLARG:
                    logweight = LZERO
                else:
                    logweight = scipy.log(weight)
                    if logweight < LSMALL:
                        logweight = LZERO
                logweights.append(logweight)
            self._logweights = scipy.array(logweights)
        self._set_matrix()

    @weights.deleter
    def weights(self):
         self._logweights = None
         self._set_matrix()

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, means):
        self._means = means
        self._set_matrix()

    @means.deleter
    def means(self):
        self._means = None
        self._set_matrix()

    @property
    def variances(self):
        """
        Variances of the GMM (internally stored as inverses).  Setting
        this property updates precomputed Gaussian constants.
        """
        if self._invvars is None:
            return None
        else:
            return [1.0 / invvar for invvar in self._invvars]

    @variances.setter
    def variances(self, variances):
        if variances is None:
            self._invvars = None
        else:
            self._invvars = [1.0 / variance for variance in variances]
            self._gconsts = [len(variance) * scipy.log(2 * scipy.pi) / -2 # normalizing constant
                             - scipy.log(scipy.prod(variance)) / 2 # determinant of covariance
                              for variance in variances]
        self._set_matrix()

    @variances.deleter
    def variances(self):
        self._invvars = None
        self._gconsts = None
        self._set_matrix()

    def _set_matrix(self):
        """
        A matrix representation of the GMM that is prepared for
        optimized evaluation as a matrix-matrix multiplication when a
        matrix representation of the input is expanded as:

        X[n] = [1, x[n][0], ..., x[n][-1], x[n][0]**2, ..., x[n][-1]**2]'

        Each of the components i is a row of a matrix:

        A[i] = [_logweights[i]+_gconsts[i]-0.5*sum(_means[i]**2*_invvars[i]),
         _means[i][0]*_invvars[i][0], ..., _means[i][-1]*_invvars[i][-1],
         -0.5*_invvars[i][0], ..., -0.5*_invvars[i][-1]]

        This matrix is transposed so likelihoods may be evaluated as XA'
        """
        if any(getattr(self, a, None) is None for a in ['_logweights', '_means', '_invvars']):
            self._matrix = None
        else:
            K = len(self._logweights)
            assert K == len(self._means)
            assert K == len(self._invvars)
            D = len(self._means[0])
            assert all(D == len(v) for v in self._means)
            assert all(D == len(v) for v in self._invvars)
            rows = []
            for i in range(K):
                rows.append(scipy.concatenate([[self._logweights[i]+self._gconsts[i]-0.5*sum(self._means[i]**2 * self._invvars[i])],
                                               self._means[i] * self._invvars[i],
                                               -0.5*self._invvars[i]]))
            self._matrix = scipy.matrix(rows).transpose()

    def pdf(self, x):
        return scipy.exp(self.logpdf(x))

    def logpdf(self, x, permix=False):
        """
        Return the log-likelihood of an observation.
        If permix=True, return likelihoods for each mixture.
        """
        mixture_lls = self._logweights + self._loglikelihoods(x)
        if permix:
            return mixture_lls
        else:
            return logsumexp(mixture_lls)

    def generate_logpdfs(self, data, buffer_size=64, permix=False):
        """
        Batch together input frames and do matrix-matrix multiplies
        Yield batched outputs
        """
        buffer = []
        for x in data:
            buffer.append(scipy.concatenate([[1], x, x**2]))
            if len(buffer) == buffer_size:
                for row in scipy.matrix(buffer) * self._matrix:
                    if permix:
                        yield row
                    else:
                        yield logsumexp(row)
                buffer = []
        for row in scipy.matrix(buffer) * self._matrix:
            if permix:
                yield row
            else:
                yield logsumexp(row)

    def _loglikelihoods(self, x):
        return [gconst - scipy.dot((x - mean) ** 2, invvar) / 2.0
                for mean, invvar, gconst in zip(self.means, self._invvars, self._gconsts)]

    def e_step(self, data):
        """
        Perform the E-step of the EM algorithm, determining mixture
        posterior probabilities and producing sufficient statistics.
        """
        counts = scipy.zeros(len(self.weights))
        sums = [scipy.zeros(len(mean)) for mean in self.means]
        sums_squares = [scipy.zeros(len(mean)) for mean in self.means]

        data_ll = 0.0
        if POOL:
            estep_iter = POOL.imap(_gmm_estep, ((self, x) for x in data), CHUNKSIZE)
        else:
            estep_iter = (_gmm_estep((self, x)) for x in data)
        for loglikelihood, posteriors, weighted_x, weighted_x2 in estep_iter:
            data_ll += loglikelihood
            counts += posteriors
            sums = [s + x for s, x in zip(sums, weighted_x)]
            sums_squares = [ss + x2 for ss, x2 in zip(sums_squares, weighted_x2)]

        logging.debug("Data log likelihood: %e" % data_ll)
        return {'counts': counts, 'sums': sums, 'sums_squares': sums_squares,
                'data_ll': data_ll} # Not a sufficient statistic, but useful

    def m_step(self, sufficient_statistics, variance_floor_scale=0.0):
        """
        Perform the M-step of the EM algorithm, providing the ML
        estimate of model parameters given the sufficient statistics.
        Also floor any mixture variances that are below a given factor
        of the global variance.
        """
        counts = sufficient_statistics['counts']
        sums = sufficient_statistics['sums']
        sums_squares = sufficient_statistics['sums_squares']

        # Maximum likelihood estimators
        N = scipy.sum(counts)
        weights = counts / N
        means = []
        variances = []

        # Handle defunct mixtures specially
        for i in range(len(weights)):
            if weights[i] < MINMIX:
                logging.warning("Mixture %d is defunct" % i)
                weights[i] = 0.0
                d = len(sums[i])
                means.append(scipy.zeros(d))
                variances.append(scipy.ones(d))
            else:
                s = sums[i]
                n = counts[i]
                m = s / n
                ss = sums_squares[i]
                means.append(m)
                variances.append(ss / n - m ** 2)

        # Compute global variance and floor mixture variances
        if variance_floor_scale != 0.0:
            if variance_floor_scale > 0.0:
                global_mean = sum(sums) / N
                reference_variance = sum(sums_squares) / N - global_mean ** 2
                logging.debug("Global variance: %s" % reference_variance)
            else: # < 0.0: Use the average within-class (mixture) variance
                variance_floor_scale *= -1.0
                reference_variance = scipy.mean(variances, 0)
                logging.debug("Average mixture variance: %s" % reference_variance)
            variance_floor = reference_variance * variance_floor_scale
            floored_variances = []
            for i in range(len(variances)):
                variance = variances[i]
                for d, v, vf in zip(range(len(variance)), variance, variance_floor):
                    if v < vf and weights[i] != 0.0:
                        logging.warning("Flooring variances[%d][%d]: %f < %f" % (i, d, v, vf))
                floored_variances.append(scipy.array(map(max, zip(variance, variance_floor))))
            variances = floored_variances

        # Invoke property setter
        self.weights = weights
        self.means = means
        self.variances = variances


def _l2norm(args):
    """
    A simple distance between two vectors, specified as a function
    suitable for multiprocessing imports.
    """
    return norm(args[0] - args[1], 2)


def _gmm_logpdf(args):
    """
    A GMM log likelihood function, specified externally to be suitable
    for multiprocessing imports.
    """
    model, x, permix = args
    return model.logpdf(x, permix)


def _gmm_estep(args):
    """
    A GMM helper function for the E-step, specified externally to be
    suitable for multiprocessing imports.
    """
    model, x = args
    x2 = x ** 2
    mixture_likelihoods = scipy.exp(model._logweights + model._loglikelihoods(x)) # TODO: matrix multiply
    observation_likelihood = scipy.sum(mixture_likelihoods)
    posteriors = mixture_likelihoods / observation_likelihood
    loglikelihood = scipy.log(observation_likelihood)
    weighted_x = [p * x for p in posteriors]
    weighted_x2 = [p * x2 for p in posteriors]
    return loglikelihood, posteriors, weighted_x, weighted_x2


def choice(seq, likelihoods=None):
    """
    Alternative to random.choice, allowing elements of non-uniform
    distributions to be sampled with probability proportional to a
    specified sequence of likelihoods.
    """
    # Sample from a uniform distribution
    if likelihoods is None:
        return random.choice(seq)

    # Sum likelihoods over all items in the sequence. Store the
    # cumulative subtotals, which will be in ascending order.
    total = 0.0
    subtotals =  []
    for likelihood in likelihoods:
        total += likelihood
        subtotals.append(total)

    # Binary search to find insertion point
    threshold = random.random() * total
    return seq[bisect.bisect_left(subtotals, threshold)]


def sample(population, k):
    """
    Alternative to random.sample, allowing iterators to be sampled
    (uniformly, without replacement) using an algorithm that does not
    load the entire population into memory.  The selected samples are
    biased (more likely to be ordered rather than a random
    permutation), so shuffle the output if needed.
    """
    samples = []
    for i, x in enumerate(population):
        if i < k:
            # Base case: include all sample if len(list(population)) < =k
            samples.append(x)
        else:
            # Inductive step: provably retains random selection property
            j = random.randint(0, i)
            if j < k:
                samples[j] = x
    return samples


def parsed_data_generator(stream):
    """
    Parse ASCII or binary input data vectors, yielding scipy arrays
    """
    dim = None
    if args.w is None:
        logging.debug('reading text input data')
        for line in stream:
            a = scipy.array(map(float, line.strip().split()))
            if dim is None:
                dim = len(a)
            elif dim != len(a):
                logging.error("input dimensionality changed: %d -> %d" % (dim, len(a)))
            yield a
    else:
        data_fmt = "<%df" % args.w
        logging.debug("reading little-endian float input data (%s)" % data_fmt)
        while True:
            raw_bytes = stream.read(struct.calcsize(data_fmt))
            if not raw_bytes:
                break
            else:
                a = scipy.array(struct.unpack(data_fmt, raw_bytes))
                yield a


if __name__ == '__main__':
    import argparse
    import cPickle
    import struct
    import sys
    import StringIO

    # Parse arguments
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description=USAGE)
    argparser.add_argument('model', nargs='?', default=None,
                           help='model file to be loaded')
    argparser.add_argument('-k', type=int,
                           help='Number of mixtures to initialize with k-means++')
    argparser.add_argument('-f', type=float, default=0.0,
                           help='Set variance floor, scaled from global variance,' +
                           ' or relative to average mixture variance if negative')
    argparser.add_argument('-n', type=int,
                           help='Number of data vectors to use, selected at random')
    argparser.add_argument('-r', type=int, default=0,
                           help='Randomization seed')
    argparser.add_argument('-rt', type=int, default=64,
                           help='Real-time latency (buffered frames)')
    argparser.add_argument('-l', default='obs', choices=['obs', 'mix'],
                           help='Output log likelihoods of observations, or per-mixture (default: obs)')
    argparser.add_argument('-s', action='store_true', default=False,
                           help='Output the sum over all data, not per observation')
    argparser.add_argument('-e', action='store_true', default=False,
                           help='Do the E-step, write sufficient statistics')
    argparser.add_argument('-m', action='store_true', default=False,
                           help='Do the M-step, read statistics, write model')
    argparser.add_argument('-em', action='store_true', default=False,
                           help='Do one iteration of EM, write new model')
    argparser.add_argument('-emi', type=int, metavar='I',
                           help='Do multiple iterations of EM, write new model')
    argparser.add_argument('-w', type=int,
                           help='Raw input of specified width, as little-endian floats')
    argparser.add_argument('-c', type=int, default=multiprocessing.cpu_count(),
                           help='CPUs to utilize (default: full multiprocessing)')
    argparser.add_argument('-b', action='store_true', default=False,
                           help='Write output as binary format (Python cPickle)')
    argparser.add_argument('-o', metavar='FILE',
                           help='Write output to file (default: stdout) and logging to stdout (default: stderr)')
    argparser.add_argument('--loglevel', default='INFO',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                           help='Python logging module (default %(default)s)')
    args = argparser.parse_args()

    # Output may be fully buffered, to allow overwriting input model; logging stream is also modified
    if args.o is None:
        output = sys.stdout
        logging.basicConfig(level=args.loglevel, format="%(module)s:%(levelname)s: %(message)s", stream=sys.stderr)
        _exit = sys.exit
    else:
        output = StringIO.StringIO()
        logging.basicConfig(level=args.loglevel, format="%(module)s:%(levelname)s: %(message)s", stream=sys.stdout)
        def _exit(code):
            open(args.o, 'w').write(output.getvalue())
            sys.exit(code)

    # Determine training mode
    em_modes = int(args.e) + int(args.m) + int(args.em) + int(bool(args.emi))
    if em_modes > 1:
        logging.error('specify at most one option of -e, -m, -em, or -emi')
        _exit(1)

    # Seed randomization for deterministic results
    logging.debug('seeding randomization with seed ' + str(args.r))
    random.seed(args.r)

    # Parse and load data into memory if needed
    data = parsed_data_generator(sys.stdin)
    if args.n:
        logging.debug("selecting %d lines to load in memory" % args.n)
        data = sample(data, args.n)
    elif args.k or args.emi:
        logging.debug('loading all data into memory')
        data = list(data)

    # Initialize multiprocessing pool
    if args.c > 1:
        POOL = multiprocessing.Pool(args.c)

    # Initialize or load model
    if args.k:
        logging.info("initializing GMM with %d means" % args.k)
        model = GMM(None, data, args.k)
    elif not args.m:
        logging.info("loading GMM from %s" % args.model)
        try:
            logging.debug('attempting binary format --> model')
            with open(args.model) as f:
                start = time.time()
                model = cPickle.load(f)
                logging.debug("... loaded in %.2f seconds" % (time.time() - start))
        except cPickle.UnpicklingError:
            logging.debug('loading from JSON format --> model')
            with open(args.model) as f:
                start = time.time()
                model = GMM(json.load(f))
                logging.debug("... loaded in %.2f seconds" % (time.time() - start))
        if not em_modes:
            logging.info('evaluating log likelihood of observations')
            permix = (args.l == 'mix')
            if POOL:
                logpdfs = POOL.imap(_gmm_logpdf, ((model, x, permix) for x in data), CHUNKSIZE)
            else:
                if args.rt <= 1:
                    logpdfs = (model.logpdf(x, permix) for x in data)
                else:
                    logpdfs = model.generate_logpdfs(data, args.rt, permix)
            if args.s:
                result = scipy.sum(logpdfs)
                if permix:
                    if args.b:
                        output.write(''.join(map(lambda x: struct.pack('<f', x), result)))
                    else:
                        output.write(' '.join(map(lambda x: "%e" % x, result)) + '\n')
                else:
                    if args.b:
                        output.write(struct.pack('<f', result))
                    else:
                        output.write("%e\n" % result)
            else:
                for logpdf in logpdfs:
                    if permix:
                        if args.b:
                            output.write(''.join(map(lambda x: struct.pack('<f', x), logpdf)))
                        else:
                            output.write(' '.join(map(lambda x: "%e" % x, logpdf)) + '\n')
                    else:
                        if args.b:
                            output.write(struct.pack('<f', logpdf))
                        else:
                            output.write("%e\n" % logpdf)
            _exit(0)

    # EM training modes
    if args.e:
        logging.info('performing E-step')
        sufficient_stats = model.e_step(data)
        logging.info('writing sufficient statistics')
        if args.b:
            cPickle.dump(sufficient_stats, output, protocol=-1)
        else:
            sufficient_stats['counts'] = list(sufficient_stats['counts'])
            sufficient_stats['sums'] = map(list, sufficient_stats['sums'])
            sufficient_stats['sums_squares'] = map(list, sufficient_stats['sums_squares'])
            json.dump(sufficient_stats, output)
        _exit(0)
    elif args.m:
        accumulated_stats = None
        for line in sys.stdin:
            stats_file = line.strip()
            logging.info("accumulating sufficient statistics from %s" % stats_file)
            try:
                logging.debug('attempting binary format --> sufficient stats')
                with open(stats_file) as f:
                    sufficient_stats = cPickle.load(f)
            except cPickle.UnpicklingError:
                logging.debug('loading from JSON format --> sufficient stats')
                with open(stats_file) as f:
                    sufficient_stats = json.load(f)
                sufficient_stats['counts'] = scipy.array(sufficient_stats['counts'])
                sufficient_stats['sums'] = map(scipy.array, sufficient_stats['sums'])
                sufficient_stats['sums_squares'] = map(scipy.array, sufficient_stats['sums_squares'])
            if accumulated_stats is None:
                accumulated_stats = sufficient_stats
            else:
                accumulated_stats['counts'] += sufficient_stats['counts']
                accumulated_stats['sums'] = [s1 + s2 for s1, s2 in zip(sufficient_stats['sums'], accumulated_stats['sums'])]
                accumulated_stats['sums_squares'] = [ss1 + ss2 for ss1, ss2 in zip(sufficient_stats['sums_squares'], accumulated_stats['sums_squares'])]
                accumulated_stats['data_ll'] += sufficient_stats['data_ll']
        logging.info("Data log likelihood: %e" % accumulated_stats['data_ll'])
        logging.info('performing M-step')
        model = GMM()
        model.m_step(accumulated_stats, args.f)
    elif args.em:
        logging.info('performing E-step')
        sufficient_stats = model.e_step(data)
        logging.info('performing M-step')
        model.m_step(sufficient_stats, args.f)
    elif args.emi:
        for i in range(1, args.emi + 1):
            logging.info("performing EM iteration %d" % i)
            sufficient_stats = model.e_step(data)
            model.m_step(sufficient_stats, args.f)

    # Write model in text or binary format
    if args.b:
        logging.info('writing out binary-formatted model')
        cPickle.dump(model, output, protocol=-1)
    else:
        logging.info('writing out JSON-formatted model')
        json.dump(model.parameters, output)

    _exit(0)
