from __future__ import print_function
import sys
sys.path.append('./BayesianOptimization')
from bayes_opt.bayesian_optimization import unique_rows, acq_max, BayesianOptimization
from bayes_opt.helpers import AcquisitionFunction, PrintInfo
from datetime import datetime
from sklearn.gaussian_process import GaussianProcess as GP
import numpy


class ModBayesianOptimization(BayesianOptimization):
    f_args = {}
    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a combination of
        points passed by the user, and randomly sampled ones.

        :param init_points: Number of random points to probe.
        """

        # Generate random points
        l = [numpy.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        # Concatenate new random points to possible existing points from self.explore method.
        self.init_points += list(map(list, zip(*l)))

        # Create empty list to store the new values of the function
        y_init = []

        # Evaluate target function at all initialization points (random + explore)
        for x in self.init_points:

            if self.verbose:
                print('Initializing function at point: ', dict(zip(self.keys, x)), end='')
            arg_dic = dict(zip(self.keys, x))
            arg_dic.update(self.f_args)
            y_init.append(self.f(**arg_dic))
            if self.verbose:
                print(' | result: %f' % y_init[-1])

        # Append any other points passed by the self.initialize method (these also have
        # a corresponding target value passed by the user).
        self.init_points += self.x_init

        # Append the target value of self.initialize method.
        y_init += self.y_init

        # Turn it into numpy array and store.
        self.X = numpy.asarray(self.init_points)
        self.Y = numpy.asarray(y_init)

        # Updates the flag
        self.initialized = True

    def maximize(self, init_points=5, restarts=50, n_iter=25, acq='ei', **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points: Number of randomly chosen points to sample the target function before fitting the gp.

        :param restarts: The number of times minimation if to be repeated. Larger number of restarts
                         improves the chances of finding the true maxima.

        :param n_iter: Total number of times the process is to reapeated. Note that currently this methods does not have
                       stopping criteria (due to a number of reasons), therefore the total number of points to be sampled
                       must be specified.

        :param acq: Acquisition function to be used, defaults to Expected Improvement.

        :param gp_params: Parameters to be passed to the Scikit-learn Gaussian Process object

        Returns
        -------
        :return: Nothing
        """
        # Start a timer
        total_time = datetime.now()

        # Create instance of printer object
        printI = PrintInfo(self.verbose)

        # Set acquisition function
        AC = AcquisitionFunction()
        ac_types = {'ei': AC.EI, 'pi': AC.PoI, 'ucb': AC.UCB}
        ac = ac_types[acq]

        # Initialize x, y and find current ymax
        if not self.initialized:
            self.init(init_points)

        ymax = self.Y.max()

        # ------------------------------ // ------------------------------ // ------------------------------ #
        # Fitting the gaussian process.
        # Since scipy 0.16 passing lower and upper bound to theta seems to be
        # broken. However, there is a lot of development going on around GP
        # is scikit-learn. So I'll pick the easy route here and simple specify
        # only theta0.
        gp = GP(theta0=numpy.random.uniform(0.001, 0.05, self.dim),
                random_start=25)

        gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        gp.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac, gp, ymax, restarts, self.bounds)

        # ------------------------------ // ------------------------------ // ------------------------------ #
        # Iterative process of searching for the maximum. At each round the most recent x and y values
        # probed are added to the X and Y arrays used to train the Gaussian Process. Next the maximum
        # known value of the target function is found and passed to the acq_max function. The arg_max
        # of the acquisition function is found and this will be the next probed value of the tharget
        # function in the next round.
        for i in range(n_iter):
            op_start = datetime.now()

            # Append most recently generated values to X and Y arrays
            self.X = numpy.concatenate((self.X, x_max.reshape((1, self.dim))), axis=0)
            arg_dic = dict(zip(self.keys, x_max))
            arg_dic.update(self.f_args)
            self.Y = numpy.append(self.Y, self.f(**arg_dic))

            # Updating the GP.
            ur = unique_rows(self.X)
            gp.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if self.Y[-1] > ymax:
                ymax = self.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac, gp, ymax, restarts, self.bounds)

            # Print stuff
            printI.print_info(op_start, i, x_max, ymax, self.X, self.Y, self.keys)

        # ------------------------------ // ------------------------------ // ------------------------------ #
        # Output dictionary
        self.res = {}
        self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, self.X[self.Y.argmax()]))}
        self.res['all'] = {'values': [], 'params': []}

        # Fill values
        for t, p in zip(self.Y, self.X):
            self.res['all']['values'].append(t)
            self.res['all']['params'].append(dict(zip(self.keys, p)))

        # Print a final report if verbose active.
        if self.verbose:
            tmin, tsec = divmod((datetime.now() - total_time).total_seconds(), 60)
            print('Optimization finished with maximum: %8f, at position: %8s.' % (self.res['max']['max_val'],\
                                                                                  self.res['max']['max_params']))
            print('Time taken: %i minutes and %s seconds.' % (tmin, tsec))

    def add_f_args(self, key, value):
        self.f_args[key] = value
