# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.zero_order_momentum.zero_order_momentum -- TODO 
===========================================================

.. module:: zero_order_momentum
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
import numpy

from sklearn.neighbors import KNeighborsRegressor

from orion.algo.base import BaseAlgorithm
from orion.algo.space import (pack_point, unpack_point)


class ZeroOrderMomentumOptimizer(BaseAlgorithm):
    """
    TODO: Class docstring
    """

    requires = 'linear'

    def __init__(self, space, momentum=0.7, patience=100, sigma=1, seed=None):
        """
        TODO: init docstring
        """
        super(ZeroOrderMomentumOptimizer, self).__init__(
            space, momentum=momentum, patience=patience, sigma=sigma)
        self.best_score = float('inf')
        self.rng = numpy.random.RandomState(seed)
        self.best_params = numpy.array(unpack_point(space.sample()[0], self.space))  # TODO: Default values
        self.mom = numpy.zeros(len(self.best_params))
        self.worst = float('inf')
        self.params = []
        self.results = []
        self.model = None

        # NOTE: Drop static

    def suggest(self, name=1):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

        """
        if self.model is None:
            return [pack_point(self.best_params, self.space)]
        rejected = True
        attemps = 0
        while rejected and attemps < self.patience:
            sample = (self.best_params + self.mom +
                      self.rng.normal(0, self.sigma, size=len(self.best_params)))
            # TODO: Use a smaller version of the space
            # TODO: Sample may exceed the global space, clamp it if it is the case
            estimate = self.model.predict(sample[None, :])# [0, 0]
            attemps += 1
            p_of_reject = ((numpy.log(estimate) - numpy.log(self.best_score)) /
                           (numpy.log(self.worst) - numpy.log(self.best_score)))
            if p_of_reject > self.rng.uniform():
                rejected = False

        point = pack_point(sample, self.space)
        # Clamp points
        if point not in self.space:
            for i, (space_dim, dim) in enumerate(zip(self.space.values(), point)):
                point[i] = numpy.minimum(numpy.maximum(dim, space_dim.interval()[0]),
                                         space_dim.interval()[1] * 0.9999999999)
                if numpy.prod(space_dim.shape) > 1:
                    point[i] = list(point[i])

        assert point in self.space, point

        return [point]

    def update_stats(self, points, results):
        """
        TODO
        """
        for point, result in zip(points, results):
            params = numpy.array(unpack_point(point, self.space))
            result = result['objective']
            if result < self.best_score:
                self.mom = self.momentum * (params - self.best_params) + (1 - self.momentum) * self.mom
                self.best_score = result
                self.best_params = params
            else:
                self.mom = self.momentum * (self.best_params - params) + (1 - self.momentum) * self.mom

            self.params.append(params)
            self.results.append(result)

        print(self.mom)

        if len(self.results) > 1:
            self.worst = numpy.median(self.results)

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        TODO: document how observe work for this algo

        """
        # self.params += [numpy.array(unpack_point(point, self.space)) for point in points]
        # self.results += [result['objective'] for result in results]

        self.update_stats(points, results)

        self.model = KNeighborsRegressor(n_neighbors=1)
        self.model.fit(numpy.array(self.params), numpy.array(self.results))
