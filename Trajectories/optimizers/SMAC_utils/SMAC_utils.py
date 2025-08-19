from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
from typing_extensions import override

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


from smac.initial_design import AbstractInitialDesign

import copy

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
)

from collections import OrderedDict

from typing import Callable, Iterator

from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.random_design.modulus_design import ModulusRandomDesign

from smac.acquisition import maximizer

logger = logging.getLogger(__name__)


class FixedSetRandomInitialDesign(AbstractInitialDesign):
    """Initial design that evaluates random configurations."""

    def __init__(self, limit_to_configs: list, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.limit_to_configs = limit_to_configs

    def _select_configurations(self) -> list[Configuration]:
        configs_indx = self._rng.randint(
            0, len(self.limit_to_configs), size=self._n_configs
        )
        configs = [self.limit_to_configs[i] for i in configs_indx]
        for config in configs:
            config.origin = "Initial Design: FixedSet Random"
        return configs


def select_configurations(initial_class) -> list[Configuration]:
    """Selects the initial configurations. Internally, `_select_configurations` is called,
    which has to be implemented by the child class.

    Returns
    -------
    configs : list[Configuration]
        Configurations from the child class.
    """

    configs: list[Configuration] = []

    # Adding additional configs
    configs += initial_class._additional_configs

    for config in configs:
        if config.origin is None:
            config.origin = "Initial Design:Additional Configs"

    if initial_class._n_configs == 0:
        logger.info("No initial configurations are used.")
    else:
        configs += initial_class._select_configurations()

    for config in configs:
        if config.origin is None:
            config.origin = "Initial Design"

    # Removing duplicates
    # (Reference: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)
    configs = list(OrderedDict.fromkeys(configs))
    logger.info(
        f"Using {len(configs) - len(initial_class._additional_configs)} initial design configurations "
        f"and {len(initial_class._additional_configs)} additional configurations."
    )

    return configs



class ChallengerList(Iterator):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    configspace : ConfigurationSpace
    challenger_callback : Callable
        Callback function which returns a list of challengers (without interleaved random configurations, must a be a
        python closure.
    random_design : AbstractRandomDesign | None, defaults to ModulusRandomDesign(modulus=2.0)
        Which random design should be used.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        challenger_callback: Callable,
        configurations: list,
        random_design: AbstractRandomDesign | None = ModulusRandomDesign(modulus=2.0),
        rng=None,
        previous_configs: list = None,
    ):
        self._challengers_callback = challenger_callback
        self._challengers: list[Configuration] | None = None
        self._configspace = configspace
        self._index = 0
        self._iteration = (
            1  # 1-based to prevent from starting with a random configuration
        )
        self._random_design = random_design
        self._configurations = configurations
        self._rng = rng
        self._previous_configs = previous_configs

    def __next__(self) -> Configuration:
        # If we already returned the required number of challengers
        if self._challengers is not None and self._index == len(self._challengers):
            raise StopIteration
        # If we do not want to have random configs, we just yield the next challenger
        elif self._random_design is None:
            if self._challengers is None:
                self._challengers = self._challengers_callback()

            config = self._challengers[self._index]
            self._index += 1

            return config
        # If we want to interleave challengers with random configs, sample one
        else:
            if self._random_design.check(self._iteration):
                configurations = [
                    config
                    for config in self._configurations
                    if config not in self._previous_configs
                ]
                config = configurations[self._rng.randint(0, len(configurations))]
                # config = self._configspace.sample_configuration()
                config.origin = "FixedSet Random Search"
            else:
                if self._challengers is None:
                    self._challengers = self._challengers_callback()

                config = self._challengers[self._index]
                self._index += 1
            self._iteration += 1

            return config

    def __len__(self) -> int:
        if self._challengers is None:
            self._challengers = self._challengers_callback()

        return len(self._challengers) - self._index



class FixedSet(maximizer.abstract_acquisition_maximizer.AbstractAcquisitionMaximizer):
    def __init__(
        self,
        configurations: list[Configuration],
        acquisition_function: maximizer.abstract_acquisition_maximizer.AbstractAcquisitionFunction,
        configspace: ConfigurationSpace,
        challengers: int = 5000,
        seed: int = 0,
    ):
        """Maximize the acquisition function over a finite list of configurations.
        Parameters
        ----------
        configurations : list[~smac._configspace.Configuration]
            Candidate configurations
        acquisition_function : ~smac.acquisition.AbstractAcquisitionFunction

        configspace : ~smac._configspace.ConfigurationSpace

        rng : np.random.RandomState or int, optional
        """
        super().__init__(
            acquisition_function=acquisition_function,
            configspace=configspace,
            challengers=challengers,
            seed=seed,
        )
        self.configurations = configurations

    @override
    def maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int | None = None,
        random_design: AbstractRandomDesign | None = None,
    ) -> Iterator[Configuration]:
        """Maximize acquisition function using `_maximize`, implemented by a subclass.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previous evaluated configurations.
        n_points: int, defaults to None
            Number of points to be sampled. If `n_points` is not specified,
            `self._challengers` is used.
        random_design: AbstractRandomDesign, defaults to None
            Part of the returned ChallengerList such that we can interleave random configurations
            by a scheme defined by the random design. The method `random_design.next_iteration()`
            is called at the end of this function.

        Returns
        -------
        challengers : Iterator[Configuration]
            An iterable consisting of configurations.
        """

        if n_points is None:
            n_points = self._challengers

        def next_configs_by_acquisition_value() -> list[Configuration]:
            assert n_points is not None
            # since maximize returns a tuple of acquisition value and configuration,
            # and we only need the configuration, we return the second element of the tuple
            # for each element in the list
            return [t[1] for t in self._maximize(previous_configs, n_points)]

        challengers = ChallengerList(
            self._configspace,
            next_configs_by_acquisition_value,
            self.configurations,
            random_design,
            self._rng,
            previous_configs,
        )

        if random_design is not None:
            random_design.next_iteration()

        return challengers

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        configurations = [
            copy.deepcopy(config)
            for config in self.configurations
            if config not in previous_configs
        ]
        for config in configurations:
            config.origin = "Fixed Set"
        res = self._sort_by_acquisition_value(configurations)
        return res