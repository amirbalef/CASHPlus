"""A simple random search optimizer.

This optimizer will sample from the space provided and return the results
without doing anything with them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, overload
from typing_extensions import override

from amltk.randomness import as_int, as_rng
from amltk.types import Space

from datetime import datetime

from amltk.store.paths.path_bucket import PathBucket

from amltk.optimization import Metric, Optimizer, Trial


if TYPE_CHECKING:
    from amltk.types import Config, Seed


class RandomSearch(Optimizer[None]):
    """A random search optimizer."""

    def __init__(
        self,
        metrics: Metric | Sequence[Metric],
        space: Space,  # type: ignore
        bucket: PathBucket | None = None,
        seed: Seed | None = None,
        duplicates: bool = False,
        max_sample_attempts: int = 50,
        initial_configs: list | None = None,
        limit_to_configs: list | None = None,
    ):
        """Initialize the optimizer.

        Args:
            space: The space to sample from.
            seed: The seed to use for the sampler.
            duplicates: Whether to allow duplicate configurations.
            max_sample_attempts: The maximum number of attempts to sample a
                unique configuration. If this number is exceeded, an
                `ExhaustedError` will be raised. This parameter has no
                effect when `duplicates=True`.
        """
        self.space = space
        self.trial_count = 0
        metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        self.metrics = metrics
        self.seed = as_int(seed)
        self.space.seed(seed)
        self.as_rng = as_rng(self.seed)
        self.max_sample_attempts = max_sample_attempts
        self.bucket = (
            bucket
            if bucket is not None
            else PathBucket(f"{self.__class__.__name__}-{datetime.now().isoformat()}")
        )

        # We store any configs we've seen to prevent duplicates
        self._configs_seen: list[Config] | None = [] if not duplicates else None
        self.duplicates = duplicates
        self.initial_configs = [] if initial_configs is None else initial_configs
        
        if limit_to_configs is not None:
            self.limit_to_configs = []
            for config in limit_to_configs: 
                if(config not in self.initial_configs):
                    self.limit_to_configs.append(config)
            self.as_rng.shuffle(self.limit_to_configs)
        else:
            self.limit_to_configs = None

    @overload
    def ask(self, n: int) -> Iterable[Trial[None]]: ...

    @overload
    def ask(self, n: None = None) -> Trial[None]: ...

    @override
    def ask(self) -> Trial[None]:
        """Sample from the space.

        Raises:
            ExhaustedError: If the sampler is exhausted of unique configs.
                Only possible to raise if `duplicates=False` (default).
        """
        name = f"random-{self.trial_count}"
        if self.trial_count < len(self.initial_configs):
            config = self.initial_configs[self.trial_count]
        else:
            if self.limit_to_configs is not None:
             config =  self.limit_to_configs[self.trial_count]
            else:
                config = self.space.sample_configuration()
                try_number = 0
                while config in self._configs_seen and self.duplicates is False:
                    config = self.space.sample_configuration()
                    if try_number >= self.max_sample_attempts:
                        break
                    try_number += 1

        trial = Trial.create(
            name=name,
            metrics=self.metrics,
            config=dict(config),
            info=None,
            seed=self.seed,
            bucket=self.bucket,
        )
        return trial


    @override
    def tell(self, report: Trial.Report[None]) -> None:
        """Tell the optimizer about the result of a trial.
        Args:
            report: The report of the trial.
        """
        config = report.trial.config
        if self._configs_seen is not None:
            if config not in self._configs_seen:
                self._configs_seen.append(config)
        self.trial_count = self.trial_count + 1

    @override
    @classmethod
    def preferred_parser(cls) -> Literal["configspace"]:
        """The preferred parser for this optimizer."""
        return "configspace"