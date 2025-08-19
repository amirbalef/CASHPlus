from collections.abc import Iterable, Sequence
from typing import Literal, overload
from typing_extensions import override

from amltk.randomness import as_int, as_rng
from amltk.types import Space

from datetime import datetime

from amltk.store.paths.path_bucket import PathBucket

from amltk.optimization import Metric, Optimizer, Trial

from amltk.types import Seed


class DefaultConfiguration(Optimizer[None]):
    def __init__(
        self,
        metrics: Metric | Sequence[Metric],
        space: Space,  # type: ignore
        bucket: PathBucket | None = None,
        seed: Seed | None = None,
        name: str = "default configuration",
    ):
        self.space = space
        self.name = name
        self.trial_count = 0
        metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        self.metrics = metrics
        self.seed = as_int(seed)
        self.space.seed(seed)
        self.as_rng = as_rng(self.seed)
        self.bucket = (
            bucket
            if bucket is not None
            else PathBucket(f"{self.__class__.__name__}-{datetime.now().isoformat()}")
        )

    @overload
    def ask(self, n: int) -> Iterable[Trial[None]]: ...

    @overload
    def ask(self, n: None = None) -> Trial[None]: ...

    @override
    def ask(self) -> Trial[None]:
        name = self.name + f"-{self.trial_count}"
        config = self.space.get_default_configuration()
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
        self.trial_count += 1

    @override
    @classmethod
    def preferred_parser(cls) -> Literal["configspace"]:
        """The preferred parser for this optimizer."""
        return "configspace"