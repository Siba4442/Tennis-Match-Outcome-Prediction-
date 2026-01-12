from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    DATA_RAW: str = "data/raw"
    DATA_INTERIM: str = "data/interim"
    DATA_PROCESSED: str = "data/processed"
    OUTPUTS_FIG: str = "outputs/figures"
    OUTPUTS_MODELS: str = "outputs/models"


@dataclass(frozen=True)
class SplitConfig:
    # Train <= this date, test > this date
    SPLIT_DATE: int = 20221231


@dataclass(frozen=True)
class RandomConfig:
    SEED: int = 42
