"""File path utilities for GTMP."""
from pathlib import Path

import yaml

import gtmp


def get_root_path() -> Path:
    """Get the project root directory."""
    return Path(gtmp.__path__[0]).resolve().parent


def get_data_path() -> Path:
    """Get the data directory."""
    return get_root_path() / "data"


def get_configs_path() -> Path:
    """Get the configs directory."""
    return get_root_path() / "configs"


def get_data_config_path() -> Path:
    """Get the data/configs directory."""
    return get_data_path() / "configs"


def load_yaml(filename: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    filename : str or Path
        Path to YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(filename, "r") as stream:
        return yaml.safe_load(stream)