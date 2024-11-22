import gtmp
from pathlib import Path
import yaml


# get paths
def get_root_path():
    path = Path(gtmp.__path__[0]).resolve() / '..'
    return path


def get_data_path():
    path = get_root_path() / 'data'
    return path


def get_configs_path():
    path = get_root_path() / 'configs'
    return path


def get_data_config_path():
    path = get_data_path() / 'configs'
    return path


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
            return config
        except yaml.YAMLError as exc:
            print(exc)