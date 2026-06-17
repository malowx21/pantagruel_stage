import yaml
from pathlib import Path


def load_config(config_path) :
    """
    Load configuration file YAML and returns a dictionnary 
    
    Args:
        config_path: path to the config file

    Returns:
        dictionnary containing all the configuration 
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found : {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
