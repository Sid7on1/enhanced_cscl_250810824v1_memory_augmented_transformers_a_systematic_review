import logging
import os
import sys
import threading
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

# Constants
LOG_LEVEL = logging.INFO
CONFIG_FILE = 'config.json'
DATA_DIR = 'data'
MODELS_DIR = 'models'

# Logging setup
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentMode(Enum):
    TRAIN = 1
    EVAL = 2
    TEST = 3

@dataclass
class EnvironmentConfig:
    mode: EnvironmentMode
    data_dir: str
    models_dir: str
    batch_size: int
    num_workers: int

class EnvironmentException(Exception):
    pass

class Environment:
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.models_dir = config.models_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.lock = threading.Lock()

    def setup(self):
        with self.lock:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)

    def teardown(self):
        with self.lock:
            pass

    def get_data_dir(self) -> str:
        return self.data_dir

    def get_models_dir(self) -> str:
        return self.models_dir

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_num_workers(self) -> int:
        return self.num_workers

    def get_mode(self) -> EnvironmentMode:
        return self.mode

    def set_mode(self, mode: EnvironmentMode):
        with self.lock:
            self.mode = mode

    def load_config(self, config_file: str) -> EnvironmentConfig:
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return EnvironmentConfig(
                    mode=EnvironmentMode[config_data['mode']],
                    data_dir=config_data['data_dir'],
                    models_dir=config_data['models_dir'],
                    batch_size=config_data['batch_size'],
                    num_workers=config_data['num_workers']
                )
        except Exception as e:
            logger.error(f'Failed to load config: {str(e)}')
            raise EnvironmentException('Failed to load config')

    def save_config(self, config: EnvironmentConfig, config_file: str):
        try:
            import json
            config_data = {
                'mode': config.mode.name,
                'data_dir': config.data_dir,
                'models_dir': config.models_dir,
                'batch_size': config.batch_size,
                'num_workers': config.num_workers
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
        except Exception as e:
            logger.error(f'Failed to save config: {str(e)}')
            raise EnvironmentException('Failed to save config')

class DataProvider(ABC):
    @abstractmethod
    def get_data(self) -> List[Dict]:
        pass

class DataLoader:
    def __init__(self, data_provider: DataProvider, batch_size: int, num_workers: int):
        self.data_provider = data_provider
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_data(self) -> List[Dict]:
        data = self.data_provider.get_data()
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        return batches

class ModelLoader:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def load_model(self, model_name: str):
        try:
            import torch
            model_path = os.path.join(self.models_dir, model_name)
            model = torch.load(model_path)
            return model
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')
            raise EnvironmentException('Failed to load model')

class ModelSaver:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def save_model(self, model, model_name: str):
        try:
            import torch
            model_path = os.path.join(self.models_dir, model_name)
            torch.save(model, model_path)
        except Exception as e:
            logger.error(f'Failed to save model: {str(e)}')
            raise EnvironmentException('Failed to save model')

@contextmanager
def environment_context(environment: Environment):
    try:
        environment.setup()
        yield environment
    finally:
        environment.teardown()

def main():
    config = EnvironmentConfig(
        mode=EnvironmentMode.TRAIN,
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        batch_size=32,
        num_workers=4
    )
    environment = Environment(config)
    with environment_context(environment):
        data_loader = DataLoader(DataProvider(), environment.get_batch_size(), environment.get_num_workers())
        model_loader = ModelLoader(environment.get_models_dir())
        model_saver = ModelSaver(environment.get_models_dir())
        # Use the environment and its components

if __name__ == '__main__':
    main()