import pytest

from logistic_model.config.core import config
from logistic_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    dataset = load_dataset(file_name=config.app_config.test_data_file)
    new_columns = {key: key.lower() for key in dataset.columns}
    dataset = dataset.rename(columns=new_columns)
    return dataset
