import numpy as np

from logistic_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
