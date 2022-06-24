from logistic_model.config.core import config
from logistic_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(variables=config.model_config.cabin)
    assert sample_input_data["cabin"].iat[50] == "C31"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[50] == "C"
