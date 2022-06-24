# for encoding categorical variables
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# for imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from logistic_model.config.core import config
from logistic_model.processing import features as pp

# set up the pipeline
titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string 'missing'
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_variables,
            ),
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_variables),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_variables,
            ),
        ),
        # extract letter
        (
            "extract_first_letter",
            pp.ExtractLetterTransformer(variables=config.model_config.cabin),
        ),
        # == CATEGORICAL ENCODING
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05,
                n_categories=1,
                variables=config.model_config.categorical_variables,
            ),
        ),
        (
            "one_hot_encoder",
            OneHotEncoder(
                variables=config.model_config.categorical_variables, drop_last=True
            ),
        ),
        # scale using standardization
        ("scaler", StandardScaler()),
        # logistic regression (use C=0.0005 and random_state=0)
        (
            "Logit",
            LogisticRegression(
                random_state=config.model_config.random_state, C=config.model_config.C
            ),
        ),
    ]
)
