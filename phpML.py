import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
sns.set()
# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import cross_val_score
import xgboost as xgb
# tune
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow.xgboost
import mlflow.sklearn
import mlflow
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
import shap
import logging
import sys
import warnings


## data loading
df = pd.read_csv('.data/php_data_all.csv', index_col=0)
# selecting data according to temperature range
# NOTE: Data selected between [300, 355]
df = df[(df['Te[K]'] > 300) & (df['Te[K]'] < 355)]


## data split
x = df[['Te[K]', 'dT[K]', 'P[bar]', 'Q[W]', 'Fluid', 'FR']]
y = df['TR[K/W]']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


## data pipeline preparation
numeric_features = ['Te[K]', 'dT[K]', 'P[bar]', 'Q[W]','FR']
categorical_features = ['Fluid']

numeric_transformer = make_pipeline(StandardScaler())
categorical_tranformer = make_pipeline(OneHotEncoder(sparse_output=False))

preprocessor = ColumnTransformer(
    transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_tranformer, categorical_features)
    ])


## a function to evaluate a trained ML model
def evaluate(y_test, y_pred, k=6):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = y_test.shape[0]
    k = k
    r2_adj = 1 - (((1-r2)*(n-1)) / (n-k-1))
    return rmse, mae, r2, r2_adj


## objective of a ML model training
# mlflow manual logging of metrics and model
mlflow.set_experiment('CLPHP_TR_Prediction')
## Model dictionary
models = [
    {
        'name': 'XGBoost',
        'model': xgb.XGBRegressor,
        'search_space': {
            'eta': hp.uniform('eta', 0.1, 1),
            'max_depth': hp.randint('max_depth', 2, 5)
        }
    },
    {
        'name': 'Random Forest',
        'model': RandomForestRegressor,
        'search_space': {
            'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
            'max_depth': hp.randint('max_depth', 2, 20)
        }
    },
    {
        'name': 'Linear Regression',
        'model': LinearRegression,
        'search_space': {}  # No hyperparameters for Linear Regression
    },
    {
        'name': 'Elastic Net',
        'model': ElasticNet,
        'search_space': {
            'alpha': hp.uniform('alpha', 0.1, 1),
            'l1_ratio': hp.uniform('l1_ratio', 0.1, 0.9)
        }
    }
]

# Loop through the list of models and train/tune each model
for model_info in models:
    model_name = model_info['name']
    model_class = model_info['model']
    search_space = model_info['search_space']

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag('model', model_name)
            mlflow.log_params(params=params)

            # data pipeline
            model = model_class(**params)
            data_pipeline = Pipeline(steps=[('Preprocessing', preprocessor), 
                                            (model_name, model)])
            # cross validation
            cv_scores = cross_val_score(data_pipeline, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
            avg_cv_rmse = np.sqrt(-cv_scores.mean())
            mlflow.log_metric('avg_cv_rmse', avg_cv_rmse)

            # training
            data_pipeline.fit(x_train, y_train)

            # train and test prediction
            pred_train = data_pipeline.predict(x_train)
            pred = data_pipeline.predict(x_test)
            rmse_train, mae_train, r2_train, r2_adj_train = evaluate(y_test=y_train, y_pred=pred_train)
            rmse, mae, r2, r2_adj = evaluate(y_test=y_test, y_pred=pred)
            signature = infer_signature(x_train, pred)

            # # Log feature importance for tree-based models
            # if hasattr(model, 'feature_importances_'):
            #     mlflow.log_param('feature_importances', model.feature_importances_)

            # # Log coefficients for linear models
            # if hasattr(model, 'coef_'):
            #     mlflow.log_param('coefficients', model.coef_)

            # test
            mlflow.log_metric('rmse_test', rmse)
            mlflow.log_metric('mae_test', mae)
            mlflow.log_metric('r2_test', r2)
            mlflow.log_metric('r2_adj_test', r2_adj)
            # train
            mlflow.log_metric('rmse_train', rmse_train)
            mlflow.log_metric('mae_train', mae_train)
            mlflow.log_metric('r2_train', r2_train)
            mlflow.log_metric('r2_adj_train', r2_adj_train)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name.lower()}-model",
                signature=signature,
                registered_model_name=f"{model_name.lower()}-regressor",
            )

            # Log SHAP values for model explanation
            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_test, feature_names=x_test.columns)

    # Hyperparameter tuning
    best_results = fmin(fn=objective, 
                        space=search_space, 
                        algo=tpe.suggest, 
                        max_evals=10, 
                        trials=Trials())















# def objective(params):
#     with mlflow.start_run():
#         mlflow.set_tag('model', 'xgb')
#         mlflow.log_params(params=params)

#         model_xgb = xgb.XGBRegressor(**params)
#         data_pipeline_xgb = Pipeline(steps=[('Preprocessing', preprocessor),
#                                 ('xgb_model', model_xgb)])
        
#         data_pipeline_xgb.fit(x_train, y_train)

#         pred = data_pipeline_xgb.predict(x_test)
        
#         rmse, ame, r2, r2_adj = evaluate(y_test=y_test, y_pred=pred) # NEED TO CHECK WITH ONE TARGET VARIABLE
        
#         signature = infer_signature(x_train, pred)

#         mlflow.log_metric('rmse', rmse)
#         mlflow.log_metric('ame', ame)
#         mlflow.log_metric('r2', r2)
#         mlflow.log_metric('r2_adj', r2_adj)
#         mlflow.sklearn.log_model(
#         sk_model=model_xgb,
#         artifact_path="sklearn-model",
#         signature=signature,
#         registered_model_name="xgb-regressor",
#     )
#     return {'loss': rmse, 'status': STATUS_OK}


# # hyper params space
# search_space_xgb = {'eta': hp.uniform('eta', 0.1,1), 
#                 'max_depth': hp.randint('max_depth', 2,5)}

# # hyperopt - hyper param tunning
# best_results_xgb = fmin(fn=objective,
#                     space=search_space_xgb,
#                     algo=tpe.suggest,
#                     max_evals=15,
#                     trials=Trials())