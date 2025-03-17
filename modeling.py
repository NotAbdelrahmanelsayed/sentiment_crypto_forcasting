from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
import xgboost as xgb

# Common GPU configuration for all models
GPU_CONFIG = {
    'tree_method': 'hist',
    'device': 'cuda:0',
    'enable_categorical': False,
    'verbosity': 0
}

def _prepare_data(X, y=None):
    """Convert data to GPU-compatible format"""
    X = np.asarray(X, dtype=np.float32, order='C')
    if y is not None:
        y = np.asarray(y, dtype=np.float32 if isinstance(y, pd.Series) else 'int32')
    return X, y

# Classification Models
def train_classification_model(X_train, y_train):
    X, y = _prepare_data(X_train, y_train)
    model = XGBClassifier(**GPU_CONFIG, use_label_encoder=False)
    model.fit(X, y)
    return model

def evaluate_classification_model(model, X_test, y_test):
    X, y = _prepare_data(X_test, y_test)
    y_pred = model.predict(X)
    return (
        accuracy_score(y, y_pred),
        precision_score(y, y_pred, average="weighted"),
        recall_score(y, y_pred, average="weighted"),
        f1_score(y, y_pred, average="weighted")
    )

def tune_classification_model(X_train, y_train):
    X, y = _prepare_data(X_train, y_train)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        XGBClassifier(**GPU_CONFIG, use_label_encoder=False),
        param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='accuracy'
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

# Regression Models
def train_regression_model(X_train, y_train):
    X, y = _prepare_data(X_train, y_train)
    model = XGBRegressor(**GPU_CONFIG, objective='reg:squarederror')
    model.fit(X, y)
    return model

def tune_regression_model(X_train, y_train):
    X, y = _prepare_data(X_train, y_train)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        XGBRegressor(**GPU_CONFIG, objective='reg:squarederror'),
        param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_regression_model(model, X_test, y_test):
    X, y = _prepare_data(X_test, y_test)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse, np.sqrt(mse), r2_score(y, y_pred)