from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor, XGBClassifier
import numpy as np # type: ignore


def train_classification_model(X_train, y_train):
    model = XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        use_label_encoder=False)
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average="weighted")
  recall = recall_score(y_test, y_pred, average="weighted")
  f1 = f1_score(y_test, y_pred, average="weighted")
  return accuracy, precision, recall, f1


def tune_classification_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        XGBClassifier(),
        param_grid,
        cv=tscv,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def train_regression_model(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror',
        tree_method='gpu_hist',  
        predictor='gpu_predictor',
        gpu_id=0
)
    model.fit(X_train, y_train)
    return model

def tune_regression_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        XGBRegressor(objective='reg:squarederror'),
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2
