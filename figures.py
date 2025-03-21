from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns # type: ignore
import numpy as np # type: ignore
from modeling import evaluate_classification_model
import shap # type: ignore
from statsmodels.graphics.tsaplots import plot_acf #type: ignore

def visualize_classification_model(model, X_test, y_test, coin_name):
  y_pred = model.predict(X_test)
  accuracy, precision, recall, f1 = evaluate_classification_model(model, X_test, y_test)

  cm = confusion_matrix(y_test, y_pred)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.set_title(f"Confusion Matrix for {coin_name}")
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  print(f"Classification Model Performance for {coin_name}:")
  print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
  plt.show()
  return fig

  
def save_fig_as_image(fig, image_path):
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    return image_path

def visualize_feature_importance(model, feature_names, coin_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Only show top N features for readability
    top_n = 10
    top_indices = indices[:top_n]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(range(top_n), importances[top_indices], color="r", align="center")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right', fontsize=10)
    ax.set_xlim([-1, top_n])
    ax.set_title(f'Feature Importances for {coin_name}')
    plt.tight_layout()
    plt.show()

    return fig

def visualize_regression_model(y_test, y_pred, coin_name):
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.scatter(y_test.values, y_pred, alpha=0.5, label='Predicted vs Actual', color='blue', edgecolor='w', s=80)
  ax.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--', label='Best Fit')
  ax.set_xlabel('Actual Close Price')
  ax.set_ylabel('Predicted Close Price')
  ax.set_title(f'Regression Model Performance for {coin_name}')
  ax.legend()
  ax.grid(True)
  plt.show()
  return fig

def visualize_residuals(y_true, y_pred, coin_name):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs Time
    ax1.scatter(range(len(residuals)), residuals, alpha=0.5)
    ax1.set_title(f'Residuals Over Time - {coin_name}')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Residuals')
    
    # ACF Plot
    plot_acf(residuals, lags=40, ax=ax2)
    ax2.set_title(f'Autocorrelation - {coin_name}')
    
    plt.tight_layout()
    return fig


def explain_model(model, X_train, feature_names, coin_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    fig1 = plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.title(f'{coin_name} Feature Impact (SHAP)')
    
    return fig1


def visualize_residual_analysis(y_true, y_pred, dates):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Residuals timeline
    ax1.plot(dates, residuals, color='purple')
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title('Residuals Over Time')
    
    # ACF plot
    plot_acf(residuals, lags=40, ax=ax2)
    ax2.set_title('Autocorrelation')
    
    plt.tight_layout()
    return fig

