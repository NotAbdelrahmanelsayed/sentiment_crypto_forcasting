from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np 
from modeling import evaluate_classification_model


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