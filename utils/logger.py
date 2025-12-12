"""
Unified logging module for model training with W&B integration.
"""
import wandb
import joblib
import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, 
    roc_curve, auc, average_precision_score
)
from sklearn.base import BaseEstimator
import pandas as pd


class ExperimentLogger:
    """Unified logger for model training, evaluation, and visualization."""
    
    def __init__(self, experiment_name: str, wandb_config: Dict[str, Any]):
        """
        Initialize logger with W&B.
        
        Args:
            experiment_name: Name of the experiment
            wandb_config: W&B config from config.yaml (project_name, entity, etc.)
        """
        self.experiment_name = experiment_name
        
        # Initialize W&B
        self.run = wandb.init(
            project=wandb_config.get('project_name', 'german-credit'),
            entity=wandb_config.get('entity'),
            name=experiment_name,
            config={}
        )
    
    def log_config(self, config: Dict[str, Any]):
        """Update W&B config."""
        wandb.config.update(config)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to W&B."""
        wandb.log(metrics)
    
    def log_model_results(self, 
                         model_name: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray, 
                         y_prob: np.ndarray,
                         feature_names: list = None,
                         model: BaseEstimator = None):
        """
        Log all visualizations for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            feature_names: Feature names for importance plot
            model: Model for feature importance
        """
        prefix = model_name.lower().replace(" ", "_")
        
        # Confusion Matrix
        fig = self._plot_confusion_matrix(y_true, y_pred, model_name)
        wandb.log({f"{prefix}_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        
        # ROC Curve
        fig = self._plot_roc_curve(y_true, y_prob, model_name)
        wandb.log({f"{prefix}_roc_curve": wandb.Image(fig)})
        plt.close(fig)
        
        # Precision-Recall Curve
        fig = self._plot_pr_curve(y_true, y_prob, model_name)
        wandb.log({f"{prefix}_pr_curve": wandb.Image(fig)})
        plt.close(fig)
        
        # Feature Importance
        if model is not None and feature_names is not None:
            try:
                fig = self._plot_feature_importance(model, feature_names, model_name)
                wandb.log({f"{prefix}_feature_importance": wandb.Image(fig)})
                plt.close(fig)
            except Exception as e:
                print(f"Could not plot feature importance: {e}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Good', 'Bad'],
                   yticklabels=['Good', 'Bad'], ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        return fig
    
    def _plot_roc_curve(self, y_true, y_prob, model_name):
        """Plot ROC curve."""
        # Handle 2D probability arrays
        if len(y_prob.shape) > 1:
            y_prob = y_prob[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    def _plot_pr_curve(self, y_true, y_prob, model_name):
        """Plot Precision-Recall curve."""
        # Handle 2D probability arrays
        if len(y_prob.shape) > 1:
            y_prob = y_prob[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        # Find best F1 threshold
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = f1.argmax()
        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
        ax.scatter(recall[best_idx], precision[best_idx], 
                  color='red', s=100, zorder=5,
                  label=f'Best F1 = {f1[best_idx]:.3f} @ {best_thr:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} - Precision-Recall Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    def _plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance."""
        # Extract model from pipeline
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps.get('clf') or list(model.named_steps.values())[-1]
        else:
            actual_model = model
        
        # Get importances
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
            coeffs = None
        elif hasattr(actual_model, 'coef_'):
            coeffs = actual_model.coef_[0] if len(actual_model.coef_.shape) > 1 else actual_model.coef_
            importances = np.abs(coeffs)
        else:
            raise ValueError("Model has no feature importance")
        
        # Handle length mismatch
        if len(feature_names) != len(importances):
            print(f"WARNING: Feature names mismatch ({len(feature_names)} vs {len(importances)})")
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame and sort
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'coef': coeffs if coeffs is not None else importances
        }).sort_values('importance', ascending=False).head(20)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if coeffs is not None:
            colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in df['coef']]
            ax.barh(range(len(df)), df['coef'], color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Coefficient')
        else:
            ax.barh(range(len(df)), df['importance'], color='steelblue', alpha=0.7)
            ax.set_xlabel('Importance')
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'])
        ax.set_title(f'{model_name} - Feature Importance')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
    
    def save_model(self, model: BaseEstimator, model_name: str, save_dir: str = "models"):
        """
        Save model to disk and log as W&B artifact.
        
        Args:
            model: Trained model
            model_name: Name for the saved model
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save locally
        model_path = os.path.join(save_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # Log to W&B
        artifact = wandb.Artifact(
            name=f"{model_name}_model",
            type="model",
            description=f"Trained {model_name} model"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        print(f"Model saved: {model_path}")
    
    def finish(self):
        """Finish W&B run."""
        wandb.finish()