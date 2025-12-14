"""
Trainer module for German Credit classification experiments.
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    accuracy_score,
    make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from utils.logger import ExperimentLogger
import numpy as np

def minority_recall(y_true, y_pred):
    """Recall for Bad credits (minority class = 1)"""
    return recall_score(y_true, y_pred, pos_label=1)

def find_optimal_threshold(y_true, y_prob, metric='f1', cost_fp=1, cost_fn=5):
    """
    Find optimal classification threshold based on metric or business cost.
    
    Args:
        y_true: True labels (0=Good, 1=Bad)
        y_prob: Predicted probabilities for positive class (Bad)
        metric: 'f1', 'recall', 'precision', 'business_cost'
        cost_fp: Cost of False Positive (rejecting Good customer)
        cost_fn: Cost of False Negative (accepting Bad customer)
    
    Returns:
        tuple: (optimal_threshold, threshold_info_dict)
    """
    thresholds = np.arange(0.05, 0.95, 0.01)  # Extended range
    scores = []
    threshold_details = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
        tn = np.sum((y_pred_thresh == 0) & (y_true == 0))
        fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
        fn = np.sum((y_pred_thresh == 0) & (y_true == 1))
        
        if metric == 'business_cost':
            total_cost = (cost_fp * fp) + (cost_fn * fn)
            score = -total_cost
            
            # Store detailed info
            threshold_details.append({
                'threshold': thresh,
                'cost': total_cost,
                'fp': fp,
                'fn': fn,
                'good_recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'bad_recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        elif metric == 'f1':
            score = f1_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall_good':
            score = recall_score(y_true, y_pred_thresh, pos_label=0, zero_division=0)
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    
    info = threshold_details[optimal_idx] if threshold_details else {}
    
    return optimal_threshold, info

class Trainer:
    """Trainer with integrated W&B logging."""
    
    def __init__(self, experiment_name: str, wandb_config: dict, models_config: list, 
                 use_calibration: bool = True, cost_fp: float = 1.0, cost_fn: float = 3.0):
        """
        Args:
            experiment_name: Name of the experiment
            wandb_config: Config dict for W&B (from config.yaml WandB section)
            models_config: List of model dicts from experiments.py
            use_calibration: Whether to calibrate probabilities (recommended!)
            cost_fp: Business cost of rejecting Good customer (False Positive)
            cost_fn: Business cost of accepting Bad customer (False Negative)
        """
        self.experiment_name = experiment_name
        self.models_config = models_config
        self.use_calibration = use_calibration
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.logger = ExperimentLogger(
            experiment_name=experiment_name,
            wandb_config=wandb_config
        )
        self.results = {}
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Run grid search for all models and evaluate with logging.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data  
            feature_names: List of feature names after preprocessing
        
        Returns:
            dict: Results with best models and metrics
        """
        for model_config in self.models_config:
            print(f"\n{'='*60}")
            print(f"Training {model_config['name']}")
            print(f"{'='*60}")
            
            scoring = model_config.get('scoring', 'roc_auc')
            if scoring == 'recall':
                scoring = make_scorer(minority_recall)
            
            # Grid search
            grid = GridSearchCV(
                model_config['pipeline'],
                model_config['param_grid'],
                cv=model_config.get('cv_folds', 5),
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            
            # Optional: Calibrate probabilities for better threshold selection
            if self.use_calibration:
                print("Calibrating probabilities with isotonic regression...")
                calibrated_model = CalibratedClassifierCV(
                    best_model, 
                    method='isotonic',  # Better for tree-based models
                    cv='prefit'  # Already trained
                )
                # Use a small validation split for calibration
                from sklearn.model_selection import train_test_split
                X_train_cal, X_val_cal, y_train_cal, y_val_cal = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                calibrated_model.fit(X_val_cal, y_val_cal)
                y_prob = calibrated_model.predict_proba(X_test)
            else:
                y_prob = best_model.predict_proba(X_test)
            
            # Find optimal threshold with business cost optimization
            optimal_threshold, threshold_info = find_optimal_threshold(
                y_test, 
                y_prob[:, 1], 
                metric='business_cost',
                cost_fp=self.cost_fp,
                cost_fn=self.cost_fn
            )
            
            print(f"\nOptimal Threshold (Cost FP={self.cost_fp}, FN={self.cost_fn}): {optimal_threshold:.3f}")
            if threshold_info:
                print(f"  Expected FP (rejected Good): {threshold_info['fp']}")
                print(f"  Expected FN (accepted Bad): {threshold_info['fn']}")
                print(f"  Good Customer Recall: {threshold_info['good_recall']:.2%}")
                print(f"  Bad Customer Recall: {threshold_info['bad_recall']:.2%}")
            
            # Apply optimal threshold
            y_pred = (y_prob[:, 1] >= optimal_threshold).astype(int)
            
            # Compare with default 0.5 threshold
            y_pred_default = (y_prob[:, 1] >= 0.5).astype(int)
            good_recall_default = recall_score(y_test, y_pred_default, pos_label=0)
            
            # Metrics with optimal threshold
            test_auc = roc_auc_score(y_test, y_prob[:, 1])
            test_recall = recall_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_accuracy = accuracy_score(y_test, y_pred)
            good_recall = recall_score(y_test, y_pred, pos_label=0)
            
            print(f"\nBest Params: {grid.best_params_}")
            print(f"CV Score: {grid.best_score_:.4f}")
            print(f"Test AUC: {test_auc:.4f}")
            print(f"\nWith Optimal Threshold ({optimal_threshold:.3f}):")
            print(f"  Test Recall (Bad): {test_recall:.4f}")
            print(f"  Test Recall (Good): {good_recall:.4f} ← Business Critical!")
            print(f"  Test Precision: {test_precision:.4f}")
            print(f"  Test F1-Score: {test_f1:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"\nWith Default Threshold (0.5):")
            print(f"  Test Recall (Good): {good_recall_default:.4f}")
            print(f"  Improvement: +{(good_recall - good_recall_default)*100:.1f}%")
            print(f"\n{classification_report(y_test, y_pred, target_names=['Good', 'Bad'])}")
            
            # Log to W&B
            self.logger.log_config({
                f"{model_config['name']}_best_params": grid.best_params_,
                f"{model_config['name']}_optimal_threshold": optimal_threshold,
                f"{model_config['name']}_cost_fp": self.cost_fp,
                f"{model_config['name']}_cost_fn": self.cost_fn,
                f"{model_config['name']}_calibrated": self.use_calibration
            })
            
            self.logger.log_metrics({
                f"{model_config['name']}_cv_score": grid.best_score_,
                f"{model_config['name']}_test_auc": test_auc,
                f"{model_config['name']}_test_recall_bad": test_recall,
                f"{model_config['name']}_test_recall_good": good_recall,
                f"{model_config['name']}_test_recall_good_default": good_recall_default,
                f"{model_config['name']}_test_precision": test_precision,
                f"{model_config['name']}_test_f1": test_f1,
                f"{model_config['name']}_test_accuracy": test_accuracy
            })
            
            # Log visualizations
            self.logger.log_model_results(
                model_name=model_config['name'],
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                feature_names=feature_names,
                model=best_model
            )
            
            # Save model
            self.logger.save_model(
                best_model, 
                f"{self.experiment_name}_{model_config['name']}"
            )
            
            # Store results
            self.results[model_config['name']] = {
                'best_model': best_model,
                'calibrated_model': calibrated_model if self.use_calibration else None,
                'best_params': grid.best_params_,
                'optimal_threshold': optimal_threshold,
                'threshold_info': threshold_info,
                'cv_score': grid.best_score_,
                'test_auc': test_auc,
                'test_recall_bad': test_recall,
                'test_recall_good': good_recall,
                'test_precision': test_precision,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy
            }
        
        return self.results
    
    def finish(self):
        """Finish W&B logging."""
        self.logger.finish()