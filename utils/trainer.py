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
from utils.logger import ExperimentLogger

def minority_recall(y_true, y_pred):
    """Recall for Bad credits (minority class = 1)"""
    return recall_score(y_true, y_pred, pos_label=1)

class Trainer:
    """Trainer with integrated W&B logging."""
    
    def __init__(self, experiment_name: str, wandb_config: dict, models_config: list):
        """
        Args:
            experiment_name: Name of the experiment
            wandb_config: Config dict for W&B (from config.yaml WandB section)
            models_config: List of model dicts from experiments.py
                          [{'name': 'LogReg', 'pipeline': ..., 'param_grid': {...}}]
        """
        self.experiment_name = experiment_name
        self.models_config = models_config
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
            
            # Predictions
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)
            
            # Metrics
            test_auc = roc_auc_score(y_test, y_prob[:, 1])
            test_recall = recall_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nBest Params: {grid.best_params_}")
            print(f"CV Score: {grid.best_score_:.4f}")
            print(f"Test AUC: {test_auc:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test F1-Score: {test_f1:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"\n{classification_report(y_test, y_pred)}")
            
            # Log to W&B
            self.logger.log_config({
                f"{model_config['name']}_best_params": grid.best_params_
            })
            
            self.logger.log_metrics({
                f"{model_config['name']}_cv_score": grid.best_score_,
                f"{model_config['name']}_test_auc": test_auc,
                f"{model_config['name']}_test_recall": test_recall,
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
                'best_params': grid.best_params_,
                'cv_score': grid.best_score_,
                'test_auc': test_auc,
                'test_recall': test_recall,
                'test_precision': test_precision,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy
            }
        
        return self.results
    
    def finish(self):
        """Finish W&B logging."""
        self.logger.finish()