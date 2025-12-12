from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from typing import Dict, Any


class ModelLoader:
    """Unified pipeline builder for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing 'data' and 'model' configuration
                    from config.yaml
        """
        self.data_config = config['DataConfig']
        self.model_config = config['model']
    
    def _get_logreg_preprocessor(self) -> ColumnTransformer:
        """Preprocessing for LogisticRegression (with scaling)."""
        return ColumnTransformer([
            ("nominal", OneHotEncoder(handle_unknown="ignore"), 
             self.data_config['nominal_features']),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", 
             unknown_value=-1), self.data_config['ordinal_features']),
            ("num", RobustScaler(), self.data_config['num_features'])
        ])
    
    def _get_tree_preprocessor(self) -> ColumnTransformer:
        """Preprocessing for tree models (no scaling)."""
        return ColumnTransformer([
            ("nominal", OrdinalEncoder(handle_unknown="use_encoded_value", 
             unknown_value=-1), self.data_config['nominal_features']),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", 
             unknown_value=-1), self.data_config['ordinal_features']),
            ("num", "passthrough", self.data_config['num_features'])
        ])
    
    def get_pipeline(self, model_type: str = "logreg") -> ImbPipeline:
        """
        Build complete pipeline based on model type and config.
        
        Args:
            model_type: "logreg" or "xgboost"
            
        Returns:
            Complete sklearn pipeline ready for training
        """
        model_type = model_type.lower()
        
        if model_type == "logreg":
            return self._build_logreg_pipeline()
        elif model_type == "xgboost":
            return self._build_xgb_pipeline()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _build_logreg_pipeline(self) -> ImbPipeline:
        """Build LogisticRegression pipeline."""
        cfg = self.model_config['logreg']
        
        steps = [("preprocess", self._get_logreg_preprocessor())]
        
        # Add SMOTE if enabled
        if cfg.get('use_smote', False):
            steps.append(("smote", SMOTE(random_state=42)))
        
        # Add classifier
        steps.append(("clf", LogisticRegression(
            penalty=cfg['penalty'],
            C=cfg['C'],
            solver=cfg['solver'],
            max_iter=cfg['max_iter'],
            random_state=42
        )))
        
        return ImbPipeline(steps=steps)
    
    def _build_xgb_pipeline(self) -> ImbPipeline:
        """Build XGBoost pipeline."""
        cfg = self.model_config['xgboost']
        
        # Auto-detect GPU
        tree_method, device = self._detect_device()
        
        return ImbPipeline(steps=[
            ("preprocess", self._get_tree_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=cfg['n_estimators'],
                max_depth=cfg['max_depth'],
                learning_rate=cfg['learning_rate'],
                subsample=cfg['subsample'],
                colsample_bytree=cfg['colsample_bytree'],
                reg_lambda=cfg['reg_lambda'],
                reg_alpha=cfg['reg_alpha'],
                gamma=cfg['gamma'],
                tree_method=tree_method,
                device=device,
                enable_categorical=True,
                eval_metric="logloss",
                random_state=42
            ))
        ])
    
    @staticmethod
    def _detect_device():
        """Detect if GPU is available for XGBoost."""
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix([[1, 2]], label=[0])
            xgb.train(
                params={"tree_method": "gpu_hist", "device": "cuda"},
                dtrain=dmat, num_boost_round=1, verbose_eval=False
            )
            return "gpu_hist", "cuda"
        except:
            return "hist", "cpu"