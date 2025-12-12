"""
Experiment configurations for German Credit classification.
Uses ModelLoader to build pipelines from config.yaml settings.
"""
from loaders.modelloader import ModelLoader


def logreg_baseline(config, random_state=42):
    """LogReg from config, no SMOTE"""
    # Temporarily disable SMOTE
    model_config = config['model'].copy()
    model_config['logreg']['use_smote'] = False
    temp_config = {'DataConfig': config['DataConfig'], 'model': model_config}
    
    loader = ModelLoader(temp_config)
    pipeline = loader.get_pipeline("logreg")
    param_grid = model_config['logreg']['param_grid']
    
    return [{
        'name': 'LogisticRegression',
        'pipeline': pipeline,
        'param_grid': param_grid,
        'cv_folds': model_config['logreg']['cv_folds'],
        'scoring': model_config['logreg']['scoring']
    }]


def logreg_with_smote(config, random_state=42):
    """LogReg from config, with SMOTE"""
    # Ensure SMOTE is enabled
    model_config = config['model'].copy()
    model_config['logreg']['use_smote'] = True
    temp_config = {'DataConfig': config['DataConfig'], 'model': model_config}
    
    loader = ModelLoader(temp_config)
    pipeline = loader.get_pipeline("logreg")
    param_grid = model_config['logreg']['param_grid']
    
    return [{
        'name': 'LogisticRegression_SMOTE',
        'pipeline': pipeline,
        'param_grid': param_grid,
        'cv_folds': model_config['logreg']['cv_folds'],
        'scoring': model_config['logreg']['scoring']
    }]


def xgb_baseline(config, random_state=42):
    """XGBoost from config"""
    loader = ModelLoader(config)
    pipeline = loader.get_pipeline("xgboost")
    param_grid = config['model']['xgboost']['param_grid']
    
    return [{
        'name': 'XGBoost',
        'pipeline': pipeline,
        'param_grid': param_grid,
        'cv_folds': config['model']['xgboost']['cv_folds'],
        'scoring': config['model']['xgboost']['scoring']
    }]


def logreg_vs_xgb(config, random_state=42):
    """Compare LogReg (no SMOTE) vs XGBoost"""
    # LogReg without SMOTE
    model_config = config['model'].copy()
    model_config['logreg']['use_smote'] = False
    temp_config = {'DataConfig': config['DataConfig'], 'model': model_config}
    
    loader = ModelLoader(temp_config)
    logreg_pipeline = loader.get_pipeline("logreg")
    
    # XGBoost
    loader_xgb = ModelLoader(config)
    xgb_pipeline = loader_xgb.get_pipeline("xgboost")
    
    return [
        {
            'name': 'LogisticRegression',
            'pipeline': logreg_pipeline,
            'param_grid': config['model']['logreg']['param_grid'],
            'cv_folds': config['model']['logreg']['cv_folds'],
            'scoring': config['model']['logreg']['scoring']
        },
        {
            'name': 'XGBoost',
            'pipeline': xgb_pipeline,
            'param_grid': config['model']['xgboost']['param_grid'],
            'cv_folds': config['model']['xgboost']['cv_folds'],
            'scoring': config['model']['xgboost']['scoring']
        }
    ]