"""
Main script to run German Credit classification experiments.

Usage:
    python main.py --experiment logreg_baseline
    python main.py --experiment logreg_with_smote
    python main.py --experiment xgb_baseline
    python main.py --experiment logreg_vs_xgb
"""
import yaml
from pathlib import Path
import argparse
from utils.trainer import Trainer
import utils.experiments as experiments
from loaders.dataloader import DataLoader


def load_config(config_path="configs/config.yaml"):
    """Load YAML configuration file."""
    main_dir = Path(__file__).parent
    config_path = main_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")
    return config


def get_feature_names_after_preprocessing(dataloader, X_train_raw):
    """
    Get feature names after preprocessing pipeline.
    Handles OneHotEncoder, OrdinalEncoder, and passthrough.
    """
    feature_names = []
    
    # Nominal features (OneHotEncoded)
    for feat in dataloader.config['nominal_features']:
        unique_vals = X_train_raw[feat].nunique()
        feature_names.extend([f"{feat}_{i}" for i in range(unique_vals)])
    
    # Ordinal features (OrdinalEncoded - same name)
    feature_names.extend(dataloader.config['ordinal_features'])
    
    # Numerical features (scaled - same name)
    feature_names.extend(dataloader.config['num_features'])
    
    return feature_names


def main():
    # Load config
    config = load_config()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run German Credit experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=[
            'logreg_baseline',
            'logreg_with_smote',
            'xgb_baseline',
            'logreg_vs_xgb'
        ],
        help='Experiment to run'
    )
    parser.add_argument('--random_state', type=int, default=42)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading German Credit data...")
    dataloader = DataLoader(config)
    X_raw, y_raw = dataloader.load_data()
    X_train, X_test, y_train, y_test = dataloader.get_train_test_split(
        test_size=config['DataConfig']['test_size'],
        random_state=config['DataConfig']['random_state']
    )
    
    # Get feature names (approximation for after preprocessing)
    feature_names = get_feature_names_after_preprocessing(dataloader, X_raw)
    print(f"Number of features after preprocessing: {len(feature_names)}")
    
    # Get experiment config (returns list of model configs)
    print(f"\nRunning experiment: {args.experiment}")
    experiment_func = getattr(experiments, args.experiment)
    models_config = experiment_func(config, random_state=args.random_state)
    
    # Train with integrated logging
    trainer = Trainer(
        experiment_name=args.experiment,
        wandb_config=config['WandB'],
        models_config=models_config
    )
    
    results = trainer.train_and_evaluate(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  CV Score: {result['cv_score']:.4f}")
        print(f"  Test AUC: {result['test_auc']:.4f}")
    
    # Finish logging
    trainer.finish()
    print(f"\nResults logged to W&B: {args.experiment}")


if __name__ == '__main__':
    main()