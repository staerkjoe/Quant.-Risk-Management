import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataLoader:
    """Handles complete data pipeline: loading, preprocessing, and splitting."""
    
    def __init__(self, config):
        self.attribute_mapping = {
            "Attribute1": "checking_status",
            "Attribute2": "duration",
            "Attribute3": "credit_history",
            "Attribute4": "purpose",
            "Attribute5": "credit_amount",
            "Attribute6": "savings",
            "Attribute7": "employment",
            "Attribute8": "installment_rate",
            "Attribute9": "personal_status_sex",
            "Attribute10": "other_debtors",
            "Attribute11": "residence_since",
            "Attribute12": "property",
            "Attribute13": "age",
            "Attribute14": "other_installment",
            "Attribute15": "housing",
            "Attribute16": "existing_credits",
            "Attribute17": "job",
            "Attribute18": "people_liable",
            "Attribute19": "telephone",
            "Attribute20": "foreign_worker",
            "class": "credit_risk"
        }
        # Access as dictionary instead of attribute
        self.config = config['DataConfig']
    
    def load_and_prepare(self) -> pd.DataFrame:
        """Load data from UCI, rename columns, transform target, and create features."""
        # Fetch and merge data
        dataset = fetch_ucirepo(id=self.config['dataset_id'])
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df = df.rename(columns=self.attribute_mapping)
        
        # Transform target: 1=Good->0, 2=Bad->1
        df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})
        print(f"Target distribution: {df['credit_risk'].value_counts().to_dict()}")
        
        # Create derived features
        df["credit_per_duration"] = df["credit_amount"] / (df["duration"] + 1)
        df["credit_per_age"] = df["credit_amount"] / (df["age"] + 1)
        df["credit_per_existing"] = df["credit_amount"] / (df["existing_credits"] + 1)
        
        # Validate features
        all_features = (self.config['ordinal_features'] + self.config['nominal_features'] + 
                       self.config['num_features'] + ["credit_risk"])
        missing = set(all_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        return df
    
    def get_train_test_split(self, test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load, prepare data and return train/test split."""
        df = self.load_and_prepare()
        
        X = df.drop(columns=["credit_risk"])
        y = df["credit_risk"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test