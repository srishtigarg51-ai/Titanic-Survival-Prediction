import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

class TitanicPreprocessor:
    """
    Data Preprocessing & Feature Engineering Pipeline for Titanic Survival Prediction.
    Transforms raw passenger attributes into normalized, encoded feature vectors
    ready for deep neural networks.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.age_imputer_map = {}
        self.global_age_median = 28.0
        self.fare_imputer_map = {}
        self.global_fare_median = 14.45
        self.num_cols = ['Age', 'Fare_Log', 'FamilySize', 'SibSp', 'Parch']
        self.cat_cols = ['Pclass', 'Sex', 'Title', 'Age_Group']
        self.binary_cols = ['IsAlone', 'Is_Child', 'Is_Mother']
        self.feature_names = []
        self.is_fitted = False

    @staticmethod
    def extract_title(name: str) -> str:
        """Extract and categorize title from passenger name."""
        if not isinstance(name, str) or not name.strip():
            return 'Mr'
        match = re.search(r' ([A-Za-z]+)\.', name)
        if not match:
            return 'Mr'
        title = match.group(1).strip()
        title_map = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Ms': 'Miss',
            'Mlle': 'Miss',
            'Mme': 'Mrs',
            'Lady': 'Rare',
            'Countess': 'Rare',
            'Capt': 'Rare',
            'Col': 'Rare',
            'Don': 'Rare',
            'Dr': 'Rare',
            'Major': 'Rare',
            'Rev': 'Rare',
            'Sir': 'Rare',
            'Jonkheer': 'Rare',
            'Dona': 'Rare'
        }
        return title_map.get(title, 'Rare')

    @staticmethod
    def get_age_group(age: float) -> str:
        """Assign age into demographic group."""
        if pd.isna(age) or age < 0:
            return 'Adult'
        if age < 12:
            return 'Child'
        elif age < 18:
            return 'Teen'
        elif age < 35:
            return 'YoungAdult'
        elif age < 60:
            return 'Adult'
        else:
            return 'Senior'

    def engineer_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Derive domain-specific features and handle missing values."""
        df = df.copy()

        # 1. Clean Title
        if 'Name' in df.columns:
            df['Title'] = df['Name'].apply(self.extract_title)
        elif 'Title' not in df.columns:
            df['Title'] = 'Mr'

        # Ensure Pclass is numeric/categorical
        if 'Pclass' in df.columns:
            df['Pclass'] = pd.to_numeric(df['Pclass'], errors='coerce').fillna(3).astype(int)
        else:
            df['Pclass'] = 3

        # Missing Fare Imputation
        if is_training:
            self.global_fare_median = float(df['Fare'].median() if not df['Fare'].dropna().empty else 14.45)
            self.fare_imputer_map = df.groupby('Pclass')['Fare'].median().to_dict()

        fare_impute_series = df['Pclass'].map(self.fare_imputer_map).fillna(self.global_fare_median)
        df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce').fillna(fare_impute_series)
        df['Fare_Log'] = np.log1p(np.maximum(0.0, df['Fare'].values))

        # Missing Age Imputation by (Title, Pclass) group
        if is_training:
            self.global_age_median = float(df['Age'].median() if not df['Age'].dropna().empty else 28.0)
            grouped = df.groupby(['Title', 'Pclass'])['Age'].median()
            self.age_imputer_map = grouped.to_dict()

        imputed_ages = []
        for _, row in df.iterrows():
            age_val = row.get('Age', np.nan)
            if pd.isna(age_val):
                key = (row.get('Title', 'Mr'), row.get('Pclass', 3))
                val = self.age_imputer_map.get(key, self.global_age_median)
                if pd.isna(val):
                    val = self.global_age_median
                imputed_ages.append(float(val))
            else:
                imputed_ages.append(float(age_val))
        df['Age'] = imputed_ages

        # Ensure SibSp and Parch
        df['SibSp'] = pd.to_numeric(df.get('SibSp', 0), errors='coerce').fillna(0).astype(int)
        df['Parch'] = pd.to_numeric(df.get('Parch', 0), errors='coerce').fillna(0).astype(int)

        # Family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Sex normalization
        df['Sex'] = df.get('Sex', 'male').astype(str).str.strip().str.lower()
        df['Sex'] = df['Sex'].apply(lambda s: 'female' if s == 'female' else 'male')

        # Demographics & interactions
        df['Age_Group'] = df['Age'].apply(self.get_age_group)
        df['Is_Child'] = (df['Age'] < 12).astype(int)
        df['Is_Mother'] = (
            (df['Sex'] == 'female') &
            (df['Parch'] > 0) &
            (df['Age'] > 18) &
            (df['Title'].isin(['Mrs', 'Miss']))
        ).astype(int)

        return df

    def fit(self, df: pd.DataFrame):
        """Fit scaler and encoder on training data."""
        processed_df = self.engineer_features(df, is_training=True)

        # Fit numerical scaler
        self.scaler.fit(processed_df[self.num_cols])

        # Fit categorical one-hot encoder
        self.encoder.fit(processed_df[self.cat_cols])

        # Build feature names
        cat_feature_names = list(self.encoder.get_feature_names_out(self.cat_cols))
        self.feature_names = self.num_cols + self.binary_cols + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw passenger dataframe into preprocessed feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("TitanicPreprocessor must be fitted before calling transform().")

        processed_df = self.engineer_features(df, is_training=False)

        num_scaled = self.scaler.transform(processed_df[self.num_cols])
        binary_arr = processed_df[self.binary_cols].values.astype(np.float32)
        cat_encoded = self.encoder.transform(processed_df[self.cat_cols])

        return np.hstack([num_scaled, binary_arr, cat_encoded]).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and return transformed feature matrix."""
        self.fit(df)
        return self.transform(df)

    def transform_single(self, passenger_dict: dict) -> np.ndarray:
        """Helper for single passenger inference (e.g. from web app or API)."""
        df = pd.DataFrame([passenger_dict])
        return self.transform(df)

    def save(self, filepath: str):
        """Persist preprocessor pipeline to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'TitanicPreprocessor':
        """Load persisted preprocessor from disk."""
        return joblib.load(filepath)
