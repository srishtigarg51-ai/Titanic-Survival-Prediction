import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from src.preprocessor import TitanicPreprocessor
from src.model import TitanicDeepNet

class TitanicPredictor:
    """
    Inference Engine for Titanic Passenger Survival.
    Loads persisted neural network weights and preprocessing pipeline to provide
    real-time single and batch predictions with explainable risk insights.
    """
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(project_root, 'saved_models', 'titanic_deep_model.pt')
        if preprocessor_path is None:
            preprocessor_path = os.path.join(project_root, 'saved_models', 'preprocessor.pkl')

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Model or Preprocessor not found. Ensure training has run.\n"
                f"Model: {model_path}\nPreprocessor: {preprocessor_path}"
            )

        # Load Preprocessor
        self.preprocessor = TitanicPreprocessor.load(preprocessor_path)

        # Load PyTorch Model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = TitanicDeepNet(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint.get('hidden_dim', 64),
            dropout_rate=checkpoint.get('dropout_rate', 0.2)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_passenger(self, passenger: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a single passenger dictionary.
        Returns prediction, probability percentage, confidence level, and risk insights.
        """
        # Transform features
        X_vec = self.preprocessor.transform_single(passenger)

        # Model forward pass
        prob = float(self.model.predict_proba(X_vec)[0])
        survived = int(prob >= 0.5)

        # Categorize risk and confidence
        if prob >= 0.75:
            confidence = "High Chance of Survival"
            status_color = "success"
        elif prob >= 0.50:
            confidence = "Moderate Chance of Survival"
            status_color = "info"
        elif prob >= 0.25:
            confidence = "Moderate Risk of Perishing"
            status_color = "warning"
        else:
            confidence = "High Risk of Perishing"
            status_color = "danger"

        # Generate rule/feature-based human explanations
        insights = []
        sex = str(passenger.get('Sex', 'male')).lower().strip()
        pclass = int(passenger.get('Pclass', 3))
        age = float(passenger.get('Age', 28) or 28)
        fare = float(passenger.get('Fare', 14) or 14)
        sibsp = int(passenger.get('SibSp', 0) or 0)
        parch = int(passenger.get('Parch', 0) or 0)
        family_size = sibsp + parch + 1

        if sex == 'female':
            insights.append("Female gender is a primary positive survival factor ('women and children first' protocol).")
        else:
            insights.append("Male passengers faced significantly lower evacuation priority.")

        if pclass == 1:
            insights.append("1st Class passengers had preferential proximity to the boat deck and early evacuation notices.")
        elif pclass == 2:
            insights.append("2nd Class passengers had moderate boat deck access.")
        else:
            insights.append("3rd Class passengers were situated in lower decks, facing delayed access to lifeboats.")

        if age < 12:
            insights.append("Child status prioritized lifeboat boarding.")
        elif age > 60:
            insights.append("Senior age group had lower mobility during evacuation.")

        if family_size > 4:
            insights.append(f"Large family size ({family_size} members) posed coordination challenges during escape.")
        elif family_size == 1:
            insights.append("Solo passenger without dependent family constraints.")
        else:
            insights.append(f"Small family unit ({family_size} members) had balanced mobility.")

        if fare > 50:
            insights.append(f"High ticket fare (£{fare:.2f}) indicates premium cabin location.")

        return {
            'survived': survived,
            'survival_label': 'Survived' if survived == 1 else 'Did Not Survive',
            'probability': round(prob, 4),
            'probability_percentage': round(prob * 100, 2),
            'confidence': confidence,
            'status_color': status_color,
            'insights': insights,
            'passenger_summary': {
                'Name': passenger.get('Name', 'Unknown'),
                'Sex': sex.capitalize(),
                'Age': age,
                'Pclass': f"Class {pclass}",
                'Fare': f"£{fare:.2f}",
                'FamilySize': family_size
            }
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch inference on multiple passengers."""
        X_mat = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_mat)
        preds = (probs >= 0.5).astype(int)

        result_df = df.copy()
        result_df['Predicted_Survived'] = preds
        result_df['Survival_Probability'] = np.round(probs, 4)
        return result_df
