import unittest
import json
import numpy as np
import pandas as pd
import torch
from src.preprocessor import TitanicPreprocessor
from src.dataset import TitanicTorchDataset, create_data_loaders
from src.model import TitanicDeepNet
from src.predict import TitanicPredictor
from app.server import app

class TestTitanicProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_data = pd.DataFrame([
            {
                'PassengerId': 1001,
                'Survived': 0,
                'Pclass': 3,
                'Name': 'Braund, Mr. Owen Harris',
                'Sex': 'male',
                'Age': 22.0,
                'SibSp': 1,
                'Parch': 0,
                'Ticket': 'A/5 21171',
                'Fare': 7.25
            },
            {
                'PassengerId': 1002,
                'Survived': 1,
                'Pclass': 1,
                'Name': 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                'Sex': 'female',
                'Age': 38.0,
                'SibSp': 1,
                'Parch': 0,
                'Ticket': 'PC 17599',
                'Fare': 71.2833
            },
            {
                'PassengerId': 1003,
                'Survived': 1,
                'Pclass': 3,
                'Name': 'Heikkinen, Miss. Laina',
                'Sex': 'female',
                'Age': None, # Test missing age
                'SibSp': 0,
                'Parch': 0,
                'Ticket': 'STON/O2. 3101282',
                'Fare': None # Test missing fare
            }
        ])

    def test_01_preprocessor_fit_transform(self):
        preprocessor = TitanicPreprocessor()
        X = preprocessor.fit_transform(self.raw_data)

        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[1], len(preprocessor.feature_names))
        self.assertFalse(np.isnan(X).any(), "Preprocessor output should contain no NaNs.")

        # Test single transformation
        single_passenger = {
            'Pclass': 2,
            'Name': 'Navratil, Mr. Michel',
            'Sex': 'male',
            'Age': 36.5,
            'SibSp': 0,
            'Parch': 2,
            'Ticket': '230080',
            'Fare': 26.0
        }
        X_single = preprocessor.transform_single(single_passenger)
        self.assertEqual(X_single.shape, (1, len(preprocessor.feature_names)))

    def test_02_dataset_and_loader(self):
        preprocessor = TitanicPreprocessor()
        X = preprocessor.fit_transform(self.raw_data)
        y = np.array([0, 1, 1], dtype=np.float32)

        dataset = TitanicTorchDataset(X, y)
        self.assertEqual(len(dataset), 3)

        x_sample, y_sample = dataset[0]
        self.assertEqual(x_sample.shape[0], X.shape[1])
        self.assertEqual(y_sample.shape, (1,))

        train_loader, val_loader = create_data_loaders(X, y, X, y, batch_size=2)
        batch_x, batch_y = next(iter(train_loader))
        self.assertEqual(batch_x.dtype, torch.float32)
        self.assertEqual(batch_y.dtype, torch.float32)

    def test_03_neural_net_forward(self):
        input_dim = 23
        model = TitanicDeepNet(input_dim=input_dim, hidden_dim=64, dropout_rate=0.2)
        model.eval()

        # Batch forward pass
        dummy_batch = torch.randn(4, input_dim, dtype=torch.float32)
        logits = model(dummy_batch)
        self.assertEqual(logits.shape, (4, 1))

        # Predict proba
        probs = model.predict_proba(dummy_batch)
        self.assertEqual(len(probs), 4)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        # Predict binary
        preds = model.predict(dummy_batch)
        self.assertTrue(np.all(np.isin(preds, [0, 1])))

    def test_04_predictor_inference(self):
        predictor = TitanicPredictor()
        test_passenger = {
            'Pclass': 1,
            'Name': 'Widener, Mrs. George Dunton (Eleanor Elkins)',
            'Sex': 'female',
            'Age': 50.0,
            'SibSp': 1,
            'Parch': 1,
            'Ticket': '113503',
            'Fare': 211.5
        }
        result = predictor.predict_passenger(test_passenger)
        self.assertIn('survived', result)
        self.assertIn('probability', result)
        self.assertIn('insights', result)
        self.assertGreaterEqual(result['probability'], 0.0)
        self.assertLessEqual(result['probability'], 1.0)
        self.assertIsInstance(result['insights'], list)

    def test_05_flask_api_endpoints(self):
        client = app.test_client()

        # Test GET /
        res = client.get('/')
        self.assertEqual(res.status_code, 200)

        # Test GET /api/metrics
        res = client.get('/api/metrics')
        self.assertEqual(res.status_code, 200)
        metrics = json.loads(res.data)
        self.assertIn('validation_metrics', metrics)

        # Test GET /api/passengers
        res = client.get('/api/passengers?page=1&page_size=5')
        self.assertEqual(res.status_code, 200)
        passengers = json.loads(res.data)
        self.assertIn('records', passengers)
        self.assertEqual(len(passengers['records']), 5)

        # Test POST /api/predict
        payload = {
            'Name': 'Cardeza, Mrs. James Warburton Martinez',
            'Pclass': 1,
            'Sex': 'female',
            'Age': 58,
            'SibSp': 0,
            'Parch': 1,
            'Ticket': 'PC 17755',
            'Fare': 512.33
        }
        res = client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['survived'], 1)

if __name__ == '__main__':
    unittest.main()
