import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.predict import TitanicPredictor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(project_root, 'app', 'templates')
static_dir = os.path.join(project_root, 'app', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Initialize predictor
predictor = TitanicPredictor()

# Load metrics
metrics_path = os.path.join(project_root, 'saved_models', 'metrics.json')
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
else:
    metrics_data = {}

# Load raw dataset for the explorer tab
dataset_path = os.path.join(project_root, 'data', 'titanic.csv')
if os.path.exists(dataset_path):
    raw_df = pd.read_csv(dataset_path)
else:
    raw_df = pd.DataFrame()

# Predefined passenger presets
PASSENGER_PRESETS = [
    {
        "id": "rose",
        "name": "Rose DeWitt Bukater",
        "label": "1st Class Noblewoman (High Survival)",
        "pclass": 1,
        "sex": "female",
        "age": 17,
        "sibsp": 0,
        "parch": 1,
        "ticket": "PC 17599",
        "fare": 150.0
    },
    {
        "id": "jack",
        "name": "Jack Dawson",
        "label": "3rd Class Young Traveler (High Risk)",
        "pclass": 3,
        "sex": "male",
        "age": 20,
        "sibsp": 0,
        "parch": 0,
        "ticket": "349202",
        "fare": 7.89
    },
    {
        "id": "smith",
        "name": "Captain Edward Smith",
        "label": "Senior Officer / Crew (High Risk)",
        "pclass": 1,
        "sex": "male",
        "age": 62,
        "sibsp": 0,
        "parch": 0,
        "ticket": "113054",
        "fare": 50.0
    },
    {
        "id": "child_3rd",
        "name": "Alfred Edward Peacock",
        "label": "3rd Class Infant/Child (Prioritized)",
        "pclass": 3,
        "sex": "male",
        "age": 0.75,
        "sibsp": 1,
        "parch": 1,
        "ticket": "SOTON/O.Q. 3101315",
        "fare": 13.77
    },
    {
        "id": "mrs_middle",
        "name": "Mrs. Walter H Corbett",
        "label": "2nd Class Married Woman (High Survival)",
        "pclass": 2,
        "sex": "female",
        "age": 30,
        "sibsp": 0,
        "parch": 0,
        "ticket": "237249",
        "fare": 13.0
    }
]

@app.route('/')
def index():
    return render_template(
        'index.html',
        metrics=metrics_data,
        presets=PASSENGER_PRESETS,
        total_passengers=len(raw_df)
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Validate and clean inputs
        passenger = {
            'Name': str(data.get('Name', 'Passenger')).strip(),
            'Pclass': int(data.get('Pclass', 3)),
            'Sex': str(data.get('Sex', 'male')).strip().lower(),
            'Age': float(data.get('Age', 28.0) if data.get('Age') is not None and data.get('Age') != '' else 28.0),
            'SibSp': int(data.get('SibSp', 0) if data.get('SibSp') is not None and data.get('SibSp') != '' else 0),
            'Parch': int(data.get('Parch', 0) if data.get('Parch') is not None and data.get('Parch') != '' else 0),
            'Ticket': str(data.get('Ticket', 'STANDARD')).strip(),
            'Fare': float(data.get('Fare', 14.45) if data.get('Fare') is not None and data.get('Fare') != '' else 14.45)
        }

        result = predictor.predict_passenger(passenger)
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    return jsonify(metrics_data)

@app.route('/api/passengers', methods=['GET'])
def api_passengers():
    """Return filtered/paginated passengers from the original dataset."""
    search = request.args.get('search', '').lower().strip()
    pclass = request.args.get('pclass', '')
    sex = request.args.get('sex', '')
    survived = request.args.get('survived', '')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 15))

    filtered_df = raw_df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df['Name'].str.lower().str.contains(search, na=False) |
            filtered_df['Ticket'].astype(str).str.lower().str.contains(search, na=False)
        ]

    if pclass:
        filtered_df = filtered_df[filtered_df['Pclass'] == int(pclass)]

    if sex:
        filtered_df = filtered_df[filtered_df['Sex'] == sex]

    if survived:
        filtered_df = filtered_df[filtered_df['Survived'] == int(survived)]

    total_records = len(filtered_df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    sliced = filtered_df.iloc[start_idx:end_idx]
    records = sliced.fillna('').to_dict(orient='records')

    return jsonify({
        'total': total_records,
        'page': page,
        'page_size': page_size,
        'total_pages': max(1, (total_records + page_size - 1) // page_size),
        'records': records
    })

@app.route('/api/presets', methods=['GET'])
def api_presets():
    return jsonify(PASSENGER_PRESETS)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Titanic Deep Learning Web App on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
