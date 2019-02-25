import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from Anomalies_Detector import AnomalyDetector

app = Flask(__name__)

# inputs
training_data = 'data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = include[-1]

pickelName = 'testing'

model_directory = 'model'
model_file_name = '{}/{}.pkl'.format(model_directory, pickelName)
model_columns_file_name = '{}/{}_columns.pkl'.format(model_directory, pickelName)

# These will be populated at training time
model_columns = None
predictor = None


@app.route('/', methods=['GET'])
def main():
    print(request.args['version'])
    return jsonify({'valid requests': ['/train - POST', '/predict - POST', '/wipe - GET']})


@app.route('/predict', methods=['POST'])
def predict():
    if predictor:
        try:
            body = request.json
            df = pd.DataFrame(columns=body['data']['columnNames'], data=body['data']['values'])
            global model_columns
            df = df[model_columns]
            start = time.time()
            predictions = predictor.predict(df)
            trainingTime = np.around(time.time() - start, decimals=3)
            print('prediction in {0:.3f} seconds'.format(trainingTime))
            return predictions.to_json(orient='records')

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return {'error': 'No trained model available for predictions'}


@app.route('/train', methods=['POST'])
def train():
    body = request.json
    df = pd.DataFrame(columns=body['data']['columnNames'], data=body['data']['values'])
    global model_columns
    model_columns = []
    model_columns.append(body['parameters']['datetimeName'])
    model_columns += body['parameters']['predictorNames']
    df = df[model_columns]
    global predictor
    predictor = AnomalyDetector(train_df=df, target=body['parameters']['targetName'], pastTargetasPredictor = True,
                               timeVar=body['parameters']['datetimeName'], predictors=body['parameters']['predictorNames'])
    start = time.time()
    model, score = predictor.trainModel()
    trainingTime = np.around(time.time() - start, decimals=3)
    print('Trained in {0:.3f} seconds'.format(trainingTime))
    print('Scores: {}'.format(score))
    joblib.dump(predictor, model_file_name)
    joblib.dump(model_columns, model_columns_file_name)
    features = predictor.feature_importances()
    if ('ERROR' in features):
        response = {'training time': time.time() - start, 'score': score}
    else:
        featuresImp = []
        for feature in features.index.values:
             featuresImp.append({feature: features.loc[feature]['importance']})
        print(features)
        response = {'training time': trainingTime, 'score': score, 'features importances': featuresImp} #'model params': model.get_params()
    return jsonify(response)


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('predictor')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print("type error: " + str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    port = 80

    try:
        predictor = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        predictor = None

    #app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)
