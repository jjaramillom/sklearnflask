import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify, make_response, abort
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from Anomalies_Detector import AnomalyDetector
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# inputs

pickelName = 'testing'

model_directory = 'model'
model_file_name = '{}/{}.pkl'.format(model_directory, pickelName)
model_columns_file_name = '{}/{}_columns.pkl'.format(model_directory, pickelName)

# These will be populated at training time
model_columns = None
predictor = None


@app.route('/', methods=['GET'])
def main():
    #print(request.args['version'])
	make_response(jsonify({'valid requests': ['/train - POST', '/predict - POST', '/wipe - GET']}), 201)


@app.route('/predict', methods=['POST'])
def predict():
    if predictor == None:
        print('train first')
        message = ({'error': 'There is no existing trained model. Please create one before deleting it.'})
        # return message
        abort(400, message)
        # raise BadRequest(message)
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
        message =  jsonify({'error': str(e), 'trace': traceback.format_exc()})
        raise BadRequest(message)
        


@app.route('/train', methods=['POST'])
def train():
    try:
        body = request.json
        df = pd.DataFrame(columns=body['data']['columnNames'], data=body['data']['values'][:100])
    except Exception as e:
        message =  jsonify({'error': str(e), 'trace': traceback.format_exc()})
        # return jsonify(message)
        raise BadRequest(message)
    global model_columns
    model_columns = []
    model_columns.append(body['parameters']['datetimeName'])
    model_columns.append(body['parameters']['targetName'])
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
    return make_response(jsonify(response), 201)


@app.route('/wipe', methods=['GET'])
def wipe():
    if (predictor == None):
        message = jsonify({'error': 'There is no existing trained model. Please create one before deleting it.'})
        raise BadRequest(message)
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return jsonify({'msg': 'Model deleted.'})
        predict = None

    except Exception as e:
        return jsonify({"error": str(e)})


#if __name__ == '__main__':
    #port = 80

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
    #app.run(debug=True)
