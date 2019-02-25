# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

class AnomalyDetector:
    
    def __init__(self, train_df: pd.DataFrame, target: str, timeVar: str, predictors: list, pastTargetasPredictor: bool = False, model: str = 'RandomForest'):
        self.TARGETVAR = target
        self.TIMEVAR = timeVar
        self.PASTVAR = pastTargetasPredictor
        self.MODELNAME = model
        self.TRAIN_DF = self._preprocess(train_df)
        
    def _get_x_y(self, dataFrame):
        y = dataFrame[self.TARGETVAR]
        return dataFrame.drop(self.TARGETVAR, axis = 1), y
        

    def _addPreviousTarget(self, dataFrame):
        Shiftedvar = str('previous ') + self.TARGETVAR
        dataFrame[Shiftedvar] = dataFrame[self.TARGETVAR].shift()
        return dataFrame


    def _fillNA(self, dataFrame):
        dataFrame.fillna(method = 'ffill', inplace = True)
        dataFrame.fillna(method = 'bfill', inplace = True)          


    def _preprocess(self, dataFrame):
        dataFrame.drop(self.TIMEVAR, axis = 1, inplace = True)
        if (self.PASTVAR):
            dataFrame = self._addPreviousTarget(dataFrame)
        dataFrame = dataFrame.apply(pd.to_numeric)
        self._fillNA(dataFrame)
        return dataFrame


    def predict(self, dataFrame: pd.DataFrame):
        self.ewmaAlpha = 0.4
        self.TIMESTAMP = pd.to_datetime(dataFrame[self.TIMEVAR], infer_datetime_format=True)
        dataFrame = self._preprocess(dataFrame)
        X, y = self._get_x_y(dataFrame)
        columns = ['Timestamp','Target', 'TargetEWMA']
        columns.append(self.MODELNAME)
        self.predictions = pd.DataFrame(columns = columns)
        self.predictions['Timestamp'] = self.TIMESTAMP
        self.predictions['Target'] = y
        self.predictions['TargetEWMA'] = y.ewm(alpha = self.ewmaAlpha).mean()
        self.predictions[self.MODELNAME] = self.model.predict(X)
        self.predictions['Alarm'] = self.predictions.apply(lambda row: 1 if np.abs(row[self.MODELNAME] - row['TargetEWMA']) > 5 else 0, axis = 1)
        predictions = self.predictions
        predictions['Timestamp'] = predictions['Timestamp'].astype(str)
        return self.predictions


    def _fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3) #random_state = 5,
        self.model.fit(X_train, y_train)
        score = {'training': np.around(self.model.score(X_train, y_train), decimals=3), 'test': np.around(self.model.score(X_test, y_test), decimals = 3)}
        return score
    

    def trainModel(self, parameters: dict = {'RandomForest': {'trees': 20, 'max_depth': 10}, 'KNeighbors': {'n_neighbors': 20}, 'Lasso': {'alpha': 2.5}}):
        X, y = self._get_x_y(self.TRAIN_DF)
        self.predictors = X.columns
        if self.MODELNAME == 'RandomForest':
            self.model = (RandomForestRegressor(n_jobs = -1, n_estimators = parameters['RandomForest']['trees'], max_depth = parameters['RandomForest']['max_depth']))
        elif self.MODELNAME == 'KNeighbors':
            self.model = (KNeighborsRegressor(n_jobs = -1, n_neighbors = parameters['KNeighbors']['n_neighbors']))
        elif self.MODELNAME == 'Lasso':
            self.model = (Lasso(alpha = parameters['Lasso']['alpha']))
        self.score = self._fit(X, y)
        return self.model, self.score


    def feature_importances(self):
        if self.MODELNAME == 'RandomForest':
            return pd.DataFrame(self.model.feature_importances_, index = self.predictors, columns=['importance']).sort_values('importance', ascending=False)
        return 'ERROR. This function is only valid for Random Forest models. Please create ones'
  
    def plot(self, start: int = 0, end: int = 0, title: str = 'Predictions', fileName: str = 'plot.png', dpi: int = 200, size: list = [12,5]):
        if end == 0:
            end = self.predictions.index.size
        labels = ['Target', 'Target EWMA ' + r'$\alpha=$' + str(self.ewmaAlpha)]
        labels.append(self.MODELNAME + r' $R^2 = $ {0:.3f}'.format(self.score['test']))
        toPlot = [self.predictions.drop('Timestamp', axis = 1), labels]
        size[1] *= (len(labels) - 2)
        self.plotSeries(toPlot, self.TIMESTAMP, start, end, title, fileName, size)
        
        
    def plotSeries(self, data, ticks, start, stop, title, fileName, size):
        numberOfTicks = 15
        toCompare = 'TargetEWMA' ##Target
        filteredData = data[0].iloc[start:stop]
        step = int(np.ceil((stop - start)/numberOfTicks))
        x = np.arange(filteredData.index.size)
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(size[0], size[1])
        fig.suptitle(title, fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True)
        i = 0
        my_xticks = ticks[start:stop:step]
        columns = filteredData.columns.values
        lastCol = columns.size - 1
        dif = 5
        for column in filteredData:
            if(i > 1):
                ax = plt.subplot(len(data[1]) - 2, 1, i-1)
                plt.ylim([35,80])
                if (i != lastCol):
                    plt.xticks(np.arange(0, filteredData.index.size, step))
                    plt.setp(ax.get_xticklabels(), visible = False)
                else:
                    plt.xticks(np.arange(0, filteredData.index.size, step), my_xticks, rotation=80)
    #            dif = np.std(filteredData[toCompare])
                plt.plot(x, filteredData[toCompare], label = data[1][1], c = 'lime')
                plt.plot(x, filteredData[column], label = data[1][i], c = 'chocolate')
                ax.fill_between(x, filteredData[toCompare], filteredData[column], 
                                where=filteredData[toCompare] - filteredData[column] > dif,
                                facecolor= 'wheat', interpolate=True) #(0.18,0.38,0.6,0.65)
                plt.legend()
            i += 1
        plt.tight_layout(rect = (0,0,1,0.96))
        plt.savefig(fileName, dpi=200)
        plt.show()



