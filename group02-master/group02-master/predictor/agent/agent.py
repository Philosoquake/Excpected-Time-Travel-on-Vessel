import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

class VesselTravelTimePredictor:
    def __init__(self):
        self.model = RandomForestRegressor()
        
    def load_data(self, filepath):
        try:
            self.data = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return

    def train(self):
        self._preprocess()
        self._train_model()
    
    def predict(self, new_data):
        if isinstance(new_data, pd.DataFrame):
            prediction = self.model.predict(new_data)
            print('The model predicts ', prediction)
        else:
            print('Input data should be pandas DataFrame.')

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        try:
            self.model = load(path)
        except FileNotFoundError:
            print(f"Model not found: {path}")
            return

    def _preprocess(self):
        try:
            x = self.data[['EndLatitude', 'EndLongitude', 'Latitude', 'Longitude', 'time', 'shiptype', 'SOG', 'mean_dir_sin', 'mean_dir_cos', 'mean_size', 'time_remaining']].copy()
            y = self.data['EndTime'].copy()
        except KeyError as e:
            print(f"KeyError: {e}")
            return

        # encode categorical data
        if x['shiptype'].dtype == 'object':
            le = LabelEncoder()
            x['shiptype'] = le.fit_transform(x['shiptype'])

        # split data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def _train_model(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.x_test)
        print ('The Mean Absolute Error is:', mean_absolute_error(self.y_test, predictions))


# Usage
#predictor = VesselTravelTimePredictor()
#predictor.load_data("./resources/felixstowe_rotterdam/felixstowe_rotterdam_cleaned.csv")  # load data from CSV file 1
#predictor.preprocess()
#predictor.train()
#predictor.load_data("./resources/rotterdam_hamburg/rotterdam_hamburg_cleaned.csv")  # load data from CSV file 2
#predictor.preprocess()
#predictor.train()
#new_data = predictor.data[['EndLatitude', 'EndLongitude', 'Latitude', 'Longitude', 'time', 'shiptype', 'SOG', 'mean_dir_sin', 'mean_dir_cos', 'mean_size', 'time_remaining']].tail(1)
#predictor.predict(new_data)
#predictor.save_model('predictor/agent/trainedmodel1.pkl')  # save the trained model
#predictor.load_model('predictor/agent/trainedmodel1.pkl')  # load the model
