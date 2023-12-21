from multiprocessing import Pool
import pandas as pd
from predictor.agent.agentt import VesselTravelTimePredictor

class VesselTravelTimePredictorAPI(VesselTravelTimePredictor):
    def predict_api(self, new_data):
        if isinstance(new_data, pd.DataFrame):
            prediction = self.model.predict(new_data)
            return prediction
        else:
            print('Input data should be pandas DataFrame.')
        return None

predictor = VesselTravelTimePredictorAPI()
predictor.load_model('predictor/agent/trainedmodel1.pkl')  # load the model

def worker(data):
    print(data)
    #df = pd.DataFrame(data)
   # df = pd.DataFrame([data],columns=['EndLatitude', 'EndLongitude', 'Latitude', 'Longitude', 'time', 'shiptype', 'SOG', 'mean_dir_sin', 'mean_dir_cos', 'mean_size', 'time_remaining']) 
    df = pd.DataFrame(data, index=[0])
    result = predictor.predict_api(df)
    return result

if __name__ == "__main__":
    pool = Pool(processes=4)  # create a pool of 4 processes

    # here data_list is a list of input data. Each item in the list is a set of data for one prediction.
    data_list = ['EndLatitude', 'EndLongitude', 'Latitude', 'Longitude', 'time', 'shiptype', 'SOG', 'mean_dir_sin', 'mean_dir_cos', 'mean_size', 'time_remaining'] 

    results = pool.map(worker, data_list)

    for result in results:
        print(result)
