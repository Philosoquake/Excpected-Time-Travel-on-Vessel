from agent import VesselTravelTimePredictor

predictor = VesselTravelTimePredictor()
predictor.load_data("./resources/felixstowe_rotterdam/felixstowe_rotterdam_cleaned.csv")  # load data from CSV file 1
#predictor.preprocess()
predictor.train()
predictor.load_data("./resources/rotterdam_hamburg/rotterdam_hamburg_cleaned.csv")  # load data from CSV file 2
#predictor.preprocess()
predictor.train()
new_data = predictor.data[['EndLatitude', 'EndLongitude', 'Latitude', 'Longitude', 'time', 'shiptype', 'SOG', 'mean_dir_sin', 'mean_dir_cos', 'mean_size', 'time_remaining']].tail(1)
predictor.predict(new_data)
predictor.save_model('predictor/agent/trainedmodel1.pkl')  # save the trained model
predictor.load_model('predictor/agent/trainedmodel1.pkl')  # load the model
