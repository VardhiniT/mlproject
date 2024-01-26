import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            # data_scaled=preprocessor.transform(features)
            preds=model.predict(features)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        TEMPERATURE_departures,
        TEMPERATURE_arrival,
        Aircraft_Maintainance,
        Air_Route_Traffic,
        Heavy_Precipitation,
        WIND_departures,
        WIND_arrival,
        delay_departure):

        self.TEMPERATURE_departures = TEMPERATURE_departures

        self.TEMPERATURE_arrival = TEMPERATURE_arrival

        self.Aircraft_Maintainance = Aircraft_Maintainance

        self.Air_Route_Traffic =  Air_Route_Traffic

        self.Heavy_Precipitation = Heavy_Precipitation

        self.WIND_departures = WIND_departures

        self.WIND_arrival = WIND_arrival

        self.delay_departure = delay_departure

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "TEMPERATURE_departures": [self.TEMPERATURE_departures],
                "TEMPERATURE_arrival": [self.TEMPERATURE_arrival],
                "Aircraft_Maintainance": [self.Aircraft_Maintainance],
                "Air_Route_Traffic": [self.Air_Route_Traffic],
                "Heavy_Precipitation": [self.Heavy_Precipitation],
                "WIND_departures": [self.WIND_departures],
                "WIND_arrival": [self.WIND_arrival],
                "delay_departure": [self.delay_departure]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

