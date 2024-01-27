from flask import Flask,request,render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            TEMPERATURE_departures=request.form.get('TEMPERATURE_departures'),
            TEMPERATURE_arrival=request.form.get('TEMPERATURE_arrival'),
            Aircraft_Maintainance=request.form.get('Aircraft_Maintainance'),
            Air_Route_Traffic=request.form.get('Air_Route_Traffic'),
            Heavy_Precipitation=request.form.get('Heavy_Precipitation'),
            WIND_departures=float(request.form.get('WIND_departures')),
            WIND_arrival=float(request.form.get('WIND_arrival')),
            delay_departure=float(request.form.get('delay_departure'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        


