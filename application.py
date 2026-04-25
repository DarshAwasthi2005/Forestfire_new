import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

ridge_model=pickle.load(open('Algorithms/ridge_model.pkl','rb'))
Standard_model=pickle.load(open('Algorithms/scaler.pkl','rb'))



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Date = float(request.form['Date'])
        Month = float(request.form['Month'])
        Year = float(request.form['Year'])
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
      
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        new_data=Standard_model.transform([[ Date , Month, Year, Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge_model.predict(new_data)
        return render_template('home.html', result=prediction[0])

    else :
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
