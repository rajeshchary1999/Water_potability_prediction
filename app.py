import numpy as np
from flask import Flask,  request, jsonify, render_template

import pickle


#load the pickle model
model = pickle.load(open("model.pkl","rb"))
predictions_classes = {0: "Not Potable", 1: "Potable"}

app = Flask(__name__)



@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/", methods=['POST','GET'])

def predict():
    
    if request.method == 'POST':
        #access the data from form
        #ph
        Ph = float(request.form["ph"])
        Hardness = float(request.form["Hardness"])
        Solids = float(request.form["Solids"])
        Chloramines = float(request.form["Chloramines"])
        Sulfate = float(request.form["Sulfate"])
        Conductivity = float(request.form["Conductivity"])
        Organic_carbon = float(request.form["Organic_carbon"])
        Trihalomethanes = float(request.form["Trihalomethanes"])
        Turbidity = float(request.form["Turbidity"])
        # get prediction
        
        features = [

            [Ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]

    
    
    
    results =model.predict(features)
    results = results.tolist()[0]




    

    
    pred = {"Predicted quality": predictions_classes[results]}



    return pred





if __name__ == "__main__":
    app.run(debug=True)


