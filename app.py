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

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        #ph
        ph = int(request.form["ph"])
        Hardness = int(request.form["Hardness"])
        Solids = int(request.form["Solids"])
        Chloramines = int(request.form["Chloramines"])
        Sulfate = int(request.form["Sulfate"])
        Conductivity = int(request.form["Conductivity"])
        Organic_carbon = int(request.form["Organic_carbon"])
        Trihalomethanes = int(request.form["Trihalomethanes"])
        Turbidity = int(request.form["Turbidity"])
        # get prediction


        inf_features = [
            [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
        results = model.predict(inf_features)
        results = results.tolist()[0]

        resp = {"Predicted quality": predictions_classes[results]}


        return resp



if __name__ == "__main__":
    app.run(debug=True)

