from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application =Flask(__name__)
app=application
##Route for a home page

@app.route("/")
def indx():
    return render_template('index.html')
@app.route('/predictdata',method=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else :
        data=CustomData

