from flask import Flask, render_template,request
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
app=Flask(__name__)
CORS(app)

#car=pd.read_csv("Cleaned_car.csv")

tfidf = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("model.pkl",'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


















@app.route('/')
def index():
    #companies = sorted(car["company"].unique())
    #car_models = sorted(car["name"].unique())
    #year = sorted(car["year"].unique(),reverse=True)
    #fuel_type = sorted(car["fuel_type"].unique())
    #companies.insert(0,"Select Company")

    return render_template('index.html')##companies=companies,car_models=car_models,year=year, fuel_type=fuel_type )

@app.route('/predict',methods=['POST'])
def predict():
    #company=request.form.get('company')
    #car_model = request.form.get('car_model')
    #year = int(request.form.get('year'))
    #fuel_type=request.form.get('fuel_type')
    input_data = str(request.form.get('email'))

    #1. preprocessing
    transformed_email = transform_text(input_data )
    #2. vectorize
    vector_input = tfidf.transform([transformed_email])
    #3. predict
    result=model.predict(vector_input)[0]
    #4. Display
    if result ==1:
        ans="It is a spam"
    else:
        ans="Not a spam"


    #prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(ans)

    return  ans

if __name__=="__main__":
    app.run(debug=True)
