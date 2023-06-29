from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import nltk
import csv
import re
import string 

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder='')

# Function for Cleaning the User provided text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [stemmer.stem(word) for word in tokens if word not in stopwords]
    return text

# Load model and vectorizer
tfidf_vectorizer ='vectorizer1.pkl'
Ensemble_model ='finalized_model1.pkl'


model_path = os.path.join(basedir, Ensemble_model)
tfidfvect_path = os.path.join(basedir, tfidf_vectorizer)


tfidfvect = pickle.load(open(tfidfvect_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))


stopwords = nltk.corpus.stopwords.words("english")
stemmer = nltk.PorterStemmer()



# Function for prediction
def predict(text):
    body_len = len(text) - text.count(" ")
    punct_percent = count_punct(text)
    
    X = tfidfvect.transform([text]).astype(float)
    X = pd.concat([
        pd.DataFrame({"body_len": [body_len], "punct%": [punct_percent]}).reset_index(drop=True),
        pd.DataFrame(X.toarray())
    ], axis=1)
    X.columns = X.columns.astype(str)
    predict_value = model.predict(X)
    if predict_value < 0.5:
        return "FAKE"
    else:
        return "REAL"
    

# Function for puntuation counting 
def count_punct(text):
    non_space_count = len(text) - text.count(" ")
    if non_space_count == 0:
        return 0  # or any other appropriate value
    else:
        count = sum([1 for char in text if char in string.punctuation])
        return round(count / non_space_count, 3) * 100    

# function to add user input text for futher use 
def add_text_CSV(text,classification):
    # with open('text.csv', 'a', encoding='utf-8', newline='') as f:
    with open('text.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([text,classification])
 
# Build functionalities
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html', methods=['GET'])
def home1():
    return render_template('index.html')

@app.route("/prediction.html",methods=['GET','POST'])
def prediction():
    if request.method=="POST":
        # if request.form["Submit_B"]=="newsContent":
        news = str(request.form['content'])

        result = predict(news)
        add_text_CSV(news, result)

        return render_template("prediction.html", prediction_text=" News is {} ".format(result))
        
        
    else:
        return render_template("prediction.html",prediction_text ="")

@app.route('/favicon.ico')
def get_favicon():
    return app.send_static_file('favicon.ico')


if __name__ == "__main__":
    app.run()

