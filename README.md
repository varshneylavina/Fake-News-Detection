# Fake-News-Detection

## Abstract
Fake news has become a prevalent issue in today's digital age, spreading misinformation and creating social and political unrest. The detection of fake news has thus become a crucial task to ensure the dissemination of accurate information. In this project, fake news detection  I have used an ensemble method. The ensemble method combines the predictions of multiple individual classifiers, leveraging their collective knowledge to make more accurate decisions.

## Inspiration
In recent years, we've seen how misinformation can have serious consequences, from influencing political campaigns to fueling public health crises. We believe that combatting fake news is an important way to promote responsible journalism and protect the public.

## What it does
The fake news detector project uses Flask, a Python web framework, to build a web app that allows users to check whether a given news article is real or fake. The app takes in a news article input by the user and applies machine learning models to predict its authenticity. The models used are loaded from pre-trained files stored in the backend server and made using different models in a Python notebook file, where we explored the runtime and square error of different models on a Fake News dataset. The app presents the prediction results to the user. The text entered by the user is stored in a CSV file on the server for future analysis. 

## How I built it
I have built the fake news detector using Flask, a web application framework for Python, as the backend server. The server receives user input, which is the text of the news article and passes it through several machine-learning models to predict whether the news is real or fake. I have used the Natural Language Toolkit (NLTK) library to preprocess the text and extract its features. The models we used include Logistic Regression, MultnomialNB, SVM, Decision Tree Classifier,Adaboost Classifier,Xgboost Classifier,Random Forest Classifier, and Ensemble method. We also used a TF-IDF vectorizer to convert the text to numerical data that can be fed into the models.

 A simple user interface has been built using HTML, CSS, and JavaScript. The interface allows users to enter the text of the news article, submit it to the backend server for prediction, and see the result.The user input data  is stored in a CSV file that can be used to improve the performance of the machine learning models over time. 

## Challenges
During the development of our fake news classifier, we faced several challenges. One of the main challenges was the cleaning the dataset, as the dataset had more Nan value in both fields title as well as text.

Another challenge I faced was the selection of the right set of features for the classifier. We experimented with different feature sets and finally decided to use the TfidfVectorizer to convert the text data to a numerical representation. However, fine-tuning the hyperparameters of the vectorizer was a time-consuming task, as it required several iterations to find the optimal values.

## Accomplishments
I am proud to have successfully built and deployed a machine learning model to detect fake news with an accuracy of over 96.3% on our test dataset that is taken from kaggle. I have put in countless hours of research, experimentation, and collaboration to develop and fine-tune the model to achieve this high level of accuracy. I am also proud to have created a user-friendly interface for our model using open online sources l̥ike tailwind , tailblocks,bootstrap etc , which allows users to enter text and get a prediction on whether it's fake or real news.

In addition, I am proud of the technical skills I have  developed throughout this project, including proficiency in data cleaning, natural language processing, and machine learning.  Finally, I am proud to have contributed to the ongoing effort to combat the spread of misinformation, which has become a critical issue in today's society.

## What I learned
Throughout this project, we learned how to build a machine learning model to classify news articles as real or fake. I have used the different differnt classifier algorithm and a TfidfVectorizer for feature extraction. Using these classifier a comparitive stuy has been done. I have  also learned how to preprocess text data by removing special characters and punctuation, lowercasing the text, and removing stopwords.

In addition, I have learned how to use get dataset from Kaggle and also learned how to use the pandas library to manipulate and combine datasets.

I have also learned how to evaluate the performance of the model using metrics such as accuracy and confusion matrix. And how to save the trained model and the vectorizer to disk using the joblib library.

Overall, this project provided me with valuable experience in working with text data and building a machine learning model to classify it. I have gained insights into the preprocessing techniques, feature extraction, model selection, and evaluation that are essential in building a successful machine learning model.

## What's next for Fake News Detector
Improve accuracy: The current detector version is based on a ensemble voting hard method. While it has shown decent accuracy, there may be more advanced models that could perform better. Developers could explore using more complex models, such as neural networks or other ensemble methods that combine multiple models for improved accuracy.

Expand to new domains: The Fake News Detector is trained on news articles from a specific dataset. However, it could be expanded to work with other text types, such as social media posts or product reviews. This would require retraining the model on a new dataset but could greatly expand the project's scope.

Integrate with news apps: The Fake News Detector could be integrated with news apps or websites to help users evaluate the credibility of articles they are reading. This could help users make more informed decisions about what they read and share, potentially reducing the spread of fake news.

Develop a browser extension: A browser extension could be created that allows users to quickly evaluate the credibility of news articles they come across while browsing the web. This would make the Fake News Detector more accessible to a broader audience and help combat the spread of fake news online.


## Disclaimer
Disclaimer: Fake News Detector App

Please read this disclaimer carefully before using the Fake News Detector App ("the App"). By using the App, you agree to the terms and conditions outlined below.

Information Accuracy: The App is designed to assist users in identifying potentially fake news articles, but it may not guarantee 100% accuracy. The authenticity assessments provided by the App are based on automated algorithms and available data at the time of analysis. While we strive to present reliable and up-to-date information, there may be limitations and inaccuracies in the results.

Educational Tool: The App is intended to raise awareness about the challenges of fake news and enhance media literacy skills. It does not constitute professional advice or endorse any particular political viewpoint, ideology, or news source.

User Responsibility: Users of the App are responsible for evaluating the information presented and should not solely rely on the App's assessments. We encourage users to cross-check information, verify facts from multiple sources, and exercise critical thinking when evaluating news and information.

Data Privacy and Security: The App may collect certain user data for the purpose of improving the accuracy and functionality of the service. We are committed to protecting user privacy and will handle data in accordance with our Privacy Policy.

Acceptance of Disclaimer: By using the App, you acknowledge that you have read, understood, and agreed to this disclaimer.

## Load_Model

final_model.joblib : https://drive.google.com/file/d/1Ea3HOgoPmFXBsMwCp6uSt8__GUqRdKJH/view?usp=sharing

vectorizer.joblib : https://drive.google.com/file/d/1rotlvnPRSw3YTDiSCutqJov72IYfFwJC/view?usp=sharing

## Built With:
css
flask
html5
javascript
natural-language-processing
nltk
numpy
pandas
python
scikit-learn
