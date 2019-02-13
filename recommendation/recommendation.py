import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, validators
from wtforms.validators import DataRequired, Email
import io
from flask_restful import Resource, Api
import string
import re
import pickle
from flask_jsonpify import jsonpify

DEBUG = True
app = Flask(__name__)
# app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'abcdefgh'
api = Api(app)

class TextFieldForm(FlaskForm):
    text = StringField('Document Content', validators=[validators.data_required()])


class Main(Resource):
    def __init__(self):
        print('Executing Main Funciton')
        self.df = pd.read_csv('dataset_food_online.txt', encoding="ISO-8859-1")
        self.create_model()

    def clean_text(self, text):
        ## Remove puncuation
        text = text.translate(string.punctuation)

        ## Convert words to lower case and split them
        text = text.lower().split()

        ## Remove stop words
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops and len(w) >= 3]

        text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def matrix_factorization(self,R, P, Q, steps=1, gamma=0.001, lamda=0.02):
        for step in range(steps):
            for i in R.index:
                for j in R.columns:
                    if R.loc[i, j] > 0:
                        eij = R.loc[i, j] - np.dot(P.loc[i], Q.loc[j])
                        P.loc[i] = P.loc[i] + gamma * (eij * Q.loc[j] - lamda * P.loc[i])
                        Q.loc[j] = Q.loc[j] + gamma * (eij * P.loc[i] - lamda * Q.loc[j])
            e = 0
            print('Hello')
            for i in R.index:
                for j in R.columns:
                    if R.loc[i, j] > 0:
                        # Sum of squares of the errors in the rating
                        e = e + pow(R.loc[i, j] - np.dot(P.loc[i], Q.loc[j]), 2) + lamda * (
                                    pow(np.linalg.norm(P.loc[i]), 2) + pow(np.linalg.norm(Q.loc[j]), 2))
            print(e)
            if e < 0.001:
                break

            print(step)
        return P, Q


    def create_model(self):
        self.P, self.Q = self.preprocessing()
        self.P, self.Q = self.matrix_factorization(self.userid_rating_matrix, self.P, self.Q, steps=1, gamma=0.001, lamda=0.02)
        print(self.P)

    def preprocessing(self):
        self.yelp_data = self.df[['business_id', 'user_id', 'stars', 'text']]

        self.yelp_data['text'] = self.yelp_data['text'].apply(self.clean_text)

        self.userid_df = self.yelp_data[['user_id', 'text']]
        self.business_df = self.yelp_data[['business_id', 'text']]

        self.userid_df = self.userid_df.groupby('user_id').agg({'text': ' '.join})
        self.business_df = self.business_df.groupby('business_id').agg({'text': ' '.join})


        # userid vectorizer
        self.userid_vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=1000)
        self.userid_vectors = self.userid_vectorizer.fit_transform(self.userid_df['text'])


        # Business id vectorizer
        self.businessid_vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=1000)
        self.businessid_vectors = self.businessid_vectorizer.fit_transform(self.business_df['text'])

        # # Matrix Factorization
        self.userid_rating_matrix = pd.pivot_table(self.yelp_data, values='stars', index=['user_id'], columns=['business_id'])


        self.P = pd.DataFrame(self.userid_vectors.toarray(), index=self.userid_df.index, columns=self.userid_vectorizer.get_feature_names())
        self. Q = pd.DataFrame(self.businessid_vectors.toarray(), index=self.business_df.index,
                         columns=self.businessid_vectorizer.get_feature_names())

        return self.P, self.Q


class Flask_Work(Resource):
    def __init__(self):
        pass

    def clean_text(self, text):
        ## Remove puncuation
        text = text.translate(string.punctuation)

        ## Convert words to lower case and split them
        text = text.lower().split()

        ## Remove stop words
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops and len(w) >= 3]

        text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)

    def post(self):
        f = open('recommendation.pkl', 'rb')
        P, Q, userid_vectorizer = pickle.load(f), pickle.load(f), pickle.load(f)
        print('in vect')
        sentence = request.form['Input Text']
        print(sentence)
        test_df = pd.DataFrame([sentence], columns=['text'])
        test_df['text'] = test_df['text'].apply(self.clean_text)
        test_vectors = userid_vectorizer.transform(test_df['text'])
        test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index,
                                 columns=userid_vectorizer.get_feature_names())

        predict_item_rating = pd.DataFrame(np.dot(test_v_df.loc[0], Q.T), index=Q.index, columns=['Rating'])
        top_recommendations = pd.DataFrame.sort_values(predict_item_rating, ['Rating'], ascending=[0])[:3]
        print(top_recommendations)


        df_list = top_recommendations.index.tolist()
        JSONP_data = jsonpify(df_list)
        return JSONP_data

        return make_response(render_template('index.html'), 200, JSONP_data)


api.add_resource(Flask_Work, '/')


if __name__ == '__main__':
    # Main()
    app.run(host='127.0.0.1', port=4000, debug=True)
