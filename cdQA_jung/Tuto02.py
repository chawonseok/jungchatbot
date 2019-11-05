# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:12:05 2019

@author: Chacrew
"""

# -*- coding: utf-8 -*-

import requests
from flask import Flask, request, Response
from flask_ngrok import run_with_ngrok
import pandas as pd
from ast import literal_eval
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline


API_KEY = '936714777:AAGFCBbeOAClrTsgmMMOsYG3HkaV7Ck5p-w'

app = Flask(__name__)
run_with_ngrok(app)
df = pd.read_csv('data/bnpp_newsroom_v1.1/jungchat_result.csv',converters={'paragraphs': literal_eval})
cdqa_pipeline = QAPipeline(reader='models/bert_qa_korquad_vCPU.joblib')
cdqa_pipeline.fit_retriever(df)

def parse_message(message):
    chat_id = message['message']['chat']['id']
    msg = message['message']['text']

    return chat_id, msg


def send_message(chat_id,query):
    url = 'https://api.telegram.org/bot{token}/sendMessage'.format(token=API_KEY)
    # 변수들을 딕셔너리 형식으로 묶음


    prediction = cdqa_pipeline.predict(query)
    params = {'chat_id': chat_id, 'text':prediction[2]}

    # Url 에 params 를 json 형식으로 변환하여 전송
    # 메세지를 전송하는 부분
    response = requests.post(url, json=params)
    print(response)
    return response


# 경로 설정, URL 설정
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        message = request.get_json()

        # parse_message 함수는 두가지 return 값을 가진다 (chat_id, msg)
        # 순서대로 chat_id, msg의 변수로 받아준다.
        chat_id, msg = parse_message(message)

        # send_message 함수에 두가지 변수를 전달
        send_message(chat_id,msg)

        # 여기까지 오류가 없으면 서버상태 200 으로 반응
        return Response('ok', status=200)
    else:
        return 'Hello World!'


@app.route('/about')
def about():
    return 'About page'


if __name__ == '__main__':
   # app.run(port=80, host='127.0.0.1')
   print('안녕하세요 서울시 정책챗봇 청명이입니다. \n무엇을 도와드릴까요?\n') 
   app.run()
