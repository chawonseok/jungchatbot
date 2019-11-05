# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:48:47 2019

@author: Chacrew
"""
import pandas as pd
from ast import literal_eval
import urllib3
import json
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline
import time
from cdqa.retriever import BM25Retriever

def ETRI_POS_Tagging(text) :
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
    accessKey = "14af2341-2fde-40f3-a0b9-b724fa029380"
    analysisCode = "morp"
    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text,
            "analysis_code": analysisCode
        }
    }
    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    return Pos_extract(response)
	
def Pos_extract(Data) :
    Noun = []
    Extract_a = json.loads(str(Data.data,"utf=8"))['return_object']['sentence']
    for i in range(len(Extract_a)) : 
        Extract_b = dict(Extract_a[i])
        for i in range(len(Extract_b['morp'])) : 
            if (Extract_b['morp'][i]['type'] =='NNG' or Extract_b['morp'][i]['type'] =='NNP') or Extract_b['morp'][i]['type'] =='VV': 
                Noun.append(Extract_b['morp'][i]['lemma'])
    return " ".join(Noun)


#df = pd.read_csv('data/bnpp_newsroom_v1.1/jungchat_result_191015.csv',converters={'paragraphs': literal_eval})
df = pd.read_csv('data/bnpp_newsroom_v1.1/jungchat_result_191015_fix.csv',converters={'paragraphs': literal_eval})
#유사도를 구할 retriever 선언 및 훈련
retriever = BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)
#df의 content열을 선택해서 훈련
retriever.fit(df)

#학습된 bert모델 불러오기
cdqa_pipeline = QAPipeline(reader='models/bert_qa_korquad_vCPU.joblib')
df = filter_paragraphs(df,min_length=10,max_length=100)

#첫번째 입력을위한 초기화
best_idx_scores=''
while 100:
    query=input('입력창:')
    #정책과 질문의 유사도가 1보다 높으면 해당 정책의 문서를 선택하여 재학습
    #질문의 유사도가 낮으면 해당정책에 대해 질의응답 
    print(list(list(retriever.predict(ETRI_POS_Tagging(query)).values())[0])[0])
    if list(list(retriever.predict(ETRI_POS_Tagging(query)).values())[0])[0]>=4. or not best_idx_scores:
        #질문과 유사도가 높은 정책의 인덱스 예측 
        best_idx_scores = retriever.predict(ETRI_POS_Tagging(query))
        #청년정책에서 선택된 문서학습
        cdqa_pipeline.fit_retriever(df.loc[best_idx_scores.keys()].head(1))
    #정답예측
    prediction = cdqa_pipeline.predict(query)
    print('paragraph: {}\n'.format(prediction[0]))
    print('paragraph: {}\n'.format(prediction[1]))
    print('paragraph: {}\n'.format(prediction[2]))
    print('paragraph: {}\n'.format(prediction[3]))


