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
def ETRI_wiki(text) :
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseQAnal"
    accessKey = "14af2341-2fde-40f3-a0b9-b724fa029380"
    text = text

    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    response_json = json.loads(str(response.data,"utf-8"))
    response_Extract_A = response_json['return_object']['orgQInfo']['orgQUnit']
    if not response_Extract_A['vTitles'] :
        content = "위키에 없는 문서입니다."
        return content
    response_Extract_B = response_Extract_A['vQTopic'][0]['vEntityInfo']
    response_Extract_C = response_Extract_B[0]['strExplain']
    return response_Extract_C


def ETRI_korBERT(text,query) :
    openApiURL = "http://aiopen.etri.re.kr:8000/MRCServlet"
    accessKey = "14af2341-2fde-40f3-a0b9-b724fa029380"
    question = query
    passage = text
     
    requestJson = {
    "access_key": accessKey,
        "argument": {
            "question": question,
            "passage": passage
        }
    }
     
    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    return json.loads(str(response.data,"utf-8"))['return_object']['MRCInfo']['answer']

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
    Extract_a = json.loads(str(Data.data,"utf-8"))['return_object']['sentence']
    for i in range(len(Extract_a)) : 
        Extract_b = dict(Extract_a[i])
        for i in range(len(Extract_b['morp'])) : 
            if (Extract_b['morp'][i]['type'] =='NNG' or Extract_b['morp'][i]['type'] =='NNP') or Extract_b['morp'][i]['type'] =='VV': 
                Noun.append(Extract_b['morp'][i]['lemma'])
    return " ".join(Noun)

df = pd.read_csv('data/bnpp_newsroom_v1.1/jungchat_result_191031.csv',converters={'paragraphs': literal_eval})

retriever = BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)
retriever_temp= BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)
retriever.fit(df)

df = filter_paragraphs(df,min_length=10)

cdqa_pipeline = QAPipeline(reader='models/bert_qa_korquad_vCPU.joblib')


best_idx_scores=''

while 100:
    query=input('입력창:')
    if query=='quit':
        break
    print(list(list(retriever.predict(ETRI_POS_Tagging(query)).values())[0])[0])
    if list(list(retriever.predict(ETRI_POS_Tagging(query)).values())[0])[0]>=1.5 or not best_idx_scores:
        if best_idx_scores:
            if max(retriever_temp.predict(ETRI_POS_Tagging(query)).values())<1.5:
                best_idx_scores = retriever.predict(ETRI_POS_Tagging(query))
        if not best_idx_scores:
            best_idx_scores = retriever.predict(ETRI_POS_Tagging(query))
       # temp=best_idx_scores
        para= ','.join(df['paragraphs'][list(best_idx_scores.keys())[0]]).split(',')
        
        POS_para=[]
        for i in para:
            POS_para.append(ETRI_POS_Tagging(i))
        
        retriever_temp.fit(pd.DataFrame({'content':POS_para}))
        #if max(retriever_temp.predict(ETRI_POS_Tagging(query)).values()) > 1.5:
        #    best_idx_scores=temp
            
        sentence_idx_scores=retriever_temp.predict(ETRI_POS_Tagging(query))
        if list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(ETRI_POS_Tagging(query)).values())[0])[0]>=1.:
            #print(list(best_idx_scores.keys())[0])
            print('유사도가 있고 리스트를 조회하는 경우\n')
            print('{}\n\n{}\n\n{}'.format(df.loc[list(best_idx_scores.keys())[0]]['content'],ETRI_POS_Tagging(query),query))
            print(''.join(df.loc[list(best_idx_scores.keys())[0]]['paragraphs']))
            best_idx_scores=''
            continue#강빈이 코드에서는 필요없을것
            #return 강빈이 코드에서 리턴하면 될것     
        elif list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(ETRI_POS_Tagging(query)).values())[0])[0]<1.:           
            #첫번째 케이스를 대비한것
            print(ETRI_wiki(query))
            best_idx_scores=''
            continue
        
        print('0~8의 인덱스가 안나온 경우 \n'+df.loc[list(best_idx_scores.keys())[0]]['title'])#테스트를 위한
        if max(retriever_temp.predict(ETRI_POS_Tagging(query)).values())>max(retriever.predict(ETRI_POS_Tagging(query)).values()): pass
        else:
            cdqa_pipeline.fit_retriever(df.loc[best_idx_scores.keys()].head(1))
#
    kor_query=ETRI_korBERT(' '.join(list(df.loc[best_idx_scores.keys()].head(1)['paragraphs'])[0]),query)
    print('{}\n\n{}\n\n{}\n\n'.format(df.loc[list(best_idx_scores.keys())[0]]['content'],ETRI_POS_Tagging(query),query))
    
    
    #test=cdqa_pipeline.predict(query)
    #max(retriever_temp.predict(ETRI_POS_Tagging('지원방법')).values()),max(retriever.predict(ETRI_POS_Tagging('지원방법')).values())

    if max(retriever_temp.predict(ETRI_POS_Tagging(query)).values())<1.5 and max(retriever.predict(ETRI_POS_Tagging(query)).values())<1.5:
        print(ETRI_wiki(query))
        print('유사도가 낮어:'+str(max(sentence_idx_scores.values())))#위키로 처리하면 될듯
           
    else:
        prediction=cdqa_pipeline.predict(kor_query)
        print('cdqa 유사도 수치 '+str(prediction[3]))
        print(prediction[2])
   
#    print('paragraph: {}\n'.format(prediction[0]))
#    print('paragraph: {}\n'.format(prediction[1]))
#    print('paragraph: {}\n'.format(prediction[2]))
#    print('paragraph: {}\n'.format(prediction[3]))
 if max(retriever_temp.predict(ETRI_POS_Tagging(content)).values()) > 1.5 and best_idx_scores:
