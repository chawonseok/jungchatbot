from flask import Flask, request, jsonify
from ast import literal_eval
import pandas as pd
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from cdqa.retriever import BM25Retriever
from ETRI import *
import time
from khaiii_def import *
app = Flask(__name__)
df = pd.read_csv('jungchat_result_191102.csv',converters={'paragraphs': literal_eval})
cdqa_pipeline = QAPipeline(reader='bert_qa_korquad_vCPU.joblib')#모델을 불러온다
retriever = BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)#문서와의 유사도를 구하기위한 리트리버
retriever_temp= BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)#문장과의 유사도를 구하기 위한 리트리버
retriever.fit(df)#모든 문서의 내용을 담는다
df = filter_paragraphs(df)
best_idx_scores = ''

def text_tranform(text) :
    return '\n'.join(text.split(', '))

def make_query(text) :
    dataSend = {
          "version": "2.0",
          "template": {
             "outputs": [{
                    "simpleText":{
                       "text" : text}
               }]
           }
     }
    return dataSend
def extract_content() : 
    content = request.get_json()
    content = content['userRequest']
    content = content['utterance']
    content = content.rsplit("\n")[0]
    return content
@app.route('/message', methods=['POST'])

def Message():
    start = time.time() 
    content = extract_content()
    POS_content=ETRI_POS_Tagging(content)
    global best_idx_scores
    print(kai('테스트'))
    #문서와의 유사도가 있거나 처음 실행될때
    if max(retriever.predict(POS_content).values())>=1.5 or not best_idx_scores:
        print(retriever.predict('장애 정책 리스트'),POS_content)
        print(ETRI_POS_Tagging('장애인'))
        #이미 best_idx_scores가 있는 상태, 즉 2번째 질문이 들어 왔을때
        if best_idx_scores:
            #선택된 문서의 문장과 유사도 수치가 낮으면 문서를 재선정 
            if max(retriever_temp.predict(POS_content).values())<1.5:
                best_idx_scores=retriever.predict(POS_content)
            else:
                #질문에 대한 문서의 유사도와 문장의 유사도의 차이가 3보다 높으면 문서 재선정 =
                if abs(max(retriever_temp.predict(POS_content).values()))-abs(max(retriever.predict(POS_content).values()))>3:
                    best_idx_scores = retriever.predict(POS_content)
            #결국 if문의 조건에 해당하지 않으면 best_idx_scores는 변하지 않는다 즉 문서를 재활용한다
        #질문이 처음 들어왔을 경우
        if not best_idx_scores:
            #유사도가 높은 문서를 선정한다
            best_idx_scores=retriever.predict(POS_content)
        #선정된 문서의 문장을 형태소 분석
        para= df['paragraphs'][list(best_idx_scores.keys())[0]]
        POS_para=[]
        for i in para:
            print(i,ETRI_POS_Tagging(str(i)))
            POS_para.append(ETRI_POS_Tagging(str(i)))
        #형태소 분석된 문장들을 retriever_temp에 담는다
        retriever_temp.fit(pd.DataFrame({'content':POS_para}))
        #질문과 유사도가 높은 문장을 구한다
        sentence_idx_scores=retriever_temp.predict(POS_content)
        #선택된 문서의 인댁스가 8 미만이고 질문과 문서의 유사도가 있을때(리스트를 조회하는 질문을 했을때) 
        if list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(POS_content).values())[0])[0]>=1.:
            changed_text = text_tranform(''.join(df.loc[list(best_idx_scores.keys())[0]]['paragraphs']))
            dataSend=changed_text
            return jsonify(make_query(dataSend))
        #상관없는 질문이 들어오면 best_idx_scores의 인덱스는 0 고정 그리고  질문과 문서의 유사도가 낮을때
        elif list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(POS_content).values())[0])[0]<1.:

            dataSend=ETRI_wiki(content)
            return jsonify(make_query(dataSend)) 
    #문장과의 유사도 수치가 없고 문서와 유사도 수치도 없을떄 wikiQA로 처리
    if max(retriever_temp.predict(POS_content).values())<1.5 and max(retriever.predict(POS_content).values())<0.5:
 
        dataSend=ETRI_wiki(content)
        return jsonify(make_query(dataSend))
    #ETRI KORBERT API에 문서와 질문 전달
    cdqa_query,validity=ETRI_korBERT(' '.join(list(df.loc[best_idx_scores.keys()].head(1)['paragraphs'])[0]),content)
    para=df['paragraphs'][list(best_idx_scores.keys())[0]]#유사도가 높은 문서
    prediction = retriever_temp.predict(cdqa_query)#KORBERT의 답과 유사도가 높은 문장의 인댁스를 뽑는다
    dataSend =para[list(prediction)[0]]#정답문장
    print("time :", time.time() - start)
    return jsonify(make_query(dataSend))

if __name__ == "__main__":
    app.run()
