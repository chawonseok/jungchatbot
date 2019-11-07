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
cdqa_pipeline = QAPipeline(reader='bert_qa_korquad_vCPU.joblib')
retriever = BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)
retriever_temp= BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)
retriever.fit(df)
df = filter_paragraphs(df,min_length=10)
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
    if max(retriever.predict(POS_content).values())>=1.5 or not best_idx_scores:
        print(retriever.predict('장애 정책 리스트'),POS_content)
        print(ETRI_POS_Tagging('장애인'))
        if best_idx_scores:
            if max(retriever_temp.predict(POS_content).values())<1.5:
                best_idx_scores=retriever.predict(POS_content)
            else:
                if abs(max(retriever_temp.predict(POS_content).values()))-abs(max(retriever.predict(POS_content).values()))>3:
                    best_idx_scores = retriever.predict(POS_content)
        if not best_idx_scores:
            best_idx_scores=retriever.predict(POS_content)
#            print('다문화나와라!{}'.format(best_idx_scores))
        para= df['paragraphs'][list(best_idx_scores.keys())[0]]
        POS_para=[]
        for i in para:
            print(i,ETRI_POS_Tagging(str(i)))
            POS_para.append(ETRI_POS_Tagging(str(i)))
       # print(para,POS_para)
        retriever_temp.fit(pd.DataFrame({'content':POS_para}))
        sentence_idx_scores=retriever_temp.predict(POS_content)
        if list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(POS_content).values())[0])[0]>=1.:
            changed_text = text_tranform(''.join(df.loc[list(best_idx_scores.keys())[0]]['paragraphs']))
            dataSend=changed_text
            return jsonify(make_query(dataSend))
        elif list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(POS_content).values())[0])[0]<1.:
 #           print('first')  
            dataSend=ETRI_wiki(content)
            return jsonify(make_query(dataSend)) 
    if max(retriever_temp.predict(POS_content).values())<1.5 and max(retriever.predict(POS_content).values())<0.5:
 #       print('second')  
        dataSend=ETRI_wiki(content)
        return jsonify(make_query(dataSend))
    cdqa_query,validity=ETRI_korBERT(' '.join(list(df.loc[best_idx_scores.keys()].head(1)['paragraphs'])[0]),content)
    para=df['paragraphs'][list(best_idx_scores.keys())[0]]
    prediction = retriever_temp.predict(cdqa_query)
    dataSend =para[list(prediction)[0]]
    print("time :", time.time() - start)
    return jsonify(make_query(dataSend))

if __name__ == "__main__":
    app.run()
