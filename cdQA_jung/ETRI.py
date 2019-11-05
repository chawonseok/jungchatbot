import urllib3
import json

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
        for j in range(len(Extract_b['morp'])) : 
            if Extract_b['morp'][j]['type'] =='NNG' or Extract_b['morp'][j]['type'] =='NNP': 
                Noun.append(Extract_b['morp'][j]['lemma'])
    return Noun