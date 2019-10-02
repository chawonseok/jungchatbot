from konlpy.tag import Okt
import pandas as pd
import tensorflow as tf
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from configs import DEFINES
from tqdm import tqdm

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

# 판다스를 통해서 데이터를 불러와 학습 셋과 평가 셋으로 나누어 그 값을 리턴한다.
def load_data():
    data_df = pd.read_csv(DEFINES.data_path, header=0,encoding='cp949')
    question, answer = list(data_df['Q']), list(data_df['A'])
    train_input, eval_input, train_label, eval_label = \
        train_test_split(question, answer, test_size=0.01, random_state=42)
    return train_input, train_label, eval_input, eval_label

# 형태소 분석
# 감성분석이나 문서 분류에는 형태소 분석이 필요하다. 하지만 챗봇에 형태소 분석을 적용하면
# 형태소로 답변하게 된다. 따라서 챗봇에는 형태소 분석을 적용하기 곤란하다.
# 챗봇에 형태소 분석을 적용하려면 답변시 형태소를 이용하여 문장을 generation하는 기술이
# 필요하다. ex : 언어론적 기술인 grammar, parse tree, etc.
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data

# 인코더, 디코더의 입력과 출력 데이터를 생성한다.
# 디코더 입력과 타켓에는 앞 뒤에 STD, END가 들어간다.
# 예시:
# DEFINES.max_sequence_length = 10 인 경우
# 인코더 입력 : "가끔 궁금해" -> [9310, 17707, 0, 0, 0, 0, 0, 0, 0, 0]
# 디코더 입력 : "그 사람도 그럴 거예요" -> [STD, 20190, 4221, 13697, 14552, 0, ...]
# 디코더 타켓 : [20190, 4221, 13697, 14552, END, 0, ...]
def data_processing(value, dictionary, pType):
    # 형태소 토크나이징 사용 유무
    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)

    sequences_input_index = []
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        
        if pType == DEFINES.dec_input:
            # 디코더 입력은 <START>로 시작한다.
            sequence_index = [dictionary[STD]]
        else:
            sequence_index = []
        
        for word in sequence.split():
            # word가 딕셔너리에 없으면 UNK (out of vacabulary)를 넣는다.
            if dictionary.get(word) is not None:
                sequence_index.append(dictionary[word])
            else:
                sequence_index.append(dictionary[UNK])
        
            # 문장의 단어수를  제한한다.
            if len(sequence_index) >= DEFINES.max_sequence_length:
                break
        
        # 디코더 출력은 <END>로 끝난다.
        if pType == DEFINES.dec_target:
            if len(sequence_index) < DEFINES.max_sequence_length:
                sequence_index.append(dictionary[END])
            else:
                sequence_index[len(sequence_index)-1] = dictionary[END]
                
        # max_sequence_length보다 문장 길이가 작으면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index)

# input_fn()의 map() 함수
def rearrange(inputX, outputX, targetY):
    features = {"input": inputX, "output": outputX}
    return features, targetY

# Estimator()에서 사용할 배치 데이터를 만드는 함수이다.
def input_fn(input_enc, output_dec, target_dec, batch_size, repeats):
    dataset = tf.data.Dataset.from_tensor_slices((input_enc, output_dec, target_dec))\
                    .shuffle(buffer_size=len(input_enc))\
                    .batch(batch_size)\
                    .map(rearrange)\
                    .repeat(repeats)\
                    .make_one_shot_iterator()\
                    .get_next()
    return dataset

# 토크나이징
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

# 사전 파일을 만든다
def load_vocabulary():
    vocabulary_list = []
    # 사전 파일의 존재 유무를 확인한다.
    if (not (os.path.exists(DEFINES.vocabulary_path))):
        if (os.path.exists(DEFINES.data_path)):
            data_df = pd.read_csv(DEFINES.data_path, encoding='cp949')
            question, answer = list(data_df['Q']), list(data_df['A'])
            
            # 질문과 응답 문장의 단어를 형태소로 바꾼다
            if DEFINES.tokenize_as_morph:  
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
                
            data = []
            data.extend(question)
            data.extend(answer)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER
            
        # 사전 리스트를 사전 파일로 만들어 넣는다.
        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    # 사전 파일이 존재하면 여기에서 그 파일을 불러서 배열에 넣어 준다.
    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip()) # strip()  양쪽 끝에 있는 공백과 \n 기호 삭제

    word2idx, idx2word = make_vocabulary(vocabulary_list)
    
    # 두가지 형태의 키와 값이 있는 형태를 리턴한다. 
    # (예) 단어: 인덱스 , 인덱스: 단어)
    return word2idx, idx2word, len(word2idx)

# 리스트를 키가 단어이고 값이 인덱스인 딕셔너리를 만든다.
# 리스트를 키가 인덱스이고 값이 단어인 딕셔너리를 만든다.
def make_vocabulary(vocabulary_list):
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    return word2idx, idx2word

def main(self):
    char2idx, idx2char, vocabulary_length = load_vocabulary()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
