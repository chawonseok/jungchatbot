import tensorflow as tf
import data
import model as ml
from configs import DEFINES
tf.logging.set_verbosity(tf.logging.ERROR)

# 사전을 구성 한다.
char2idx, idx2char, vocabulary_length = data.load_vocabulary()

def serving_input_fn():
    x = tf.placeholder(dtype=tf.int32, shape=[1, DEFINES.max_sequence_length], name='x')
    y = tf.placeholder(dtype=tf.int32, shape=[1, DEFINES.max_sequence_length], name='y')
    inputs = {'input': x, 'output': y }

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# 에스티메이터 구성한다.
classifier = tf.estimator.Estimator(
    model_fn=ml.Model,
    model_dir=DEFINES.check_point_path,
    params={
        'embedding_size': DEFINES.embedding_size,
        'model_hidden_size': DEFINES.model_hidden_size,
        'ffn_hidden_size': DEFINES.ffn_hidden_size,
        'attention_head_size': DEFINES.attention_head_size,
        'learning_rate': DEFINES.learning_rate,
        'vocabulary_length': vocabulary_length,
        'layer_size': DEFINES.layer_size,
        'max_sequence_length': DEFINES.max_sequence_length,
    })
    
print("\n# 채팅 준비중 입니다...")
estimator_predictor = tf.contrib.predictor.from_estimator(classifier, serving_input_fn)

# 모델 빌드, checkpoint 로드를 위해 한 번의 dummy 예측을 수행한다.
print("# 모델을 빌드하고 checkpoint를 로드하고 있습니다...")
predic_input_enc = data.data_processing(["dummy"], char2idx, DEFINES.enc_input)
predic_output_dec = data.data_processing([""], char2idx, DEFINES.dec_input)
predictions = estimator_predictor({"input": predic_input_enc, "output": predic_output_dec})
print("# 채팅 준비가 완료됐습니다.")
print("# 채팅을 종료하려면 'quit'를 입력하세요")
      
for q in range(1000):
    question = input("Q: ")
    if question == 'quit':
        break
    
    predic_input_enc = data.data_processing([question], char2idx, DEFINES.enc_input)
    predic_output_dec = data.data_processing([""], char2idx, DEFINES.dec_input)
    
    predictions = estimator_predictor({"input": predic_input_enc, "output": predic_output_dec})
    sentence_string = [idx2char[index] for index in predictions['indexs'][0]]
    
    answer = ""
    for word in sentence_string:
        if word == '<END>':
            is_finished = True
            break
    
        if word != '<PAD>' and word != '<END>':
            answer += word
            answer += " "
    
    print("A:", answer)
