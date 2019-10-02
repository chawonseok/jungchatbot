import tensorflow as tf
import model as ml
import data
from configs import DEFINES

# 사전을 구성 한다.
char2idx, idx2char, vocabulary_length = data.load_vocabulary()

# 학습 데이터와 테스트 데이터를 가져온다.
train_input, train_label, eval_input, eval_label = data.load_data()

# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
train_input_enc = data.data_processing(train_input, char2idx, DEFINES.enc_input)
train_input_dec = data.data_processing(train_label, char2idx, DEFINES.dec_input)
train_target_dec = data.data_processing(train_label, char2idx, DEFINES.dec_target)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
eval_input_enc = data.data_processing(eval_input, char2idx, DEFINES.enc_input)
eval_input_dec = data.data_processing(eval_label, char2idx, DEFINES.dec_input)
eval_target_dec = data.data_processing(eval_label, char2idx, DEFINES.dec_target)

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
        'max_sequence_length': DEFINES.max_sequence_length
    })

# 학습
tf.logging.set_verbosity(tf.logging.INFO)
classifier.train(input_fn=lambda: data.input_fn(
    train_input_enc,
    train_input_dec,
    train_target_dec,
    DEFINES.batch_size,
    DEFINES.train_repeats), steps=10000)

# 평가
print("#### 평가 ####")
eval_result = classifier.evaluate(input_fn=lambda: data.input_fn(
    eval_input_enc,
    eval_input_dec,
    eval_target_dec,
    DEFINES.batch_size,
    DEFINES.train_repeats), steps=1)

print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

