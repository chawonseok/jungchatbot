# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# batch, layer, group normalization 개념 참조할 것.
# batch data 마다 평균과 표준 편차가 계속 달라지고, 표준화하면 바이어스도 무시된다.
# 이를 안정화 시키기위해 gamma와 beta를 도입한다. gamma와 beta는 학습 동안에 업데이트되는
# 파라메터이다. beta는 바이어스 역할도 한다.
def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.get_variable("beta", initializer=tf.zeros(feature_shape))
    gamma = tf.get_variable("gamma", initializer=tf.ones(feature_shape))

    return gamma * (inputs - mean) / (std + eps) + beta

# 교재 p.348 리지듀얼 커넥션 참조
def sublayer_connection(inputs, sublayer, dropout=0.2):
    outputs = layer_norm(inputs + tf.keras.layers.Dropout(rate = dropout)(sublayer))
    return outputs


# 교재의 positional_encoding() 결과가 달라, 아래의 소스로 변경했다.
# https://www.tensorflow.org/beta/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, num_units, heads, masked=False):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.masked = masked

        self.query_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.key_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.value_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)

    # 교재 P.341 순방향 마스크 어텐션 참조. 디코더에서 필요함.
    def scaled_dot_product_attention(self, query, key, value, masked=False):
        key_seq_length = float(key.get_shape().as_list()[-2])
        key = tf.transpose(key, perm=[0, 2, 1])
        outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)

        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        attention_map = tf.nn.softmax(outputs)

        return tf.matmul(attention_map, value)

    def call(self, query, key, value):
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = tf.concat(tf.split(query, self.heads, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.heads, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.heads, axis=-1), axis=0)

        attention_map = self.scaled_dot_product_attention(query, key, value, self.masked)

        attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1)

        return attn_outputs

# Multi-Head attention --> Add & Norm --> Feed forward --> Add & Norm : 이렇게 여러개 (num_layers) 층
class Encoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Encoder, self).__init__()

        self.self_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]

    def call(self, inputs):
        output_layer = None

        for i, (s_a, p_f) in enumerate(zip(self.self_attention, self.position_feedforward)):
            with tf.variable_scope('encoder_layer_' + str(i + 1)):
                attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))

                inputs = output_layer

        return output_layer

# Masked Multi-Head attention --> Add & Norm --> Multi-Head attention (Encoder-decoder attention) -->
# Add & Norm --> Feed forward --> Add & Norm
class Decoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Decoder, self).__init__()

        # Masked mult-head attention
        self.self_attention = [MultiHeadAttention(model_dims, attn_heads, masked=True) for _ in range(num_layers)]
        
        # Encoder-decoder attention
        self.encoder_decoder_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        
        # Feed forward
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]

    def call(self, inputs, encoder_outputs):
        output_layer = None

        for i, (s_a, ed_a, p_f) in enumerate(zip(self.self_attention, self.encoder_decoder_attention, self.position_feedforward)):
            with tf.variable_scope('decoder_layer_' + str(i + 1)):
                masked_attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))
                attention_layer = sublayer_connection(masked_attention_layer, ed_a(masked_attention_layer,
                                                                                           encoder_outputs,
                                                                                           encoder_outputs))
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))
                inputs = output_layer

        return output_layer


class PositionWiseFeedForward(tf.keras.Model):
    def __init__(self, num_units, feature_shape):
        super(PositionWiseFeedForward, self).__init__()

        self.inner_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.output_dense = tf.keras.layers.Dense(feature_shape)

    def call(self, inputs):
        inner_layer = self.inner_dense(inputs)
        outputs = self.output_dense(inner_layer)

        return outputs


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['max_sequence_length'], params['embedding_size'])

    embedding = tf.keras.layers.Embedding(params['vocabulary_length'],
                                          params['embedding_size'])

    encoder_layers = Encoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])

    decoder_layers = Decoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])

    logit_layer = tf.keras.layers.Dense(params['vocabulary_length'])

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x_embedded_matrix = embedding(features['input']) + position_encode
        encoder_outputs = encoder_layers(x_embedded_matrix)     # Encoder의 call()을 호출함.

    loop_count = params['max_sequence_length'] if PREDICT else 1

    predict, output, logits = None, None, None

    for i in range(loop_count):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            if i > 0:
                output = tf.concat([tf.ones(shape=(tf.shape(output)[0],1), dtype=tf.int64), predict[:, :-1]], axis=-1)
            else:
                output = features['output']
            y_embedded_matrix = embedding(output) + position_encode
            decoder_outputs = decoder_layers(y_embedded_matrix, encoder_outputs)
            logits = logit_layer(decoder_outputs)
            predict = tf.argmax(logits, 2)

    if PREDICT:
        predictions = {
            'indexs': predict
            #'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)