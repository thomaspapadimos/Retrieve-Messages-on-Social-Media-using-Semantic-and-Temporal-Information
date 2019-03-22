import keras
from keras import optimizers, regularizers
from keras.layers import Dense, Dropout
from keras.layers import Input, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.merge import Dot
from keras.models import Model

import metrics


def sm_model(embed_dim, max_query_len, max_doc_len, vocab_size, embeddings, addit_feat_len, addit_kde_len, no_conv_filters=100):

    """Neural architecture as mentioned in the original paper."""
    print('Preparing model with the following parameters: ')
    print('''embed_dim, max_ques_len, max_ans_len, vocab_size, embedding, addit_feat_len, no_conv_filters: ''',)
    print(embed_dim, max_query_len, max_doc_len, vocab_size, embeddings.shape, addit_feat_len, no_conv_filters)

    # Prepare layers for Query
    input_q = Input(shape=(max_query_len,), name='query_input')

    # Load embedding values from corpus here.
    embed_q = Embedding(input_dim=vocab_size,
                        output_dim=embed_dim,
                        input_length=max_query_len,
                        weights=[embeddings],
                        trainable=False)(input_q)

    # Padding means, if input size is 32x32, output will also be 32x32, i.e, the dimensions will not reduce
    conv_q = Conv1D(filters=no_conv_filters,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-5),
                    name='query_conv')(embed_q)
    # Dropout and Max pooling
    conv_q = Dropout(0.5)(conv_q)
    pool_q = GlobalMaxPooling1D(name='query_pool')(conv_q)

    # Prepare layers for Document
    input_d = Input(shape=(max_doc_len,), name='document_input')

    # Load embedding values from corpus here.
    embed_d = Embedding(input_dim=vocab_size,
                        output_dim=embed_dim,
                        input_length=max_doc_len,
                        weights=[embeddings],
                        trainable=False)(input_d)

    conv_d = Conv1D(filters=no_conv_filters,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-5),
                    name='document_conv')(embed_d)
    # Dropout and Max pooling
    conv_d = Dropout(0.5)(conv_d)
    pool_d = GlobalMaxPooling1D(name='document_pool')(conv_d)

    # similarity layer: SIM = Xq M Xd
    M = Dense(no_conv_filters, use_bias=False, kernel_regularizer=regularizers.l2(1e-4), name='similarity_matrix')
    x_d = M(pool_d)
    sim = Dot(axes=-1)([pool_q, x_d])

    # Input additional features.
    input_additional_feat = Input(shape=(addit_feat_len,), name='input_addn_feat')
    input_kde_feat = Input(shape=(addit_kde_len,), name='input_k_feat')

    # Combine Question, sim, Answer pooled outputs and additional input features
    join_layer = keras.layers.concatenate([pool_q, sim, pool_d, input_additional_feat, input_kde_feat])
    join_layer = Dropout(0.5)(join_layer)

    # hidden_units = join_layer.output_shape[1]
    hidden_units = no_conv_filters + 1 + no_conv_filters + addit_feat_len + addit_kde_len

    # Using relu here too? Not mentioned in the paper.
    hidden_layer = Dense(units=hidden_units,
                         activation='tanh',
                         kernel_regularizer=regularizers.l2(1e-4),
                         name='hidden_layer')(join_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    # Final Softmax Layer
    softmax_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=[input_q, input_d, input_additional_feat, input_kde_feat], outputs=softmax_layer)

    ada_delta = optimizers.Adadelta(rho=0.95, epsilon=1e-06)
    model.compile(optimizer=ada_delta, loss='binary_crossentropy', metrics=['accuracy', metrics.precision, metrics.recall])

    return model
