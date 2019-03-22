
import config
import models
import metrics

import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session

from text_preprocessing import create_vocab, load_embeddings_from_file, get_sequences
from batch_generator import batch_gen


def create_train_data(train_dirs, include_extra_features=True):
    # Load train set.
    train_query_sequences = []
    train_document_sequences = []
    train_labels = []

    train_submission_ids = []
    train_extra_features = []
    train_kde_data = []
    train_submission_kde = []
    

    for file in train_dirs:
        with open(config.DATA_PATH + file + '/a.seq') as f:
            train_query_sequences.extend(
                [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f]) 
        with open(config.DATA_PATH + file + '/b.seq') as f:
            train_document_sequences.extend(
                [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f])
        with open(config.DATA_PATH + file + '/sim.txt') as f:
            train_labels.extend([[line.replace('\n', '')] for line in f])
        with open(config.DATA_PATH + file + '/id.txt') as f:
            temp = [line.replace('\n', '').split(' ') for line in f]
            train_submission_ids.extend(temp)
            train_extra_features.extend([[1 / np.log(1 + int(line[3]))] for line in temp])
        with open(config.DATA_PATH + file + '/kde.txt') as f:
            temp_kde = [line.replace('\n', '').split(' ') for line in f]
            train_submission_kde.extend(temp_kde)
            train_kde_data.extend([[1 / np.log(1 + int(line[3]))] for line in temp_kde])
           # train_kde_data.extend([[float(line)] for line in f])
    
    train_query_sequences = pad_sequences(train_query_sequences, maxlen=config.QUERY_MAX_SEQUENCE_LENGTH)
    train_document_sequences = pad_sequences(train_document_sequences, maxlen=config.DOC_MAX_SEQUENCE_LENGTH)

    q_train = np.array(train_query_sequences)
    d_train = np.array(train_document_sequences)
    y_train = np.array(train_labels)
    y_train = y_train.astype(np.int)
    if include_extra_features:
        addit_feat_train = np.array(train_extra_features)
        addit_kde_train = np.array(train_kde_data)
    else:
        addit_feat_train = np.zeros(y_train.shape)

    return q_train, d_train, y_train, addit_feat_train, addit_kde_train


def create_test_data(test_dir, include_extra_features=True):
    with open(config.DATA_PATH + test_dir + '/a.seq') as f:
        test_query_sequences = [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f]
        test_query_sequences = pad_sequences(test_query_sequences, maxlen=config.QUERY_MAX_SEQUENCE_LENGTH)
    with open(config.DATA_PATH + test_dir + '/b.seq') as f:
        test_document_sequences = [[0] if line is '\n' else list(map(int, line.replace('\n', '').split(','))) for line in f]
        test_document_sequences = pad_sequences(test_document_sequences, maxlen=config.DOC_MAX_SEQUENCE_LENGTH)
    with open(config.DATA_PATH + test_dir + '/sim.txt') as f:
        test_labels = [[line.replace('\n', '')] for line in f]
    with open(config.DATA_PATH + test_dir + '/id.txt') as f:
        test_submission_ids = [line.replace('\n', '').split(' ') for line in f]
        test_extra_features = [[1 / np.log(1 + int(line[3]))] for line in test_submission_ids]
    with open(config.DATA_PATH + test_dir + '/kde.txt') as f:
        test_submission_kde = [line.replace('\n', '').split(' ') for line in f]
        test_kde_data = [[1 / np.log(1 + float(line[3]))] for line in test_submission_kde]

    q_test = np.array(test_query_sequences)
    d_test = np.array(test_document_sequences)
    y_test = np.array(test_labels)
    y_test = y_test.astype(np.int)
    if include_extra_features:
        addit_feat_test = np.array(test_extra_features)
        addit_kde_test = np.array(test_kde_data)
    else:
        addit_feat_test = np.zeros(y_test.shape)
        addit_kde_test = np.zeros(y_test.shape)
        
    qids_test = [id[0] for id in test_submission_ids]
    docn_test = [id[2] for id in test_submission_ids]
    ql_pred = [1 / int(id[3]) for id in test_submission_ids]
    ql_kde_pred = [1 / int(id[3]) for id in test_submission_kde]

    return q_test, d_test, y_test, addit_feat_test, addit_kde_test, qids_test, docn_test, ql_pred , ql_kde_pred


def train_model():

    # create vocab
    texts = []
    for file in ['trec-2011', 'trec-2012', 'trec-2013', 'trec-2014']:
        with open(config.DATA_PATH + file + '/a.toks') as f:
            texts.extend([line for line in f])
        with open(config.DATA_PATH + file + '/b.toks') as f:
            texts.extend([line for line in f])
    vocab, tokenizer = create_vocab(texts)

    
    for file in ['trec-2011', 'trec-2012', 'trec-2013', 'trec-2014']:
        with open(config.DATA_PATH + file + '/a.toks') as f:
            a_seq = get_sequences(tokenizer, [line for line in f])
            with open(config.DATA_PATH + file + '/a.seq', 'w') as o_f:
                wr = csv.writer(o_f)
                wr.writerows(a_seq)

        with open(config.DATA_PATH + file + '/b.toks') as f:
            b_seq = get_sequences(tokenizer, [line for line in f])
            with open(config.DATA_PATH + file + '/b.seq', 'w') as o_f:
                wr = csv.writer(o_f)
                wr.writerows(b_seq)


    embeddings, embed_dim, _ = load_embeddings_from_file(config.DATA_PATH)
    print('embedding dim:', embed_dim)

    #embeddings, dim, _ = load_embeddings(config.EMBEDDING_PATH, vocab)
    #print('embedding dim:', dim)
    #save_embeddings(embeddings, './output/')

    clear_session()


    all_dirs = ['trec-2014','trec-2013', 'trec-2012', 'trec-2011']
    for test_dir in all_dirs:

      #  train_dirs = ['trec-2013', 'trec-2012', 'trec-2011']
        train_dirs = list(all_dirs)
        train_dirs.remove(test_dir)
        print('TEST ON %s, TRAIN ON %s ' % (test_dir, train_dirs))

        # Get train data
        q_train, d_train, y_train, addit_feat_train, addit_kde_train = create_train_data(train_dirs, True)
        addit_feat_len = addit_feat_train.shape[1] if addit_feat_train.ndim > 1 else 1
        addit_kde_len = addit_kde_train.shape[1] if addit_kde_train.ndim > 1 else 1

        # Get test data and baseline ranking
        q_test, d_test, y_test, addit_feat_test, addit_kde_test, qids_test, docn_test,  ql_pred, ql_kde_pred = create_test_data(test_dir, True)

        # Create model
        sm_model = models.sm_model(embed_dim=embed_dim,
                                   max_query_len=config.QUERY_MAX_SEQUENCE_LENGTH,
                                   max_doc_len=config.DOC_MAX_SEQUENCE_LENGTH,
                                   vocab_size=len(vocab) + 1,
                                   embeddings=embeddings,
                                   addit_feat_len=addit_feat_len,
                                   addit_kde_len=addit_kde_len)
        # print(sm_model.summary())
        # Obtain a shuffled batch of the samples
        q_train, a_train, y_train, addit_feat_train, addit_kde_train = shuffle(q_train,
                                                              d_train,
                                                              y_train,
                                                              addit_feat_train,
                                                              addit_kde_train,
                                                              random_state=config.RANDOM_STATE)
        early_stop = False
        best_test_map = 0
        iters_not_improved = 0
        
        
        for epoch in range(config.EPOCHS):
            f= open("%s_kde_predictions.txt" %test_dir,"w+") 
        

            if early_stop:
                print("Early Stopping. Epoch: {}".format(epoch))
                break

            batches = zip(batch_gen(q_train, config.BATCH_SIZE),
                          batch_gen(d_train, config.BATCH_SIZE),
                          batch_gen(y_train, config.BATCH_SIZE),
                          batch_gen(addit_feat_train, config.BATCH_SIZE),
                          batch_gen(addit_kde_train, config.BATCH_SIZE))

            batch_no = 0
            test_metrics = []
            for q_train_batch, d_train_batch, y_train_batch, addit_feat_train_batch, addit_kde_train_batch in batches:
                sm_model.train_on_batch([q_train_batch, d_train_batch, addit_feat_train_batch, addit_kde_train_batch], y_train_batch)
                batch_metrics = sm_model.test_on_batch([q_train_batch, d_train_batch, addit_feat_train_batch, addit_kde_train_batch], y_train_batch)
                test_metrics.append(batch_metrics)
                batch_no += 1

            # TODO: average metrics across batches

            test_metrics = [sum(i) / len(test_metrics) for i in zip(*test_metrics)]
            y_pred = sm_model.predict([q_test, d_test, addit_feat_test, addit_kde_test])
            
        
            for i in range(len(q_test)): 
                f.writelines(str(qids_test[i]) + ' ' + 'Q0' + ' ' +  str(docn_test[i]) + ' ' + str(0) + ' ' + '%f' %y_pred[i]  +  ' ' + 'lucene4lm' + '\n')
            
            f.close()
            

            test_acc = roc_auc_score(y_test, y_pred)
            test_map = metrics.map_score(qids_test, y_test, y_pred)
            precision_at_15 = metrics.precision_at_k(qids_test, y_test, y_pred, k=15)
            precision_at_30 = metrics.precision_at_k(qids_test, y_test, y_pred, k=30)
            precision_at_100 = metrics.precision_at_k(qids_test, y_test, y_pred, k=100)

            # print results
            print('Epoch %s/%s' % (epoch + 1, config.EPOCHS))
            print(', '.join(['%s: %f' % (name, value) for name, value in zip(sm_model.metrics_names, test_metrics)]))
            print('auc: %f, map: %f, p@15: %f, p@30: %f, p@100: %f' % (test_acc,
                                                                       test_map,
                                                                       precision_at_15,
                                                                       precision_at_30,
                                                                       precision_at_100))

            if test_map > best_test_map:
                iters_not_improved = 0
                best_test_map = test_map
            else:
                iters_not_improved += 1
                if iters_not_improved >= 5:  
                    early_stop = True

            
       
        
        print('=================================================================')

        ql_test_map = metrics.map_score(qids_test, y_test, ql_pred)
        ql_precision_at_15 = metrics.precision_at_k(qids_test, y_test, ql_pred, k=15)
        ql_precision_at_30 = metrics.precision_at_k(qids_test, y_test, ql_pred, k=30)
        ql_precision_at_100 = metrics.precision_at_k(qids_test, y_test, ql_pred, k=100)
        print('Test QL model, MAP: %f, P@15: %f, P@30: %f, P@100: %f' %
              (ql_test_map, ql_precision_at_15, ql_precision_at_30, ql_precision_at_100))
        
        ql_kde_test_map = metrics.map_score(qids_test, y_test, ql_kde_pred)
        ql_kde_precision_at_15 = metrics.precision_at_k(qids_test, y_test, ql_kde_pred, k=15)
        ql_kde_precision_at_30 = metrics.precision_at_k(qids_test, y_test, ql_kde_pred, k=30)
        ql_kde_precision_at_100 = metrics.precision_at_k(qids_test, y_test, ql_kde_pred, k=100)
        print('Test QL (with kde feature) model, MAP: %f, P@15: %f, P@30: %f, P@100: %f' %
              (ql_kde_test_map, ql_kde_precision_at_15, ql_kde_precision_at_30, ql_kde_precision_at_100))
        


def model_fit(model, x_train, y_train, x_test, y_test, qids_test):
    model.fit(x=x_train,
              y=y_train,
              epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE,
              verbose=2,
              validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    test_acc = roc_auc_score(y_test, y_pred)
    test_map = metrics.map_score(qids_test, y_test, y_pred)
    precision_at_15 = metrics.precision_at_k(qids_test, y_test, y_pred, k=15)
    precision_at_30 = metrics.precision_at_k(qids_test, y_test, y_pred, k=30)
    precision_at_100 = metrics.precision_at_k(qids_test, y_test, y_pred, k=100)
    print('auc: %f, map: %f, p@15: %f, p@30: %f, p@100: %f' % (test_acc,
                                                               test_map,
                                                               precision_at_15,
                                                               precision_at_30,
                                                               precision_at_100))


if __name__ == '__main__':
    train_model()