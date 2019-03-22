import numpy as np

train_kde_data = []

with open('data/trec-2011/kde.txt') as f:
    #train_kde_data.extend([[float(line)] for line in f])
   # test_kde_data = [float(line) for line in f]
    test_submission_kde = [line.replace('\n', '').split(' ') for line in f]
    test_extra_kde_features = [[1 / np.log(1 + float(line[3]))] for line in test_submission_kde]
with open('data/trec-2011/id.txt') as f:
        test_submission_ids = [line.replace('\n', '').split(' ') for line in f]
        test_extra_features = [[1 / np.log(1 + int(line[3]))] for line in test_submission_ids]
#addit_kde_train = np.array(train_kde_data)
        
qids_test = [id[0] for id in test_submission_ids]
ql_kde_pred = [1 / int(id[3]) for id in test_submission_kde]        

ql_pred = [1 / int(id[3]) for id in test_submission_ids]


for j in range(2):
    f= open("de%d_predictions.txt" % j,"w+")  
    for i in range(10):
        f.writelines(str(i) +'\n')
    f.close()        
#addit_kde_test = np.array(test_kde_data)
#ql_kde_pred = [1/ addit_kde_test[i] for i in range(len(test_kde_data))]
#ql_kde_pred = [1/ i[0] for i in test_kde_data]
#ql_pred = [1 / int(id[3]) for id in test_submission_ids]
#ql_pred = [1/ test_kde_data[i] for i in range(len(test_kde_data))]