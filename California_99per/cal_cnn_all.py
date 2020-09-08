import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import metrics
import tensorflow as tf
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.regularizers import l2
from keras.layers import MaxPooling2D
from copy import deepcopy
import statistics
from statistics import mean 
import time
from itertools import combinations

nex_nq500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nq500.txt", header = None, delim_whitespace=True)
nex_nqv2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nqv2m.txt", header = None, delim_whitespace=True)
nex_nqv10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nqv10m.txt", header = None, delim_whitespace=True)
nex_nrh500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nrh500.txt", header = None, delim_whitespace=True)
nex_nrh700 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nrh700.txt", header = None, delim_whitespace=True)
nex_nt2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nt2m.txt", header = None, delim_whitespace=True)
nex_nt2mdew = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nt2mdew.txt", header = None, delim_whitespace=True)
nex_nt10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nt10m.txt", header = None, delim_whitespace=True)
nex_nt500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nt500.txt", header = None, delim_whitespace=True)
nex_ntpw = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_ntpw.txt", header = None, delim_whitespace=True)
nex_nu2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nu2m.txt", header = None, delim_whitespace=True)
nex_nu10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nu10m.txt", header = None, delim_whitespace=True)
nex_nu500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nu500.txt", header = None, delim_whitespace=True)
nex_nuqv = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nuqv.txt", header = None, delim_whitespace=True)
nex_nv2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nv2m.txt", header = None, delim_whitespace=True)
nex_nv10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nv10m.txt", header = None, delim_whitespace=True)
nex_nv500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nv500.txt", header = None, delim_whitespace=True)
nex_nvqv = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nvqv.txt", header = None, delim_whitespace=True)
nex_nw500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_nex_nw500.txt", header = None, delim_whitespace=True)

ex_nq500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nq500.txt", header = None, delim_whitespace=True)
ex_nqv2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nqv2m.txt", header = None, delim_whitespace=True)
ex_nqv10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nqv10m.txt", header = None, delim_whitespace=True)
ex_nrh500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nrh500.txt", header = None, delim_whitespace=True)
ex_nrh700 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nrh700.txt", header = None, delim_whitespace=True)
ex_nt2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nt2m.txt", header = None, delim_whitespace=True)
ex_nt2mdew = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nt2mdew.txt", header = None, delim_whitespace=True)
ex_nt10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nt10m.txt", header = None, delim_whitespace=True)
ex_nt500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nt500.txt", header = None, delim_whitespace=True)
ex_ntpw = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_ntpw.txt", header = None, delim_whitespace=True)
ex_nu2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nu2m.txt", header = None, delim_whitespace=True)
ex_nu10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nu10m.txt", header = None, delim_whitespace=True)
ex_nu500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nu500.txt", header = None, delim_whitespace=True)
ex_nuqv = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nuqv.txt", header = None, delim_whitespace=True)
ex_nv2m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nv2m.txt", header = None, delim_whitespace=True)
ex_nv10m = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nv10m.txt", header = None, delim_whitespace=True)
ex_nv500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nv500.txt", header = None, delim_whitespace=True)
ex_nvqv = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nvqv.txt", header = None, delim_whitespace=True)
ex_nw500 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_8005_ex_nw500.txt", header = None, delim_whitespace=True)


nq500_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nq500.txt", header = None, delim_whitespace=True)
nqv2m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nqv2m.txt", header = None, delim_whitespace=True)
nqv10m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nqv10m.txt", header = None, delim_whitespace=True)
nrh500_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nrh500.txt", header = None, delim_whitespace=True)
nrh700_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nrh700.txt", header = None, delim_whitespace=True)
nt2m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nt2m.txt", header = None, delim_whitespace=True)
nt2mdew_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nt2mdew.txt", header = None, delim_whitespace=True)
nt10m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nt10m.txt", header = None, delim_whitespace=True)
nt500_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nt500.txt", header = None, delim_whitespace=True)
ntpw_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_ntpw.txt", header = None, delim_whitespace=True)
nu2m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nu2m.txt", header = None, delim_whitespace=True)
nu10m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nu10m.txt", header = None, delim_whitespace=True)
nu500_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nu500.txt", header = None, delim_whitespace=True)
nuqv_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nuqv.txt", header = None, delim_whitespace=True)
nv2m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nv2m.txt", header = None, delim_whitespace=True)
nv10m_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nv10m.txt", header = None, delim_whitespace=True)
nv500_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nv500.txt", header = None, delim_whitespace=True)
nvqv_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nvqv.txt", header = None, delim_whitespace=True)
nw500_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_0619_nw500.txt", header = None, delim_whitespace=True)

indicator_0619 = pd.read_csv("../data_Cal_99/DJF_CASM_99per_0619_indicator.txt", header = None, delim_whitespace=True)
test_set_y_initial = indicator_0619[4]
test_y = test_set_y_initial.to_numpy().reshape(1, test_set_y_initial.shape[0])

def get_train_test(nex_names, ex_names, test_names):
    nex_combo_7905 = pd.concat(nex_names, axis = 1)
    nex_combo_7905["label"] = 0

    ex_combo_7905 = pd.concat(ex_names, axis = 1)
    ex_combo_7905["label"] = 1

    combo_7905_df = pd.concat([ex_combo_7905, nex_combo_7905])

    train_combo_y_initial = combo_7905_df['label'].to_numpy()
    #train_combo_y = train_combo_y_initial.reshape(1, train_combo_y_initial.shape[0])
    train_combo_x = combo_7905_df.iloc[:,0:combo_7905_df.shape[1]-1].to_numpy()

    test_df = pd.concat(test_names, axis = 1)

    test_combo_x = test_df.to_numpy()

#     over = SMOTE(sampling_strategy=0.1)
#     under = RandomUnderSampler(sampling_strategy=0.5)
#     steps = [('o', over), ('u', under)]
#     pipeline = Pipeline(steps=steps)
#     #transform the dataset
#     train_combo_x, train_combo_y = pipeline.fit_resample(train_combo_x, train_combo_y_initial)
    train_combo_x, train_combo_y = shuffle(train_combo_x, train_combo_y_initial, random_state = 0)
    #summarize the new class distribution
    train_combo_y = train_combo_y.reshape(1, train_combo_y.shape[0])
    #print(train_combo_y)
    return train_combo_x, test_combo_x, train_combo_y


def get_train_test_balanced(nex_names, ex_names, test_names):
    nex_combo_7905 = pd.concat(nex_names, axis = 1)
    nex_combo_7905["label"] = 0

    ex_combo_7905 = pd.concat(ex_names, axis = 1)
    ex_combo_7905["label"] = 1

    combo_7905_df = pd.concat([ex_combo_7905, nex_combo_7905])

    train_combo_y_initial = combo_7905_df['label'].to_numpy()
    #train_combo_y = train_combo_y_initial.reshape(1, train_combo_y_initial.shape[0])
    train_combo_x = combo_7905_df.iloc[:,0:combo_7905_df.shape[1]-1].to_numpy()

    test_df = pd.concat(test_names, axis = 1)

    test_combo_x = test_df.to_numpy()

    over = SMOTE(sampling_strategy=0.043)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    #transform the dataset
    train_combo_x, train_combo_y = pipeline.fit_resample(train_combo_x, train_combo_y_initial)
    train_combo_x, train_combo_y = shuffle(train_combo_x, train_combo_y, random_state = 0)
    #summarize the new class distribution
    train_combo_y = train_combo_y.reshape(1, train_combo_y.shape[0])
    #print(train_combo_y)
    return train_combo_x, test_combo_x, train_combo_y

def conf_matrix(predictions, y):
    cm = np.zeros((2,2), dtype = int)
    for i in range(y.shape[1]):
        if(y[0,i] == 1 and predictions[0,i] == 1):
            cm[0,0] += 1
        if(y[0,i] == 1 and predictions[0,i] == 0):
            cm[0,1] += 1
        if(y[0,i] == 0 and predictions[0,i] == 1):
            cm[1,0] += 1
        if(y[0,i] == 0 and predictions[0,i] == 0):
            cm[1,1] += 1
    return cm

def evalmetrics(cm):
    tp = cm[0,0]
    fn = cm[0,1]
    fp = cm[1,0]
    tn = cm[1,1]

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    precision = tp/(tp+fp)
    f1 = 2*precision*tpr/(precision + tpr)

    return tpr, fpr, f1

def create_model(channels):
    model = Sequential()
    #add model layers
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(29,33,channels), activity_regularizer=l2(0.001)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', activity_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model

def pred_train(predictions_train, shape):
    preds_train = np.zeros((1, shape[0]))
    for i in range(len(predictions_train)):

        first = predictions_train[i,0]
        second = predictions_train[i,1]

        if(first > second):
            preds_train[0,i] = 0
        else:
            preds_train[0,i] = 1
    return preds_train

def pred_test(predictions_test, shape):
    preds_test = np.zeros((1, 1263))
    for i in range(len(predictions_test)):

        first = predictions_test[i,0]
        second = predictions_test[i,1]

        if(first > second):
            preds_test[0,i] = 0
        else:
            preds_test[0,i] = 1
    return preds_test

nex_mapping = {
    0:nex_nq500,
    1:nex_nqv2m,
    2:nex_nqv10m,
    3:nex_nrh500,
    4:nex_nrh700,
    5:nex_nt2m,
    6:nex_nt2mdew,
    7:nex_nt10m,
    8:nex_nt500,
    9:nex_ntpw,
    10:nex_nu2m,
    11:nex_nu10m,
    12:nex_nu500,
    13:nex_nuqv,
    14:nex_nv2m,
    15:nex_nv10m,
    16:nex_nv500,
    17:nex_nvqv,
    18:nex_nw500,
}

ex_mapping = {
    0:ex_nq500,
    1:ex_nqv2m,
    2:ex_nqv10m,
    3:ex_nrh500,
    4:ex_nrh700,
    5:ex_nt2m,
    6:ex_nt2mdew,
    7:ex_nt10m,
    8:ex_nt500,
    9:ex_ntpw,
    10:ex_nu2m,
    11:ex_nu10m,
    12:ex_nu500,
    13:ex_nuqv,
    14:ex_nv2m,
    15:ex_nv10m,
    16:ex_nv500,
    17:ex_nvqv,
    18:ex_nw500,
}

mapping_0619 = {
    0:nq500_0619,
    1:nqv2m_0619,
    2:nqv10m_0619,
    3:nrh500_0619,
    4:nrh700_0619,
    5:nt2m_0619,
    6:nt2mdew_0619,
    7:nt10m_0619,
    8:nt500_0619,
    9:ntpw_0619,
    10:nu2m_0619,
    11:nu10m_0619,
    12:nu500_0619,
    13:nuqv_0619,
    14:nv2m_0619,
    15:nv10m_0619,
    16:nv500_0619,
    17:nvqv_0619,
    18:nw500_0619,
}



mapping_text = {
    0:'nq500',
    1:'nqv2m',
    2:'nqv10m',
    3:'nrh500',
    4:'nrh700',
    5:'nt2m',
    6:'nt2mdew',
    7:'nt10m',
    8:'nt500',
    9:'ntpw',
    10:'nu2m',
    11:'nu10m',
    12:'nu500',
    13:'nuqv',
    14:'nv2m',
    15:'nv10m',
    16:'nv500',
    17:'nvqv',
    18:'nw500',
}

# start = time.time()



def cnn_combo(numofvar):

    file1 = open("cal_cnn_all_results_100.txt", "w")

    comb = combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],numofvar) 
    # comb = combinations([0,1,2],numofvar)
    cms = []
    # Print the obtained combinations 
    for i in list(comb):
        individual_balanced_tpr_train_avg = []
        individual_balanced_fpr_train_avg = []
        individual_balanced_f1_train_avg = []
        individual_balanced_tpr_test_avg = []
        individual_balanced_fpr_test_avg = []
        individual_balanced_f1_test_avg = []

        individual_balanced_tpr_train_std = []
        individual_balanced_fpr_train_std = []
        individual_balanced_f1_train_std = []
        individual_balanced_tpr_test_std = []
        individual_balanced_fpr_test_std = []
        individual_balanced_f1_test_std = []
            
        tpr_train_avg = []
        fpr_train_avg = []
        f1_train_avg = []
        tpr_test_avg = []
        fpr_test_avg = []
        f1_test_avg = []

        nex = []
        ex = []
        test = []
        combo = list(i)

        for j in range(len(combo)):
            file1.write(mapping_text[combo[j]]) 
            file1.write('\n')
            nex.append(nex_mapping[combo[j]])
            ex.append(ex_mapping[combo[j]])
            test.append(mapping_0619[combo[j]])


        channels = len(nex)
        
        for i in range(100):
            # print(i)
            train_x, test_x, train_y = get_train_test_balanced(nex, ex, test)
            my_X_train = train_x.reshape(train_x.shape[0],29,33,channels)
            my_X_test = test_x.reshape(1263,29,33,channels)
            my_y_train = train_y.reshape(train_x.shape[0],)
            my_y_test = test_y.reshape(1263,)
            my_y_train = to_categorical(my_y_train)
            my_y_test = to_categorical(my_y_test)
            model = create_model(channels)
            model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            model.fit(my_X_train, my_y_train, validation_data = (my_X_test, my_y_test), epochs=7)
            predictions_test = model.predict(my_X_test)
            predictions_train = model.predict(my_X_train)
            predics_train = pred_train(predictions_train, train_x.shape)
            predics_test = pred_test(predictions_test, test_x.shape)  

            cm_train = conf_matrix(predics_train, train_y)
            confusion_train = pd.DataFrame(cm_train, index=['actual_extreme', 'actual_non-extreme'],
                                     columns=['predicted_extreme','predicted_non-extreme'])


            print(confusion_train)
            tpr, fpr, f1 = evalmetrics(cm_train)
            tpr_train_avg.append(tpr)
            fpr_train_avg.append(fpr)
            f1_train_avg.append(f1)
            print("True positive rate: %f" %(tpr))
            print("False positive rate: %f" %(fpr))
            print("F1 score: %f" %(f1))

            cm_test = conf_matrix(predics_test, test_y)
            confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (34)', 'actual_non-extreme (1229)'],
                             columns=['predicted_extreme','predicted_non-extreme'])

            print(confusion_test)
            tpr, fpr, f1 = evalmetrics(cm_test)
            tpr_test_avg.append(tpr)
            fpr_test_avg.append(fpr)
            f1_test_avg.append(f1)
            print("True positive rate: %f" %(tpr))
            print("False positive rate: %f" %(fpr))
            print("F1 score: %f" %(f1))


        individual_balanced_tpr_train_std.append(statistics.stdev(tpr_train_avg))
        individual_balanced_fpr_train_std.append(statistics.stdev(fpr_train_avg))
        individual_balanced_f1_train_std.append(statistics.stdev(f1_train_avg))
        individual_balanced_tpr_test_std.append(statistics.stdev(tpr_test_avg))
        individual_balanced_fpr_test_std.append(statistics.stdev(fpr_test_avg))
        individual_balanced_f1_test_std.append(statistics.stdev(f1_test_avg))

        individual_balanced_tpr_train_avg.append(mean(tpr_train_avg))
        individual_balanced_fpr_train_avg.append(mean(fpr_train_avg))
        individual_balanced_f1_train_avg.append(mean(f1_train_avg))

        individual_balanced_tpr_test_avg.append(mean(tpr_test_avg))
        individual_balanced_fpr_test_avg.append(mean(fpr_test_avg))
        individual_balanced_f1_test_avg.append(mean(f1_test_avg))

        file1.write(f"Train tpr: {round(individual_balanced_tpr_train_avg[0],3)} +- {round(individual_balanced_tpr_train_std[0],3)}\n")
        file1.write(f"Train fpr: {round(individual_balanced_fpr_train_avg[0],3)} +- {round(individual_balanced_fpr_train_std[0],3)}\n")
        file1.write(f"Train f1: {round(individual_balanced_f1_train_avg[0],3)} +- {round(individual_balanced_f1_train_std[0],3)}\n")

        file1.write(f"Test tpr: {round(individual_balanced_tpr_test_avg[0],3)} +- {round(individual_balanced_tpr_test_std[0],3)}\n")   
        file1.write(f"Test fpr: {round(individual_balanced_fpr_test_avg[0],3)} +- {round(individual_balanced_fpr_test_std[0],3)}\n")       
        file1.write(f"Test f1: {round(individual_balanced_f1_test_avg[0],3)} +- {round(individual_balanced_f1_test_std[0],3)}\n")
        file1.write('\n')

cnn_combo(1)
