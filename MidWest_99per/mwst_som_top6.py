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
from minisom import MiniSom
# from hyperopt import fmin, tpe, hp
import time
from itertools import combinations 

# nex_nq500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nq500.txt", header = None, delim_whitespace=True)
# nex_nqv2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nqv2m.txt", header = None, delim_whitespace=True)
# nex_nqv10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nqv10m.txt", header = None, delim_whitespace=True)
# nex_nrh500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nrh500.txt", header = None, delim_whitespace=True)
# nex_nrh700 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nrh700.txt", header = None, delim_whitespace=True)
# nex_nt2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nt2m.txt", header = None, delim_whitespace=True)
# nex_nt2mdew = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nt2mdew.txt", header = None, delim_whitespace=True)
# nex_nt10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nt10m.txt", header = None, delim_whitespace=True)
# nex_nt500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nt500.txt", header = None, delim_whitespace=True)
# nex_ntpw = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_ntpw.txt", header = None, delim_whitespace=True)
# nex_nu2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nu2m.txt", header = None, delim_whitespace=True)
# nex_nu10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nu10m.txt", header = None, delim_whitespace=True)
# nex_nu500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nu500.txt", header = None, delim_whitespace=True)
nex_nuqv = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nuqv.txt", header = None, delim_whitespace=True)
nex_nv2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nv2m.txt", header = None, delim_whitespace=True)
nex_nv10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nv10m.txt", header = None, delim_whitespace=True)
nex_nv500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nv500.txt", header = None, delim_whitespace=True)
nex_nvqv = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nvqv.txt", header = None, delim_whitespace=True)
nex_nw500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_nex_nw500.txt", header = None, delim_whitespace=True)

# ex_nq500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nq500.txt", header = None, delim_whitespace=True)
# ex_nqv2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nqv2m.txt", header = None, delim_whitespace=True)
# ex_nqv10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nqv10m.txt", header = None, delim_whitespace=True)
# ex_nrh500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nrh500.txt", header = None, delim_whitespace=True)
# ex_nrh700 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nrh700.txt", header = None, delim_whitespace=True)
# ex_nt2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nt2m.txt", header = None, delim_whitespace=True)
# ex_nt2mdew = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nt2mdew.txt", header = None, delim_whitespace=True)
# ex_nt10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nt10m.txt", header = None, delim_whitespace=True)
# ex_nt500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nt500.txt", header = None, delim_whitespace=True)
# ex_ntpw = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_ntpw.txt", header = None, delim_whitespace=True)
# ex_nu2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nu2m.txt", header = None, delim_whitespace=True)
# ex_nu10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nu10m.txt", header = None, delim_whitespace=True)
# ex_nu500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nu500.txt", header = None, delim_whitespace=True)
ex_nuqv = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nuqv.txt", header = None, delim_whitespace=True)
ex_nv2m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nv2m.txt", header = None, delim_whitespace=True)
ex_nv10m = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nv10m.txt", header = None, delim_whitespace=True)
ex_nv500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nv500.txt", header = None, delim_whitespace=True)
ex_nvqv = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nvqv.txt", header = None, delim_whitespace=True)
ex_nw500 = pd.read_csv("../MWST_99per/JJA_MWST_99per_8005_ex_nw500.txt", header = None, delim_whitespace=True)


# nq500_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nq500.txt", header = None, delim_whitespace=True)
# nqv2m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nqv2m.txt", header = None, delim_whitespace=True)
# nqv10m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nqv10m.txt", header = None, delim_whitespace=True)
# nrh500_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nrh500.txt", header = None, delim_whitespace=True)
# nrh700_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nrh700.txt", header = None, delim_whitespace=True)
# nt2m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nt2m.txt", header = None, delim_whitespace=True)
# nt2mdew_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nt2mdew.txt", header = None, delim_whitespace=True)
# nt10m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nt10m.txt", header = None, delim_whitespace=True)
# nt500_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nt500.txt", header = None, delim_whitespace=True)
# ntpw_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_ntpw.txt", header = None, delim_whitespace=True)
# nu2m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nu2m.txt", header = None, delim_whitespace=True)
# nu10m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nu10m.txt", header = None, delim_whitespace=True)
# nu500_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nu500.txt", header = None, delim_whitespace=True)
nuqv_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nuqv.txt", header = None, delim_whitespace=True)
nv2m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nv2m.txt", header = None, delim_whitespace=True)
nv10m_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nv10m.txt", header = None, delim_whitespace=True)
nv500_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nv500.txt", header = None, delim_whitespace=True)
nvqv_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nvqv.txt", header = None, delim_whitespace=True)
nw500_0619 = pd.read_csv("../MWST_99per/Test_Data/JJA_MWST_0619_nw500.txt", header = None, delim_whitespace=True)

indicator_0619 = pd.read_csv("../MWST_99per/JJA_MWST_99per_0619_indicator.txt", header = None, delim_whitespace=True)
test_set_y_initial = indicator_0619[4]
test_y = test_set_y_initial.to_numpy().reshape(1, test_set_y_initial.shape[0])

def train_som(x, y, input_len, sigma, learning_rate, data):
    som = MiniSom(x=x,
              y=y,
              input_len = input_len,
              sigma = sigma,
              learning_rate = learning_rate              
             )
    som.random_weights_init(data)
    start_time = time.time()
    som.train_random(data, iterations)
    elapsed_time = time.time() - start_time
    print(elapsed_time, "seconds")
    return som

def plot_som(som, data, target):
    plt.figure(figsize = (16,12))
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o','s','D']
    colors = ['r','g','b']
    for cnt, xx in enumerate(data):
        w = som.winner(xx)
        plot(w[0]+.5, w[1]+.5, markers[target[cnt]], markerfacecolor = 'None', markeredgecolor = colors[target[cnt]],
            markersize = 12, markeredgewidth = 2)
    axis([0,som._weights.shape[0], 0,som._weights.shape[1]])
    show()

def get_train_test(nex_names, ex_names, test_names):
    nex_combo_7905 = pd.concat(nex_names, axis = 1)
    nex_combo_7905["label"] = 0

    ex_combo_7905 = pd.concat(ex_names, axis = 1)
    ex_combo_7905["label"] = 1

    combo_7905_df = pd.concat([ex_combo_7905, nex_combo_7905])

    train_combo_y_initial = combo_7905_df['label'].to_numpy()
    train_combo_x = combo_7905_df.iloc[:,0:combo_7905_df.shape[1]-1].to_numpy()

    test_df = pd.concat(test_names, axis = 1)

    test_combo_x = test_df.to_numpy()
    train_combo_x, train_combo_y = shuffle(train_combo_x, train_combo_y_initial, random_state = 0)
    train_combo_y = train_combo_y.reshape(1, train_combo_y.shape[0])
    
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

#     over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.652)
#     steps = [('o', over), ('u', under)]
#     steps = [('u', under)]
#     pipeline = Pipeline(steps=steps)
    #transform the dataset
#     train_combo_x, train_combo_y = pipeline.fit_resample(train_combo_x, train_combo_y_initial)
    train_combo_x, train_combo_y = under.fit_resample(train_combo_x, train_combo_y_initial)
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


def classify(som, train_x, train_y, test_x):

    winmap = som.labels_map(train_x, train_y)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in test_x:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

nex_mapping = {
    0:nex_nvqv,
    1:nex_nv10m,
    2:nex_nv2m,
    3:nex_nw500,
    4:nex_nuqv,
    5:nex_nv500
}

ex_mapping = {
    0:ex_nvqv,
    1:ex_nv10m,
    2:ex_nv2m,
    3:ex_nw500,
    4:ex_nuqv,
    5:ex_nv500
}

mapping_0619 = {
    0:nvqv_0619,   
    1:nv10m_0619,  
    2:nv2m_0619,  
    3:nw500_0619,
    4:nuqv_0619,
    5:nv500_0619
}


mapping_text = {
    0:'nvqv', 
    1:'nv10m',
    2:'nv2m',
    3:'nw500',
    4:'nuqv',
    5:'nv500'
}

file1 = open("mwst_som_top6_results_4.txt", "w")

def som_combo(numofvar):
    comb = combinations([0,1,2,3,4,5],numofvar) 
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
            # print(mapping_text[combo[j]])
            file1.write(mapping_text[combo[j]]) 
            file1.write('\n')
            nex.append(nex_mapping[combo[j]])
            ex.append(ex_mapping[combo[j]])
            test.append(mapping_0619[combo[j]])
        
        for a in range(100):

            train_x, test_x, train_y = get_train_test_balanced(nex, ex, test)
            train_y = train_y.reshape((train_y.shape[1],))
            som = MiniSom(3, 4, train_x.shape[1], sigma=1.5, learning_rate=0.7, activation_distance='euclidean',
                      topology='hexagonal', neighborhood_function='gaussian', random_seed=10)

            som.train(train_x, 1000, verbose=True)
            result_train = classify(som, train_x, train_y, train_x)
            result_train = np.array(result_train)
            cm_train = conf_matrix(result_train.reshape((1, result_train.shape[0])), train_y.reshape(1, train_y.shape[0]))

            tpr_train, fpr_train, f1_train = evalmetrics(cm_train)
            tpr_train_avg.append(tpr_train)
            fpr_train_avg.append(fpr_train)
            f1_train_avg.append(f1_train)

            result = classify(som, train_x, train_y, test_x)
            result = np.array(result)
            cm_test = conf_matrix(result.reshape((1, result.shape[0])), test_y)
            confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (34)', 'actual_non-extreme (1229)'],
                                     columns=['predicted_extreme','predicted_non-extreme'])
            
            tpr, fpr, f1 = evalmetrics(cm_test)
            tpr_test_avg.append(tpr)
            fpr_test_avg.append(fpr)
            f1_test_avg.append(f1)

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


som_combo(4)
