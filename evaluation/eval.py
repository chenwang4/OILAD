#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:42:34 2023

"""
import numpy as np
from SequentialDecisionAD import Decision_Sequence_AD
from sklearn.ensemble import IsolationForest

win_len_q = 20
win_len_v = 60
step_size = 1
act_dim = 4
state_dim = 8

# step1: load the trained BC model and construct the feature generator 'detector'
def behavaiour_features(model):
    detector = Decision_Sequence_AD(model, act_dim, state_dim, win_len_q, win_len_v, step_size)  
    return detector

# step2: choose the isolation forest as the boundary model and fit it with normal training dataset
anomaly_algorithm = (
    "Isolation Forest",
    IsolationForest(contamination=0.0008, random_state=33),
)
def boundary_model(anomaly_algorithm, detector, X_train_norm, y_train):
    algorithm = anomaly_algorithm[1]
    q_train_online_score, v_train_online_score = detector.online_score(X_train_norm[:3000], y_train[:3000])
    normal_data = []
    for i in range(len(q_train_online_score)):
        for j in range(len(q_train_online_score[i])):
            normal_data.append(np.array([v_train_online_score[i][j], q_train_online_score[i][j]]))
    normal_data = np.array(normal_data)
    algorithm.fit(normal_data)
    return algorithm

#step3: evaluation
def traj_predict(test_size, q_score, v_score, algorithm):
    results = []
    for i in range(test_size):
        traj_feature = np.stack((v_score[i], q_score[i]), axis=1)
        result = algorithm.predict(traj_feature)
        results.append(1-np.all(result==1)) # normal:0, anomaly:1
    return np.array(results)

def evaluation(algorithm, detector, anomaly_type, X_test_norm, y_test, X_random_anomaly, y_random_anomaly, X_policy_anomaly, y_policy_anomaly, q_anomaly_score=None, v_anomaly_score=None):
    result_f1 = []
    result_pre = []
    result_recall = []
    if anomaly_type == 'perturbed':
        q_anomaly_score, v_anomaly_score = detector.online_score(X_random_anomaly, y_random_anomaly)
    elif anomaly_type == 'policy':
        q_anomaly_score, v_anomaly_score = detector.online_score(X_policy_anomaly, y_policy_anomaly)
    elif anomaly_type == 'specific':
        q_anomaly_score, v_anomaly_score = q_anomaly_score, v_anomaly_score 
    q_online_score, v_online_score = detector.online_score(X_test_norm, y_test)
    for i in range(5):  
        idx_normal = np.random.randint(len(q_online_score)-950)
        idx_anomaly = np.random.randint(len(q_anomaly_score)-50)
        y_pred_normal = traj_predict(950, q_online_score[idx_normal:idx_normal+950], v_online_score[idx_normal:idx_normal+950], algorithm)
        y_pred_anomaly = traj_predict(50, q_anomaly_score[idx_anomaly:idx_anomaly+50], v_anomaly_score[idx_anomaly:idx_anomaly+50], algorithm)
        tp = sum(y_pred_anomaly == 1)
        fp = sum(y_pred_normal == 1)
        print(tp, fp)
        fn = sum(y_pred_anomaly == 0)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(precision*recall/(precision+recall)) 
        result_f1.append(f1)
        result_pre.append(precision)
        result_recall.append(recall)
    print('F1', np.mean(result_f1), np.std(result_f1))
    print('precision', np.mean(result_pre), np.std(result_pre))
    print('recall', np.mean(result_recall), np.std(result_recall))
    return result_f1, result_pre, result_recall
 