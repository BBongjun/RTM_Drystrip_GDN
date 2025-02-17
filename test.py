import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

from evaluate import *
from util.data import *
from util.preprocess import *

from tqdm import tqdm

def test(model, dataloader):
    """
    테스트 함수. 모델의 예측 결과와 추가 정보를 반환.
    Args:
        model: PyTorch 모델.
        dataloader: 데이터로더.
    Returns:
        avg_loss: 평균 손실.
        test_result: [예측값, 실제값, 라벨, Lot ID, Wafer Number, Step Number].
    """
    # 손실 함수 및 디바이스 설정
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()
    test_loss_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    # 추가 정보 저장
    lotid_list = []
    wafernum_list = []
    stepnum_list = []

    model.eval()

    # tqdm 추가
    with tqdm(total=len(dataloader), desc="Testing") as pbar:
        for batch in dataloader:
            # 데이터 분리
            x, y, labels, edge_index, lotid, wafer_number, step_num = batch

            # 텐서 데이터만 GPU로 이동
            x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

            with torch.no_grad():
                # 모델 예측
                predicted = model(x, edge_index).float().to(device)
                loss = loss_func(predicted, y)

                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

                if len(t_test_predicted_list) == 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                    t_test_labels_list = labels
                else:
                    t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                    t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

                # 추가 정보 저장 (리스트 그대로 사용)
                lotid_list.extend(lotid)
                wafernum_list.extend(wafer_number)
                stepnum_list.extend(step_num)

            test_loss_list.append(loss.item())
            pbar.update(1)

    # 리스트를 변환
    test_predicted_list = t_test_predicted_list.cpu().numpy().tolist()
    test_ground_list = t_test_ground_list.cpu().numpy().tolist()
    test_labels_list = t_test_labels_list.cpu().numpy().tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    # test_result에 Lot ID, Wafer Number, Step Number 추가
    test_result = [
        test_predicted_list,
        test_ground_list,
        test_labels_list,
        lotid_list,
        wafernum_list,
        stepnum_list
    ]

    return avg_loss, test_result



# def test(model, dataloader):
#     # test
#     loss_func = nn.MSELoss(reduction='mean')
#     device = get_device()
#     test_loss_list = []
#     now = time.time()

#     test_predicted_list = []
#     test_ground_list = []
#     test_labels_list = []

#     t_test_predicted_list = []
#     t_test_ground_list = []
#     t_test_labels_list = []

#     test_len = len(dataloader)

#     model.eval()

#     i = 0
#     acu_loss = 0

#     # tqdm 추가
#     with tqdm(total=len(dataloader), desc="Testing") as pbar:
#         for x, y, labels, edge_index, lotid, wafer_number, step_num in dataloader:
#             x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
            
#             with torch.no_grad():
#                 predicted = model(x, edge_index).float().to(device)
#                 loss = loss_func(predicted, y)

#                 labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

#                 if len(t_test_predicted_list) <= 0:
#                     t_test_predicted_list = predicted
#                     t_test_ground_list = y
#                     t_test_labels_list = labels
#                 else:
#                     t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
#                     t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
#                     t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
            
#             test_loss_list.append(loss.item())
#             acu_loss += loss.item()
            
#             i += 1

#             if i % 10000 == 1 and i > 1:
#                 print(timeSincePlus(now, i / test_len))
            
#             # tqdm 업데이트
#             pbar.set_postfix({'Batch Loss': f"{loss.item():.6f}"})
#             pbar.update(1)

#     test_predicted_list = t_test_predicted_list.tolist()        
#     test_ground_list = t_test_ground_list.tolist()        
#     test_labels_list = t_test_labels_list.tolist()      
    
#     avg_loss = sum(test_loss_list)/len(test_loss_list)

#     return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]


# def test_wafer_with_scores(test_result, val_result, threshold):
#     """
#     Wafer 단위 평가 함수.
#     Args:
#         test_result: 테스트 데이터의 모델 예측 결과.
#         val_result: Validation 데이터의 모델 예측 결과.
#         threshold: anomaly detection threshold.
#     Returns:
#         wafer_metrics: Wafer 단위 평가 결과 (precision, recall, f1, roc_auc 등 다양한 지표).
#     """
#     # Step 1: 에러 점수 계산
#     total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)

#     # Step 2: Wafer 단위 결과 집계
#     test_labels = np.array(test_result[2])[:, 0]  # 각 Window의 실제 라벨
#     lot_ids = np.array(test_result[3])  # 각 Window의 Lot ID
#     wafer_numbers = np.array(test_result[4])  # 각 Window의 Wafer 번호

#     # 웨이퍼 단위로 그룹화
#     wafer_ids = [(lot_id, wafer_number) for lot_id, wafer_number in zip(lot_ids, wafer_numbers)]
#     unique_wafers = list(set(wafer_ids))

#     wafer_true_labels = []
#     wafer_predicted_scores = []
#     wafer_predicted_labels = []

#     for wafer_id in unique_wafers:
#         # 해당 Wafer의 모든 Window 가져오기
#         wafer_indices = [i for i, wid in enumerate(wafer_ids) if wid == wafer_id]
#         wafer_scores = np.max(total_err_scores[:, wafer_indices], axis=0)  # Top-K Feature 기반 합산
#         wafer_labels = test_labels[wafer_indices]

#         # Wafer의 true label (하나라도 이상이 있으면 1)
#         wafer_true_labels.append(1 if any(wafer_labels) else 0)

#         # Wafer의 predicted score (가장 높은 anomaly score 사용)
#         max_wafer_score = max(wafer_scores)
#         wafer_predicted_scores.append(max_wafer_score)

#         # Wafer의 predicted label (threshold 기반)
#         wafer_predicted_labels.append(1 if max_wafer_score > threshold else 0)

#     # Step 3: ROC AUC 예외 처리
#     if len(set(wafer_true_labels)) < 2:  # 클래스가 하나만 존재하면 ROC AUC 계산 불가
#         wafer_roc_auc = None
#         print("Warning: Only one class present in true labels. ROC AUC is undefined.")
#     else:
#         wafer_roc_auc = roc_auc_score(wafer_true_labels, wafer_predicted_scores)

#     # Step 4: Confusion Matrix 기반 지표 계산
#     cm = confusion_matrix(wafer_true_labels, wafer_predicted_labels)
#     tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

#     # Precision, Recall, F1-Score
#     wafer_precision = precision_score(wafer_true_labels, wafer_predicted_labels, zero_division=0)
#     wafer_recall = recall_score(wafer_true_labels, wafer_predicted_labels, zero_division=0)
#     wafer_f1 = f1_score(wafer_true_labels, wafer_predicted_labels, zero_division=0)

#     # Additional Metrics
#     wafer_accuracy = (tp + tn) / (tp + tn + fp + fn)
#     wafer_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

#     # Step 5: 지표 출력
#     print(f"Wafer-Level Metrics:")
#     print(f"Precision: {wafer_precision:.4f}, Recall: {wafer_recall:.4f}, F1-Score: {wafer_f1:.4f}")
#     print(f"ROC-AUC: {wafer_roc_auc}")
#     print(f"Accuracy: {wafer_accuracy:.4f}, Specificity: {wafer_specificity:.4f}")
#     print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

#     false_positive_wafers = get_false_positive_wafers(
#         wafer_true_labels, 
#         wafer_predicted_labels, 
#         unique_wafers, 
#         wafer_predicted_scores
#     )

#     # Step 6: 결과 반환
#     return {
#         "wafer_precision": wafer_precision,
#         "wafer_recall": wafer_recall,
#         "wafer_f1": wafer_f1,
#         "wafer_roc_auc": wafer_roc_auc,
#         "wafer_accuracy": wafer_accuracy,
#         "wafer_specificity": wafer_specificity,
#         "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
#     }, false_positive_wafers

# True Positives (TP): 실제로 이상인 웨이퍼 중 올바르게 탐지된 수
# False Positives (FP): 실제로 정상인 웨이퍼를 이상으로 잘못 탐지한 수
# True Negatives (TN): 실제로 정상인 웨이퍼를 정상으로 탐지한 수
# False Negatives (FN): 실제로 이상인 웨이퍼를 탐지하지 못한 수
# Accuracy: 정확도
# Specificity: 특이도 (정상을 정상으로 분류한 비율)

def get_false_positive_wafers(wafer_true_labels, wafer_predicted_labels, unique_wafers, wafer_predicted_scores):
    """
    False Positive에 해당하는 Wafer의 ID를 반환.
    Args:
        wafer_true_labels (list): 실제 라벨.
        wafer_predicted_labels (list): 예측 라벨.
        unique_wafers (list): 웨이퍼 ID 리스트 (lotid, wafer_number).
        wafer_predicted_scores (list): 웨이퍼 단위 anomaly score.
    Returns:
        false_positive_wafers (list): False Positive에 해당하는 Wafer의 (lotid, wafer_number, score).
    """
    false_positive_wafers = []
    
    for i, (true_label, pred_label) in enumerate(zip(wafer_true_labels, wafer_predicted_labels)):
        if true_label == 0 and pred_label == 1:  # False Positive 조건
            false_positive_wafers.append((unique_wafers[i][0], unique_wafers[i][1], wafer_predicted_scores[i]))

    return false_positive_wafers



def test_wafer_with_scores(test_result, val_result, threshold):
    """
    Wafer 단위 평가 함수.
    Args:
        test_result: 테스트 데이터의 모델 예측 결과.
        val_result: Validation 데이터의 모델 예측 결과.
        threshold: anomaly detection threshold.
    Returns:
        wafer_metrics: Wafer 단위 평가 결과 (precision, recall, f1, roc_auc 등 다양한 지표).
    """
    # Step 1: 에러 점수 계산
    total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)

    # Step 2: Wafer 단위 결과 집계
    test_labels = np.array(test_result[2])[:, 0]  # 각 Window의 실제 라벨
    lot_ids = np.array(test_result[3])  # 각 Window의 Lot ID
    wafer_numbers = np.array(test_result[4])  # 각 Window의 Wafer 번호

    # 웨이퍼 단위로 그룹화
    wafer_ids = [(lot_id, wafer_number) for lot_id, wafer_number in zip(lot_ids, wafer_numbers)]
    unique_wafers = list(set(wafer_ids))

    wafer_true_labels = []
    wafer_predicted_scores = []
    wafer_predicted_labels = []

    for wafer_id in unique_wafers:
        # 해당 Wafer의 모든 Window 가져오기
        wafer_indices = [i for i, wid in enumerate(wafer_ids) if wid == wafer_id]
        wafer_scores = np.max(total_err_scores[:, wafer_indices], axis=0)  # Top-K Feature 기반 합산
        wafer_labels = test_labels[wafer_indices]

        # Wafer의 true label (하나라도 이상이 있으면 1)
        wafer_true_labels.append(1 if any(wafer_labels) else 0)

        # Wafer의 predicted score (가장 높은 anomaly score 사용)
        max_wafer_score = max(wafer_scores)
        wafer_predicted_scores.append(max_wafer_score)

        # Wafer의 predicted label (threshold 기반)
        wafer_predicted_labels.append(1 if max_wafer_score > threshold else 0)

    # Step 3: ROC AUC 예외 처리
    if len(set(wafer_true_labels)) < 2:  # 클래스가 하나만 존재하면 ROC AUC 계산 불가
        wafer_roc_auc = None
        print("Warning: Only one class present in true labels. ROC AUC is undefined.")
    else:
        wafer_roc_auc = roc_auc_score(wafer_true_labels, wafer_predicted_scores)

    # Step 4: Confusion Matrix 기반 지표 계산
    cm = confusion_matrix(wafer_true_labels, wafer_predicted_labels)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

    # Precision, Recall, F1-Score
    wafer_precision = precision_score(wafer_true_labels, wafer_predicted_labels, zero_division=0)
    wafer_recall = recall_score(wafer_true_labels, wafer_predicted_labels, zero_division=0)
    wafer_f1 = f1_score(wafer_true_labels, wafer_predicted_labels, zero_division=0)

    # Additional Metrics
    wafer_accuracy = (tp + tn) / (tp + tn + fp + fn)
    wafer_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Step 5: 지표 출력
    print(f"Wafer-Level Metrics:")
    print(f"Precision: {wafer_precision:.4f}, Recall: {wafer_recall:.4f}, F1-Score: {wafer_f1:.4f}")
    print(f"ROC-AUC: {wafer_roc_auc}")
    print(f"Accuracy: {wafer_accuracy:.4f}, Specificity: {wafer_specificity:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    false_positive_wafers = get_false_positive_wafers(
        wafer_true_labels, 
        wafer_predicted_labels, 
        unique_wafers, 
        wafer_predicted_scores
    )

    # Step 6: 결과 반환
    return {
        "wafer_precision": wafer_precision,
        "wafer_recall": wafer_recall,
        "wafer_f1": wafer_f1,
        "wafer_roc_auc": wafer_roc_auc,
        "wafer_accuracy": wafer_accuracy,
        "wafer_specificity": wafer_specificity,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    }, false_positive_wafers

def test_thresholds(test_result, val_result, max_threshold,num_thresholds=30):
    """
    여러 threshold 값에 대해 Wafer 단위 평가를 수행하고 결과를 저장.
    Args:
        test_result: 테스트 데이터의 모델 예측 결과.
        val_result: Validation 데이터의 모델 예측 결과.
        num_thresholds: threshold를 나눌 구간의 수.
    Returns:
        DataFrame: 각 threshold에 대한 평가 결과.
    """
    import pandas as pd

    # Validation 데이터에서 anomaly score의 최대값
    # total_err_scores, _ = get_full_err_scores(test_result, val_result)
    # max_threshold = np.max(total_err_scores)

    # Threshold 범위 설정
    thresholds = np.linspace(5, max_threshold, num_thresholds)

    # 결과 저장용 리스트
    results = []

    for threshold in thresholds:
        metrics, _ = test_wafer_with_scores(test_result, val_result, threshold)
        metrics["threshold"] = threshold
        results.append(metrics)

    # 데이터프레임 변환
    results_df = pd.DataFrame(results)

    # 결과 저장
    results_df.to_csv("./results/wafer_threshold_evaluation_basesd_valths.csv", index=False)
    print("Saved threshold evaluation results to wafer_threshold_evaluation.csv")

    return results_df