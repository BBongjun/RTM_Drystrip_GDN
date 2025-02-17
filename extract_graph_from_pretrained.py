# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc, get_fc_graph_struc_from_map
from util.iostream import printsep

from datasets.TimeDataset import  WaferDataset, loader_wafer


from models.GDN import GDN
import networkx as nx

from train import train, train_sch
from test  import test, test_wafer_with_scores, test_thresholds
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm

import json
import random
#%%
class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'../preprocessed_datasets_GDN_stride1/train_data.csv')
       
        train = train_orig
        feature_map = train.columns.tolist()[1:-4]
        # print(feature_map)
        # print('feature 수:', len(feature_map))
        # fc_struc = get_fc_graph_struc(dataset)
        # fc_struc 생성

        fc_struc = get_fc_graph_struc_from_map(feature_map)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)
        self.feature_map = feature_map
        # ===============여기 까지는 graph 구조 정의하는 부분===================

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader, n_sensor= loader_wafer(args.data_dir, fc_edge_index, args.batch, label=True)
        print('Finish Data Load')

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)
        

        if not hasattr(self, "wafer_index_cache"):
            # Cache indices for quick lookup
            self.wafer_index_cache = {}
            for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers)):
                self.wafer_index_cache.setdefault((str(lid), str(wnum)), []).append(idx)


    def run(self):
        # 학습시켜 놓은거 있으면, train 생략함
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            print('Training start....')
            self.train_log, self.val_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        _, self.test_result = test(self.model, self.test_dataloader)
        _, self.val_result = test(self.model, self.val_dataloader)

        # self.test_validation_thresholds()
        self.threshold_based_val = 28.356702427293943

        # case1_wafers_ids = [('2024-06-26_688131L.00L',21), ('2024-06-28_689937L.00L',17), ('2024-07-02_690849L.00L',1)]
        case2_wafers_ids = [('2024-09-04_702814L.00L', 7),
        ('2024-07-08_692319L.00L', 9),
        ('2024-09-04_703060L.00L', 1),
        ('2024-06-24_687938L.00L', 9),
        ('2024-07-06_690643L.00L', 5),
        ('2024-06-21_689226L.00L', 5),
        ('2024-07-12_692316L.00L', 17),
        ('2024-07-12_692316L.00L', 5),
        ('2024-07-12_690935L.00L', 9),
        ('2024-07-12_693018L.00L', 9),
        ('2024-06-21_689226L.00L', 1),
        ('2024-07-14_693731L.00L', 9),
        ('2024-07-12_693018L.00L', 17),
        ('2024-07-14_693731L.00L', 5),
        ('2024-07-12_691144L.00L', 17),
        ('2024-06-22_689107L.00L', 5),
        ('2024-07-12_691144L.00L', 9),
        ('2024-06-28_688350L.00L', 6),
        ('2024-07-09_692623L.00L', 5)]
        # case4_wafers_ids = [('2024-06-26_689436L.00L',1), ('2024-07-08_692319L.00L',21),('2024-07-08_692416L.00L',1)]
        case4_wafers_ids =[('2024-06-26_689436L.00L',1)]
        # self.extract_only_graph_case1(case1_wafers_ids)
        # self.extract_only_graph_case2(self.test_result, self.val_result,case2_wafers_ids)
        # self.extract_only_graph_case4(case4_wafers_ids)
        self.extract_only_graph_case4_onlyanomaly(case4_wafers_ids)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        # np_test_result = np.array(test_result)
        # np_val_result = np.array(val_result)

        # test_labels = np_test_result[2, :, 0].tolist()
        np_test_result = np.array(test_result[:3])  # 예측값, 실제값, 라벨만 numpy 배열로 변환
        np_val_result = np.array(val_result[:3])

        # 라벨 추출 (2차원 배열에서 첫 번째 축만 사용)
        test_labels = np_test_result[2][:, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        # self.threshold_based_val = np.max(normal_scores)
        # print(self.threshold_based_val)
        print('=========================**Window 단위 Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info
            
        self.threshold_based_val = info[4] #threshold
        print('*Anomaly Threshold :',self.threshold_based_val)

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths
    
    def get_wafer_data(self, lotid, wafer_number):
        """
        특정 Lotid와 WaferNum에 해당하는 웨이퍼 데이터 추출
        Args:
            lotid: 웨이퍼 Lotid
            wafer_number: 웨이퍼 번호
        Returns:
            wafer_data: 해당 웨이퍼의 데이터 리스트 (feature, y, label, edge_index, lotid, wafernum)
        """
        # Step 2: Retrieve indices for the given lotid and wafer_number
        key = (str(lotid), str(wafer_number))
        wafer_indices = self.wafer_index_cache.get(key, [])

        # Step 3: Log warning if no data is found
        if not wafer_indices:
            print(f"Warning: No data available for Lotid {lotid}, WaferNum {wafer_number}")
            return []

        # Step 4: Retrieve wafer data
        wafer_data = [self.test_dataset[idx] for idx in wafer_indices]
        return wafer_data

    def visualize_learned_graph(self, wafer_name,window_name, threshold=0.1):
        """
        특정 윈도우의 Attention Graph를 시각화
        Args:
            model_path (str): 모델 경로 (.pt 파일)
            window_name (str): 윈도우 이름 (저장 파일 이름에 사용)
            threshold (float): Attention weight를 시각화할 임계값
        """
        import networkx as nx

        # Step 1: Attention Graph 데이터 추출
        att_edge_index = self.model.att_edge_index
        attention_weights = self.model.extract_attention_weights()[0].squeeze().cpu().numpy()  # Attention Weights
        G = nx.DiGraph()

        # 노드 추가
        for i, sensor_name in enumerate(self.feature_map):
            G.add_node(sensor_name)

        # 엣지 추가 (threshold 기반)
        for edge_idx in range(att_edge_index.shape[1]):
            src, dst = att_edge_index[0, edge_idx], att_edge_index[1, edge_idx]
            weight = attention_weights[edge_idx]
            # if abs(weight) > threshold:  # Threshold 검사
            G.add_edge(self.feature_map[src], self.feature_map[dst], weight=weight)

        # Step 2: 그래프 시각화
        pos = nx.spring_layout(G, seed=42, k=1.2)  # Force-directed Layout
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10,
                edge_color=weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2, arrowstyle='->')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in zip(edges, weights)}, font_size=10)

        # 폴더 경로 생성 및 저장
        directory = f'./graph_figures/{wafer_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.title(f"Attention Graph for {wafer_name}_window_{window_name}")
        plt.savefig(os.path.join(directory, f"{window_name}.png"))
        plt.show()
        plt.close()

    def extract_only_graph_case1(self, wafers_ids):
        """
        Args:
            wafers_ids: 웨이퍼 리스트 [(lotid, wafer_number, score)].
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수.
            n_random_windows: 랜덤으로 선택할 시점의 수.
        """
        for lotid, wafer_number in wafers_ids:
            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)
            
            # Step 4: 모든 시점의 그래프 생성
            for idx in range(len(wafer_data)):
                feature, _, _, edge_index, _, _, step_num = wafer_data[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))
                    self.visualize_learned_graph(
                        f"Case1_Lotid_{lotid}_WaferNum_{wafer_number}",f"graph_Window_{idx}_step{step_num}"
                    )

    def extract_only_graph_case2(self, test_result, val_result,wafers_ids):
        """
        Args:
            wafers_ids: 웨이퍼 리스트 [(lotid, wafer_number, score)].
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수.
            n_random_windows: 랜덤으로 선택할 시점의 수.
        """
        # Step 1: 에러 점수 계산 및 Threshold 결정
        total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)

        # Step 2: 이상 탐지된 Wafer 확인
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        predicted_anomaly_indices = np.argmax(total_err_scores, axis=0)  # max값의 변수 인덱스 추출

        correct_anomaly_indices = np.where((predicted_scores > self.threshold_based_val))[0]
        
        for lotid, wafer_number in wafers_ids:
            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)

                        # False Positive 윈도우 찾기
            wafer_indices = [
                idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
                if lid == lotid and wnum == wafer_number
            ]
            # Correct anomaly windows (test dataset의 idx)
            correct_anomaly_windows = [idx for idx in wafer_indices if idx in correct_anomaly_indices]
            # New: correct_anomaly_windows를 웨이퍼의 step으로 변환
            correct_anomaly_steps = [
                wafer_indices.index(idx) for idx in correct_anomaly_windows
            ]
            
            # Step 4: 모든 시점의 그래프 생성
            for idx in correct_anomaly_steps:
                feature, _, _, edge_index, _, _, step_num = wafer_data[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))
                    self.visualize_learned_graph(
                        f"Case2_Lotid_{lotid}_WaferNum_{wafer_number}", f"graph_Window_{idx}_step{step_num}_anomaly"
                    )

    def extract_only_graph_case4(self, wafers_ids):
        """
        Args:
            wafers_ids: 웨이퍼 리스트 [(lotid, wafer_number, score)].
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수.
            n_random_windows: 랜덤으로 선택할 시점의 수.
        """
        for lotid, wafer_number in wafers_ids:
            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)
            
            # Step 4: 모든 시점의 그래프 생성
            for idx in range(len(wafer_data)):
                feature, _, _, edge_index, _, _, step_num = wafer_data[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))
                    self.visualize_learned_graph(
                        f"Case4_Lotid_{lotid}_WaferNum_{wafer_number}", f"graph_Window_{idx}_step{step_num}"
                    )
    def extract_only_graph_case4_onlyanomaly(self, wafers_ids):
        """
        Args:
            wafers_ids: 웨이퍼 리스트 [(lotid, wafer_number, score)].
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수.
            n_random_windows: 랜덤으로 선택할 시점의 수.
        """
        # Step 1: 에러 점수 계산 및 Threshold 결정
        total_err_scores, normal_scores = get_full_err_scores(self.test_result, self.val_result)

        # Step 2: 이상 탐지된 Wafer 확인
        true_labels = np.array(self.test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        predicted_anomaly_indices = np.argmax(total_err_scores, axis=0)  # max값의 변수 인덱스 추출

        correct_anomaly_indices = np.where((predicted_scores > self.threshold_based_val))[0]
        
        for lotid, wafer_number in wafers_ids:
            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)

                        # False Positive 윈도우 찾기
            wafer_indices = [
                idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
                if lid == lotid and wnum == wafer_number
            ]
            # Correct anomaly windows (test dataset의 idx)
            correct_anomaly_windows = [idx for idx in wafer_indices if idx in correct_anomaly_indices]
            # New: correct_anomaly_windows를 웨이퍼의 step으로 변환
            correct_anomaly_steps = [
                wafer_indices.index(idx) for idx in correct_anomaly_windows
            ]
            
            # Step 4: 모든 시점의 그래프 생성
            for idx in correct_anomaly_steps:
                feature, _, _, edge_index, _, _, step_num = wafer_data[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))
                    self.visualize_learned_graph(
                        f"Case4_Lotid_{lotid}_WaferNum_{wafer_number}", f"graph_Window_{idx}_step{step_num}_anomaly"
                    )

    def test_wafer_with_scores(self, threshold):
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
        total_err_scores, normal_scores = get_full_err_scores(self.test_result, self.val_result)

        # Step 2: Wafer 단위 결과 집계
        test_labels = np.array(self.test_result[2])[:, 0]  # 각 Window의 실제 라벨
        lot_ids = np.array(self.test_result[3])  # 각 Window의 Lot ID
        wafer_numbers = np.array(self.test_result[4])  # 각 Window의 Wafer 번호

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

        # Step 6: 결과 반환
        return {
            "wafer_precision": wafer_precision,
            "wafer_recall": wafer_recall,
            "wafer_f1": wafer_f1,
            "wafer_roc_auc": wafer_roc_auc,
            "wafer_accuracy": wafer_accuracy,
            "wafer_specificity": wafer_specificity,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        }
    def test_validation_thresholds(self):
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

        total_err_scores, normal_scores = get_full_err_scores(self.test_result, self.val_result)
        normal_max_scores = np.max(normal_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산

        # 70~100 percentile을 2단위로 설정
        percentiles = np.arange(98, 100, 0.1)  # 70, 72, ..., 100
        threshold_based_val = np.percentile(normal_max_scores, percentiles)
        print(threshold_based_val)

        # 결과 저장용 리스트
        results = []

        for percentile, threshold in zip(percentiles,threshold_based_val):
            metrics = self.test_wafer_with_scores(threshold)
            metrics["threshold"] = threshold
            metrics["percentiles"] = percentile
            results.append(metrics)

        # 데이터프레임 변환
        results_df = pd.DataFrame(results)

        # 결과 저장
        results_df.to_csv("./results/wafer_threshold_evaluation_validation_percentile.csv", index=False)
        print("Saved threshold evaluation results to wafer_threshold_evaluation.csv")

        return results_df
def extract_time_series(wafer_data, model, device):
    time_series_actual = []
    time_series_predicted = []
    step_num_list = []

    for feature, y, _, edge_index, _, _, step_num in wafer_data:
        feature = feature.unsqueeze(0).to(device)
        edge_index = edge_index.to(device)
        model.eval()
        with torch.no_grad():
            predicted = model(feature, edge_index).cpu().detach().numpy()[0]
        time_series_actual.append(y.numpy())
        time_series_predicted.append(predicted)
        step_num_list.append(step_num)

    return np.array(time_series_actual), np.array(time_series_predicted), np.array(step_num_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../preprocessed_datasets_GDN_stride1_filter_window30', help='Location of datasets.')
    parser.add_argument('-batch', help='batch size', type = int, default=256)
    parser.add_argument('-epoch', help='train epoch', type = int, default=200)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=30)
    parser.add_argument('-dim', help='dimension', type = int, default=128)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=10)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=42)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=128)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=5)
    parser.add_argument('-report', help='best / val', type = str, default='val')
    parser.add_argument('-load_model', help='0: train / 1: only test', type=int, default=1)
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='./pretrained/best_02|10-04:15:38.pt')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    if args.load_model:
        print('load pretrained model')
    else:
        args.load_model_path = ''
        print('train 시작')

    env_config={ 
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)  
      
    main.run()
        





