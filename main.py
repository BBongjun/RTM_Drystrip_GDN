# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

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
        best_model = self.model.to(self.device)

        print('test result')
        _, self.test_result = test(best_model, self.test_dataloader)
        # print(self.test_result)
        # np.save(f'./results/test_result_{args.topk}.npy', np.array(self.test_result))

        print('val result')
        _, self.val_result = test(best_model, self.val_dataloader)
        # np.save(f'./results/val_result_{args.topk}.npy', np.array(self.val_result))

        self.get_score(self.test_result, self.val_result)

        
        print('=========================**Wafer 단위 Result **============================\n')
        # self.threshold_based_val = 11.450383
        self.threshold_based_val = 28.356702427293943
        wafer_metrics, false_positive_wafers = test_wafer_with_scores(self.test_result, self.val_result, threshold=self.threshold_based_val)

        # 이상 예측 시점 윈도우의 graph 시각화
        # self.visualize_anomaly_attention(self.test_result , self.val_result)

        # 시계열 시각화 및 graph 시각화
        # self.visualize_correctly_predicted_normal_wafers(self.test_result, n_random_wafers=10)
        # self.visualize_fully_normal_wafers(self.test_result, n_random_wafers=10)


        #-------------------------------------사용중-------------------------------------
        # embedding 시각화
        print('임베딩 시각화...')
        self.vis_sensor_embedding_based_tsne()
        self.vis_embedding_based_similarity() # 코사인 유사도 기반 노드 임베딩 시각화


        # print('=========================** threshold 별 테스트 결과 취합 **============================\n')
        # test_thresholds(self.test_result, self.val_result, self.threshold_based_val, num_thresholds=100)

        print('시계열 시각화')
        # self.threshold_based_val = 11.450383/
        
        # # 시각화 확인용
        self.visualize_case1(self.test_result, self.val_result,n_random_wafers=5) # 모두 정상, 모델 예측 정상
        self.visualize_case2(self.test_result, self.val_result,false_positive_wafers) # 모두 정상 , 모델 예측 이상 포함
        self.visualize_case3(self.test_result, self.val_result,n_random_wafers=5) # 모두 이상, 모델 예측 모두 정상
        self.visualize_case4(self.test_result, self.val_result) # 모두 이상, 모델 예측 이상 포함

        # print('추가 시각화')
        # self.visualize_step5_relationship_and_timeseries(self.test_result, n_random_wafers=5)  # step5에 해당하는 시계열+그래프 그리기
        # self.visualize_step3_4_relationship_and_timeseries(lotid='2024-07-11_690720L.00L', wafernum= 11)   # 특정 lotid, wafernum의 step3,4에 대해서 시계열+그래프

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
    
    def vis_sensor_embedding_based_tsne(self):
        """
        테스트 데이터셋에서 모델 임베딩 추출 및 t-SNE 시각화
        Args:
            path: 사전 학습된 모델 경로
        """
        best_model = self.model.to(self.device)
        best_model.eval()
        embeddings = []
        with torch.no_grad():
            for x, _, _, edge_index, lotid, wafernum, step_num in self.test_dataloader:
                x = x.to(self.device)
                edge_index = edge_index.to(self.device)
                
                # 임베딩 추출 과정
                all_embeddings = best_model.embedding(torch.arange(x.size(1)).to(self.device))
                embeddings.append(all_embeddings.cpu().numpy())
                break  # 한 번만 실행 (13개 센서의 임베딩만 추출)

        embeddings = np.vstack(embeddings)  # 결과 병합
        # print("Embedding Shape:", embeddings.shape)
        # print(embeddings)
        
        # 임베딩 저장
        np.save("embeddings.npy", embeddings)
        print("Embeddings saved to embeddings.npy")

        from sklearn.manifold import TSNE        
        from sklearn.cluster import KMeans

        # t-SNE 시각화
        print("Running t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)  # (13, 2)

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(reduced_embeddings)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
        for i, sensor_name in enumerate(self.feature_map):
            plt.annotate(sensor_name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9)

        plt.title("t-SNE with KMeans Clustering")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.savefig("tsne_sensor_embeddings_clustering.png")
        plt.close()
        
        # 센서 시각화
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=100, c='blue', alpha=0.7)
        for i, sensor_name in enumerate(self.feature_map):
            plt.annotate(sensor_name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9)

        plt.title("t-SNE Visualization of Sensor Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        # plt.grid(True)
        plt.savefig("tsne_sensor_embeddings.png")
        print("t-SNE visualization saved to tsne_sensor_embeddings.png")
        plt.show()

    # 센서 임베딩을 유사도 계산해서 표현
    # def vis_embedding_based_similarity(self, path):
    def vis_embedding_based_similarity(self):
        """
        테스트 데이터셋에서 모델 임베딩 추출 및 t-SNE 시각화
        """
        best_model = self.model.to(self.device)
        best_model.eval()

        embeddings = []
        with torch.no_grad():
            for x, _, _, edge_index, lotid, wafernum, step_num in self.test_dataloader:
                x = x.to(self.device)
                edge_index = edge_index.to(self.device)
                
                # 임베딩 추출 과정
                all_embeddings = best_model.embedding(torch.arange(x.size(1)).to(self.device))
                embeddings.append(all_embeddings.cpu().numpy())
                break  # 한 번만 실행 (13개 센서의 임베딩만 추출)

        embeddings = np.vstack(embeddings)  # 결과 병합
        
        # 임베딩 저장
        np.save("embeddings.npy", embeddings)
        print("Embeddings saved to embeddings.npy")


        from sklearn.metrics.pairwise import cosine_similarity
        import seaborn as sns

        similarity_matrix = cosine_similarity(embeddings)
        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", xticklabels=self.feature_map, yticklabels=self.feature_map)
        plt.title("Sensor Similarity Matrix")
        plt.savefig("sensor_embeddings_similarity.png")
        plt.close()

        

        # 임계값 설정
        threshold = 0.4  # 엣지를 그릴 유사도 임계값
        G = nx.Graph()

        # 노드 추가
        for i, sensor_name in enumerate(self.feature_map):
            G.add_node(sensor_name)

        # 엣지 추가 (유사도 값이 threshold 이상인 경우만 추가)
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                if i != j and similarity_matrix[i, j] >= threshold:
                    G.add_edge(self.feature_map[i], self.feature_map[j], weight=similarity_matrix[i, j])

        # 그래프 시각화 개선
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=1.3)  # k 값을 조정하여 노드 간 간격 조정
        weights = nx.get_edge_attributes(G, 'weight')

        # 엣지 색상과 굵기를 유사도 값에 따라 설정
        edges, edge_weights = zip(*weights.items())
        edge_colors = [cm.Blues(weight) for weight in edge_weights]  # 컬러맵 사용

        nx.draw(
            G, pos, with_labels=True,
            node_color="skyblue", node_size=1200, font_size=12, font_color="black",
            edgelist=edges, edge_color=edge_weights, edge_cmap=plt.cm.Blues,
            width=[5 * weight for weight in edge_weights]  # 엣지 굵기 조절
        )

        # 엣지 레이블 (유사도 값)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={k: f"{v:.2f}" for k, v in weights.items()},
            font_size=10, font_color="red"
        )

        plt.title(f"Sensor Similarity Graph : threshold={threshold}")
        plt.savefig(f"Sensor_Similarity_Graph : threshold={threshold}.png")
        plt.show()

    # Wafer 하나 불러와서 예시로 하나 graph 그려보는 코드(프로토타입)
    def visualize_wafer_learned_graph(self, model_path, wafer_path, threshold=0.1):
        """
        특정 웨이퍼 데이터를 예측하고, self.learned_graph를 활용하여 그래프를 시각화하는 함수
        Args:
            model_path (str): 모델 경로 (.pt 파일)
            wafer_path (str): 특정 웨이퍼 데이터 경로 (.npz 파일)
            threshold (float): Attention weight를 시각화할 임계값 (기본값: 0.1)
        """
        
        directory = os.path.join(f'./graph_figure_topk{args.topk}', os.path.basename(wafer_path).split('.')[0])
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Step 1: 데이터 로드 및 전처리
        device = self.device
        data = np.load(wafer_path, allow_pickle=True)
        features = torch.FloatTensor(data['data']).to(device)  # (n_windows, window_size, n_nodes)
        edge_index = self.model.edge_index_sets[0].to(device)  # 그래프 구조
        window_size, node_num = features.shape[1], features.shape[2]

        print(f"Loaded wafer data: {wafer_path}, Windows: {features.shape[0]}, Sensors: {node_num}")
        model_save_path = model_path
        # 모델 로드
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.to(device)
        # Step 2: 모델 Forward 및 learned_graph, Attention weight 추출
        self.model.eval()
        with torch.no_grad():
            x = features[0].T.unsqueeze(0)  # 첫 윈도우만 사용 (Shape: (1, n_nodes, window_size))
            _ = self.model(x, edge_index)  # Forward pass
            # learned_graph = self.model.learned_graph  # 각 노드별 유사도가 높은 topk 연결 정보
            att_edge_index = self.model.att_edge_index
            attention_weights = self.model.extract_attention_weights()[0]  # 첫 레이어 Attention weights 추출
            attention_weights = attention_weights.squeeze().cpu().numpy()  # Shape: (topk,)
            # attention_weights = attention_weights.reshape(node_num, self.model.topk)  # (node_num, topk)
        # Step 3: 그래프 생성
        print("Visualizing Learned Graph...")
        G = nx.DiGraph()

        # 노드 추가
        for i, sensor_name in enumerate(self.feature_map):
            G.add_node(sensor_name)

        # 엣지 추가 (attention weight와 함께)
        for edge_idx in range(att_edge_index.shape[1]):
            src, dst = att_edge_index[0, edge_idx], att_edge_index[1, edge_idx]
            weight = attention_weights[edge_idx]
            G.add_edge(self.feature_map[src], self.feature_map[dst], weight=weight)

        # Step 4: 그래프 시각화
        pos = nx.spring_layout(G, seed=42,k=1)  # Force-directed Layout
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10,
                edge_color=weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2, arrowstyle='->')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in zip(edges, weights)}, font_size=10)

        # 폴더 경로가 없을 경우 생성
        plt.title(f"Learned Graph Visualization for Wafer: {os.path.basename(wafer_path)}")
        plt.savefig(os.path.join(directory, "learned_directed_graph.png"))
        plt.show()

        pos = nx.circular_layout(G)  # circular layout
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10,
                edge_color=weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2, arrowstyle='->')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in zip(edges, weights)}, font_size=8)
        plt.title(f"Learned Graph Visualization for Wafer: {os.path.basename(wafer_path)}")
        plt.savefig(os.path.join(directory, "learned_directed_graph_circular.png"))
        plt.show()

        print("Learned Graph visualization saved'")


    def visualize_anomaly_attention(self, test_result, val_result):
        """
        모델이 이상으로 예측한 시점의 Attention Graph를 시각화 (논문의 Figure 3처럼)
        Args:
            test_result: 테스트 결과 (모델 출력, 실제 레이블)
            val_result: Validation 결과 (모델 출력, 실제 레이블)
        """
        # Step 1: 에러 점수 계산 및 Threshold 결정
        total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)
        threshold = np.max(normal_scores)  # Validation 데이터 기반 Threshold 설정
        print(f"Threshold automatically set to: {threshold:.4f}")

        # Step 2: 이상 시점 탐지
        true_labels = np.array(test_result[2])  # 실제 레이블
        true_labels = true_labels[:, 0]
        print(true_labels.sum())
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        anomaly_indices = np.where((predicted_scores > threshold) & (true_labels == 1))[0]  # 이상으로 예측한 시점
        print(f"Anomaly detected at windows: {anomaly_indices}")

        # Step 3: 이상 시점의 Attention Graph 시각화
        for idx in anomaly_indices:
            print(f"Visualizing attention graph for window {idx}")
            wafer_data = self.test_dataset[idx]
            feature, _, _, edge_index = wafer_data  # 해당 윈도우 데이터
            feature = feature.unsqueeze(0).to(self.device)

            # 모델 Forward Pass 및 Attention Graph 시각화
            with torch.no_grad():
                self.model(feature, edge_index.to(self.device))
                # Attention Graph 시각화 함수 호출
                self.visualize_learned_graph(self.env_config['load_model_path'], f"Window_{idx}")

    def visualize_learned_graph(self, model_path, window_name, threshold=0.1):
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
            if abs(weight) > threshold:  # Threshold 검사
                G.add_edge(self.feature_map[src], self.feature_map[dst], weight=weight)

        # Step 2: 그래프 시각화
        pos = nx.spring_layout(G, seed=42, k=1.2)  # Force-directed Layout
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10,
                edge_color=weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1, width=2, arrowstyle='->')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in zip(edges, weights)}, font_size=10)

        # 폴더 경로 생성 및 저장
        directory = os.path.join(f'./graph_figure_topk{args.topk}', window_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.title(f"Attention Graph for {window_name}")
        plt.savefig(os.path.join(directory, "attention_graph.png"))
        plt.show()
        plt.close()
        # print(f"Attention graph saved to {os.path.join(directory, 'attention_graph.png')}")
        
    def visualize_case4(self, test_result, val_result, n_normal_steps=5):
        """
        이상이 존재하고, 올바르게 예측된 Wafer에 대해 시계열 플롯 및 Attention Graph 생성
        Args:
            test_result: 테스트 결과 (모델 출력, 실제 레이블)
            val_result: Validation 결과 (모델 출력, 실제 레이블)
        """

        # Step 1: 에러 점수 계산 및 Threshold 결정
        total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)

        # Step 2: 이상 탐지된 Wafer 확인
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        # Anomaly로 분류된 변수 인덱스 추출 (각 윈도우별)
        predicted_anomaly_indices = np.argmax(total_err_scores, axis=0)  # max값의 변수 인덱스 추출

        correct_anomaly_indices = np.where((predicted_scores > self.threshold_based_val) & (true_labels == 1))[0]
        missed_anomaly_indices = np.where((predicted_scores <= self.threshold_based_val) & (true_labels == 1))[0]

        # 올바르게 탐지된 anomaly에서 변수 인덱스 추출
        correct_anomaly_variables = predicted_anomaly_indices[correct_anomaly_indices]


        print('====================================================')
        print(f"Total Anomalies: {len(np.where(true_labels == 1)[0])}")
        print(f"Correct Anomalies: {len(correct_anomaly_indices)}")
        print(f"Missed Anomalies: {len(missed_anomaly_indices)}")
        print('====================================================')

        if len(correct_anomaly_indices) == 0:
            print("No correct anomalies detected.")
            return

        # Step 3: 이상이 존재하는 Wafer의 모든 윈도우 추출
        wafers_with_anomalies = set(
            (self.test_dataset.lotids[idx], self.test_dataset.wafer_numbers[idx])
            for idx in correct_anomaly_indices
        )
        print(f"Correctly detected anomalies in {len(wafers_with_anomalies)} wafer(s).")

        for lotid, wafer_number in wafers_with_anomalies:
            print(f"Visualizing for Wafer Lotid {lotid}, WaferNum {wafer_number}")

            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)

            # 현재 웨이퍼에 대한 correct anomaly_steps 찾기(test dataset에서 idx)
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
            # Missed anomaly windows
            missed_anomaly_windows = [idx for idx in wafer_indices if idx in missed_anomaly_indices]
            anomaly_windows = correct_anomaly_windows + missed_anomaly_windows
            # 지금은 다 레이블이 1로 설정되어있기 때문에, anomaly_windows = 해당 wafer의 전체 windows
            # 단, 여기서 window length 90으로 잘려있는것부터 시작이므로, 맨 처음 실제값 90step은 무시되면서 진행됨!

            # Step 4: 정상 윈도우 선택
            normal_windows = []

            # anomaly_windows를 오름차순으로 정렬
            correct_anomaly_windows = sorted(correct_anomaly_windows)

            for anomaly_idx in correct_anomaly_windows:
                for step in range(1, n_normal_steps + 1):
                    normal_idx = anomaly_idx - step
                    # 정상 윈도우는 0 이상이고, anomaly_windows에 속하지 않아야 함
                    if normal_idx >= 0 and normal_idx not in correct_anomaly_windows:
                        normal_windows.append(normal_idx)

            # 정상 윈도우의 순서를 유지
            normal_windows = sorted(set(normal_windows))

            # Step 5: 실제값과 예측값 추출
            time_series_actual = []
            time_series_predicted = []
            step_num_list = []

            for feature, y, _, edge_index, _, _, step_num in wafer_data:
                feature = feature.unsqueeze(0).to(self.device)  # 입력 데이터
                edge_index = edge_index.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    predicted = self.model(feature, edge_index).cpu().detach().numpy()[0]  # 모델 예측값
                time_series_actual.append(y.numpy())  # 실제값
                time_series_predicted.append(predicted)
                step_num_list.append(step_num)

            time_series_actual = np.array(time_series_actual)
            time_series_predicted = np.array(time_series_predicted)
            step_num_list = np.array(step_num_list)

            # Step 6: 시계열 플롯 생성
            time_steps = np.arange(len(time_series_actual))
            fig, axes = plt.subplots(nrows=time_series_actual.shape[1] * 2, ncols=1, figsize=(15, 3 * time_series_actual.shape[1] * 2))
            fig.subplots_adjust(hspace=0.5)

            # Step 변화 지점 계산
            step_change_indices = np.where(np.diff(step_num_list) != 0)[0]# step 변화가 발생
            wafer_anomaly_scores = total_err_scores[:, wafer_indices]

            for sensor_idx in range(time_series_actual.shape[1]):
                # Time Series Plot
                axes[sensor_idx * 2].plot(time_steps, time_series_actual[:, sensor_idx], label="Actual", color='blue', linewidth=2)
                axes[sensor_idx * 2].plot(time_steps, time_series_predicted[:, sensor_idx], '--', label="Predicted", color='magenta', linewidth=2)
                axes[sensor_idx * 2].set_ylim(0-0.1, 1+0.1)

                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set
                # 이상 지점 색상 처리
                for step_idx, step in enumerate(correct_anomaly_steps):
                    if "Predicted Anomaly" not in legend_labels:
                        axes[sensor_idx * 2].axvspan(step - 0.5, step + 0.5, color='red', alpha=0.5, label="Predicted Anomaly")
                        legend_labels.add("Predicted Anomaly")
                    else:
                        axes[sensor_idx * 2].axvspan(step - 0.5, step + 0.5, color='red', alpha=0.5)

                    # if correct_anomaly_variables[step_idx] == sensor_idx: # 해당 센서가 anomaly 탐지에 기여
                    #     axes[sensor_idx * 2].annotate(
                    #         "",
                    #         xy=(step, 1.05),
                    #         xytext=(step, 1.2),
                    #         textcoords="data",
                    #         arrowprops=dict(
                    #             arrowstyle="wedge",
                    #             color="green",
                    #             lw=1.5,
                    #             shrinkA=0, shrinkB=10,
                    #         )
                    #     )

                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)

                axes[sensor_idx * 2].set_title(f"{self.feature_map[sensor_idx]}")
                axes[sensor_idx * 2].set_xlabel("Time Steps")
                axes[sensor_idx * 2].set_ylabel("Values")
                axes[sensor_idx * 2].legend()

                # Anomaly Score Plot
                axes[sensor_idx * 2 + 1].plot(time_steps, wafer_anomaly_scores[sensor_idx, :], label="Anomaly Score", color='orange', linewidth=2)
                axes[sensor_idx * 2 + 1].axhline(self.threshold_based_val, color='cyan', linestyle='--', label="Threshold", linewidth=2)  # Threshold 수평선 추가
                axes[sensor_idx * 2 + 1].set_ylim(-1, 30)
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)
                axes[sensor_idx * 2 + 1].set_title(f"{self.feature_map[sensor_idx]} - Anomaly Score")
                axes[sensor_idx * 2 + 1].set_xlabel("Time Steps")
                axes[sensor_idx * 2 + 1].set_ylabel("Score")
                axes[sensor_idx * 2 + 1].legend()

            # plt.suptitle(f"Wafer Lotid {lotid}, WaferNum {wafer_number}",  y=1.05, fontsize=16)
            plt.tight_layout()
            
            # 폴더 생성 및 저장
            directory = os.path.join(f'./graph_figure_topk{args.topk}', f"Wafer_Lotid_{lotid}_WaferNum_{wafer_number}")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, "time_series_with_anomalies.png"))
            plt.show()
            plt.close()

            # Step 7: Attention Graph 생성 (이상 시점에서)
            for idx in correct_anomaly_windows:
                feature, _, _, edge_index, _, _, _ = self.test_dataset[idx]
                # feature, _, _, edge_index,_,_, _ = self.test_dataset[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))

                    graph_name = f"Wafer_Lotid_{lotid}_WaferNum_{wafer_number}_Window_{idx}_Correct_Anomaly"
                    self.visualize_learned_graph(self.env_config['load_model_path'], graph_name)

            # Step 8: Attention Graph 생성 (정상 시점에서)
            for idx in normal_windows:
                feature, _, _, edge_index,_,_,_ = self.test_dataset[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))

                    self.visualize_learned_graph(self.env_config['load_model_path'], f"Wafer_Lotid_{lotid}_WaferNum_{wafer_number}_Window_{idx}_Normal")
            print(f"Finished: Wafer Lotid {lotid}, WaferNum {wafer_number}")


    def visualize_case1(self, test_result, val_result, n_random_wafers=5):
        """
        모든 윈도우가 정상인 웨이퍼 중에서 모델도 정상으로 예측한 웨이퍼에 대해 시계열 플롯 생성.
        """

        # Step 1: 에러 점수 계산 및 Threshold 결정
        total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)

        # Step 2: 이상 탐지된 Wafer 확인
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        predicted_labels = (predicted_scores > self.threshold_based_val).astype(int)

        fully_normal_wafers = set(
            (self.test_dataset.lotids[idx], self.test_dataset.wafer_numbers[idx])
            for idx, label in enumerate(true_labels)
            if label == 0
        )

        correctly_predicted_wafers = []
        for lotid, wafer_number in fully_normal_wafers:
            key = (str(lotid), str(wafer_number))
            wafer_indices = self.wafer_index_cache.get(key, [])
            # wafer_indices = [
            #     idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
            #     if lid == lotid and wnum == wafer_number
            # ]
            wafer_predicted_labels = predicted_labels[wafer_indices]
            wafer_true_labels = true_labels[wafer_indices]
            if np.all(wafer_predicted_labels == 0) and np.all(wafer_true_labels == 0):
                correctly_predicted_wafers.append((lotid, wafer_number))

        correctly_predicted_wafers = np.array(correctly_predicted_wafers)
        if len(correctly_predicted_wafers) > 1:
            selected_indices = np.random.choice(
                range(len(correctly_predicted_wafers)),
                size=min(n_random_wafers, len(correctly_predicted_wafers)),
                replace=False
            )
            selected_wafers = correctly_predicted_wafers[selected_indices]
        else:
            selected_wafers = correctly_predicted_wafers

        # Step 2: 시각화
        for lotid, wafer_number in selected_wafers:
            print(f"Visualizing for Fully Normal Wafer Lotid {lotid}, WaferNum {wafer_number}")

            wafer_data = self.get_wafer_data(lotid, wafer_number)
            key = (str(lotid), str(wafer_number))
            wafer_indices = self.wafer_index_cache.get(key, [])

            time_series_actual = []
            time_series_predicted = []
            step_num_list = []

            for feature, y, _, edge_index, _, _, step_num in wafer_data:
                feature = feature.unsqueeze(0).to(self.device)
                edge_index = edge_index.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    predicted = self.model(feature, edge_index).cpu().detach().numpy()[0]
                time_series_actual.append(y.numpy())
                time_series_predicted.append(predicted)
                step_num_list.append(step_num)

            time_series_actual = np.array(time_series_actual)
            time_series_predicted = np.array(time_series_predicted)
            step_num_list = np.array(step_num_list)

            # 현재 웨이퍼에 해당하는 anomaly scores 추출
            # wafer_indices = [
            #     idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
            #     if lid == lotid and wnum == wafer_number
            # ]
            wafer_anomaly_scores = total_err_scores[:,wafer_indices]
            # wafer_anomaly_scores = anomaly_scores[wafer_indices]

            # Step 3: 시계열 및 anomaly score 플롯 생성
            time_steps = np.arange(len(time_series_actual))
            fig, axes = plt.subplots(nrows=time_series_actual.shape[1] * 2, ncols=1, figsize=(15, 3 * time_series_actual.shape[1] * 2))
            fig.subplots_adjust(hspace=0.5)    
            # Step 변화 지점 계산
            step_change_indices = np.where(np.diff(step_num_list) != 0)[0]# step 변화가 발생

            for sensor_idx in range(time_series_actual.shape[1]):
                axes[sensor_idx * 2].plot(time_steps, time_series_actual[:, sensor_idx], label="Actual", color='blue', linewidth=2)
                axes[sensor_idx * 2].plot(time_steps, time_series_predicted[:, sensor_idx], '--', label="Predicted", color='magenta', linewidth=2)
                # Step 변화 시점에서 수직선 그리기
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)

                axes[sensor_idx * 2].set_ylim(-0.1, 1.1)
                axes[sensor_idx * 2].set_title(f"{self.feature_map[sensor_idx]}")
                axes[sensor_idx * 2].set_xlabel("Time Steps")
                axes[sensor_idx * 2].set_ylabel("Values")
                axes[sensor_idx * 2].legend()


                axes[sensor_idx * 2 + 1].plot(time_steps, wafer_anomaly_scores[sensor_idx, :], label="Anomaly Score", color='orange', linewidth=2)
                axes[sensor_idx * 2 + 1].axhline(self.threshold_based_val, color='cyan', linestyle='--', label="Threshold", linewidth=2)  # Threshold 수평선 추가
                # Step 변화 시점에서 수직선 그리기
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)
                axes[sensor_idx * 2 + 1].set_ylim(-1, 30)
                axes[sensor_idx * 2 + 1].set_title(f"{self.feature_map[sensor_idx]} - Anomaly Score")
                axes[sensor_idx * 2 + 1].set_xlabel("Time Steps")
                axes[sensor_idx * 2 + 1].set_ylabel("Score")
                axes[sensor_idx * 2 + 1].legend()

            # plt.suptitle(f"Time Series and Anomaly Scores for Fully Normal Wafer Lotid {lotid}, WaferNum {wafer_number}")
            plt.tight_layout()

            directory = os.path.join(f'./graph_figure_topk{args.topk}', f"Fully_Normal_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}")
            os.makedirs(directory, exist_ok=True)
            plt.savefig(os.path.join(directory, "time_series_and_scores_fully_normal.png"))
            plt.show()
            plt.close()

    def visualize_case2(self, test_result, val_result, false_positive_wafers, n_random_wafers=5, n_random_windows=5):
        """
        False Positive 웨이퍼에 대한 시각화 및 그래프 생성.
        Args:
            false_positive_wafers: False Positive 웨이퍼 리스트 [(lotid, wafer_number, score)].
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수.
            n_random_windows: 랜덤으로 선택할 시점의 수.
        """
        total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)
        # Step 2: 이상 탐지된 Wafer 확인
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        predicted_anomaly_indices = np.argmax(total_err_scores, axis=0)  # max값의 변수 인덱스 추출

        correct_anomaly_indices = np.where((predicted_scores > self.threshold_based_val))[0]
        
        # 올바르게 탐지된 anomaly에서 변수 인덱스 추출
        correct_anomaly_variables = predicted_anomaly_indices[correct_anomaly_indices]

        # Step 1: 에러 점수 계산 및 Threshold 결정
        if len(false_positive_wafers) == 0:
            print("False Positive 웨이퍼가 없습니다.")
            return

        print(f"총 False Positive 웨이퍼 수: {len(false_positive_wafers)}")

        if len(false_positive_wafers) > 1:
            # NumPy 배열로 변환
            false_positive_wafers = np.array(false_positive_wafers, dtype=object)

            # 랜덤으로 웨이퍼 선택
            # selected_fp_wafers = false_positive_wafers[
            #     np.random.choice(
            #         len(false_positive_wafers), 
            #         size=min(n_random_wafers, len(false_positive_wafers)), 
            #         replace=False
            #     )
            # ]
            selected_fp_wafers = false_positive_wafers
        else:
            selected_fp_wafers = false_positive_wafers

        for lotid, wafer_number, score in selected_fp_wafers:
            print(f"Visualizing False Positive Wafer Lotid {lotid}, WaferNum {wafer_number}, Score={score:.4f}")

            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)

            # Step 2: 실제값과 예측값 추출
            time_series_actual, time_series_predicted, step_num_list = extract_time_series(wafer_data, self.model, self.device)

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

            # 해당 웨이퍼의 anomaly scores 추출
            wafer_scores = total_err_scores[:, wafer_indices]  # 웨이퍼 인덱스를 사용해 점수 추출
            predicted_anomalies = np.where(wafer_scores > self.threshold_based_val)[0]  # Threshold 기반 anomaly windows

            # Step 3: 시계열 플롯 생성
            time_steps = np.arange(len(time_series_actual))
            fig, axes = plt.subplots(nrows=time_series_actual.shape[1] * 2, ncols=1, figsize=(15, 3 * time_series_actual.shape[1] * 2))
            fig.subplots_adjust(hspace=0.5)

            # Step 변화 지점 계산
            step_change_indices = np.where(np.diff(step_num_list) != 0)[0]

            for sensor_idx in range(time_series_actual.shape[1]):
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                # Time Series Plot
                axes[sensor_idx * 2].plot(time_steps, time_series_actual[:, sensor_idx], label="Actual", color='blue', linewidth=2)
                axes[sensor_idx * 2].plot(time_steps, time_series_predicted[:, sensor_idx], '--', label="Predicted", color='magenta', linewidth=2)
                axes[sensor_idx * 2].set_ylim(0 - 0.1, 1 + 0.1)
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)

                # legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set
                # 이상 지점 색상 처리
                for step_idx, step in enumerate(correct_anomaly_steps):
                    if "Predicted Anomaly" not in legend_labels:
                        axes[sensor_idx * 2].axvspan(step - 0.5, step + 0.5, color='red', alpha=0.5, label="Predicted Anomaly")
                        legend_labels.add("Predicted Anomaly")
                    else:
                        axes[sensor_idx * 2].axvspan(step - 0.5, step + 0.5, color='red', alpha=0.5)

                    if correct_anomaly_variables[step_idx] == sensor_idx: # 해당 센서가 anomaly 탐지에 기여
                        axes[sensor_idx * 2].annotate(
                            "",
                            xy=(step, 1.05),
                            xytext=(step, 1.2),
                            textcoords="data",
                            arrowprops=dict(
                                arrowstyle="wedge",
                                color="green",
                                lw=1.5,
                                shrinkA=0, shrinkB=10,
                            )
                        )

                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)

                axes[sensor_idx * 2].set_title(f"{self.feature_map[sensor_idx]}")
                axes[sensor_idx * 2].set_xlabel("Time Steps")
                axes[sensor_idx * 2].set_ylabel("Values")
                axes[sensor_idx * 2].legend()

                # Anomaly Score Plot
                axes[sensor_idx * 2 + 1].plot(time_steps, wafer_scores[sensor_idx, :], label="Anomaly Score", color='orange', linewidth=2)
                axes[sensor_idx * 2 + 1].axhline(self.threshold_based_val, color='cyan', linestyle='--', label="Threshold", linewidth=2)  # Threshold 수평선 추가
                axes[sensor_idx * 2 + 1].set_ylim(-1, 30)
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)

                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set
                # 이상 지점 색상 처리
                for step_idx, step in enumerate(correct_anomaly_steps):
                    if "Predicted Anomaly" not in legend_labels:
                        axes[sensor_idx * 2 + 1].axvspan(step - 0.5, step + 0.5, color='red', alpha=0.5, label="Predicted Anomaly")
                        legend_labels.add("Predicted Anomaly")
                    else:
                        axes[sensor_idx * 2 + 1].axvspan(step - 0.5, step + 0.5, color='red', alpha=0.5)

                    if correct_anomaly_variables[step_idx] == sensor_idx: # 해당 센서가 anomaly 탐지에 기여
                        axes[sensor_idx * 2 + 1].annotate(
                            "",
                            xy=(step, 1.05),
                            xytext=(step, 1.2),
                            textcoords="data",
                            arrowprops=dict(
                                arrowstyle="wedge",
                                color="green",
                                lw=1.5,
                                shrinkA=0, shrinkB=10,
                            )
                        )
                axes[sensor_idx * 2 + 1].set_title(f"{self.feature_map[sensor_idx]} - Anomaly Score")
                axes[sensor_idx * 2 + 1].set_xlabel("Time Steps")
                axes[sensor_idx * 2 + 1].set_ylabel("Score")
                axes[sensor_idx * 2 + 1].legend()

            # plt.suptitle(f"Time Series and Anomaly Scores for False Positive Wafer Lotid {lotid}, WaferNum {wafer_number}")
            plt.tight_layout()

            # 폴더 생성 및 저장
            directory = os.path.join(f'./graph_figure_topk{args.topk}', f"False_Positive_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}")
            os.makedirs(directory, exist_ok=True)
            plt.savefig(os.path.join(directory, "time_series_and_scores_false_positive.png"))
            plt.show()
            plt.close()

            # Step 4: 랜덤으로 선택한 시점의 그래프 생성
            random_windows = np.random.choice(range(len(wafer_data)), size=min(n_random_windows, len(wafer_data)), replace=False)

            for idx in random_windows:
                feature, _, _, edge_index, _, _, _ = wafer_data[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))
                    self.visualize_learned_graph(
                        self.env_config['load_model_path'],
                        f"False_Positive_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}_Window_{idx}"
                    )

        

    def visualize_case3(self, test_result, val_result, n_random_wafers=10, n_random_windows=5):
        """
        이상이 존재하는 웨이퍼 중에서 모든 윈도우를 정상으로 잘못 예측한 웨이퍼를 시각화
        Args:
            test_result: 테스트 결과 (모델 출력, 실제 레이블)
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수
            n_random_windows: 랜덤으로 선택할 시점의 수
        """
        total_err_scores, normal_scores = get_full_err_scores(test_result, val_result)
        # Step 2: 이상 탐지된 Wafer 확인
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(total_err_scores, axis=0)  # Top-K Feature 기반 에러 점수 합산
        predicted_labels = (predicted_scores > self.threshold_based_val).astype(int)  # Threshold로 이상 여부 판단

        # 이상 레이블이 포함된 웨이퍼 선택
        anomalous_wafers = set(
            (self.test_dataset.lotids[idx], self.test_dataset.wafer_numbers[idx])
            for idx, label in enumerate(true_labels)
            if label == 1
        )

        # 모델이 모든 윈도우를 정상으로 예측한 웨이퍼 선택
        fully_missed_wafers = []
        for lotid, wafer_number in anomalous_wafers:
            wafer_indices = [
                idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
                if lid == lotid and wnum == wafer_number
            ]
            wafer_predicted_labels = predicted_labels[wafer_indices]
            if np.all(wafer_predicted_labels == 0):
                fully_missed_wafers.append((lotid, wafer_number))

        # fully_missed_wafers를 NumPy 배열로 변환
        fully_missed_wafers = np.array(fully_missed_wafers, dtype=object)

        # fully_missed_wafers가 비어 있는 경우 예외 처리
        if len(fully_missed_wafers) == 0:
            print("No fully missed anomalous wafers detected.")
            return

        # 랜덤으로 웨이퍼 선택
        if len(fully_missed_wafers) > 1:
            selected_indices = np.random.choice(
                range(len(fully_missed_wafers)),
                size=min(n_random_wafers, len(fully_missed_wafers)),
                replace=False
            )
            selected_wafers = fully_missed_wafers[selected_indices]
        else:
            selected_wafers = fully_missed_wafers

        print(f"Selected fully missed anomalous wafers: {selected_wafers}")

        # 이후 저장된 selected_wafers로 작업 수행
        for lotid, wafer_number in selected_wafers:
            print(f"Visualizing for Fully Missed Anomalous Wafer Lotid {lotid}, WaferNum {wafer_number}")

            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)

            # Step 2: 실제값과 예측값 추출
            time_series_actual = []
            time_series_predicted = []
            step_num_list = []

            for feature, y, _, edge_index, _, _, step_num in wafer_data:
                feature = feature.unsqueeze(0).to(self.device)  # 입력 데이터
                edge_index = edge_index.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    predicted = self.model(feature, edge_index).cpu().detach().numpy()[0]  # 모델 예측값
                time_series_actual.append(y.numpy())  # 실제값
                time_series_predicted.append(predicted)
                step_num_list.append(step_num)

            time_series_actual = np.array(time_series_actual)
            time_series_predicted = np.array(time_series_predicted)
            step_num_list = np.array(step_num_list)

            # Anomaly scores for the wafer
            wafer_indices = [
                idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
                if lid == lotid and wnum == wafer_number
            ]
            # wafer_anomaly_scores = np.array(test_result[1])[wafer_indices]
            wafer_anomaly_scores = total_err_scores[:, wafer_indices]

            # Step 3: 시계열 플롯 생성
            time_steps = np.arange(len(time_series_actual))
            fig, axes = plt.subplots(nrows=time_series_actual.shape[1] * 2, ncols=1, figsize=(15, 3 * time_series_actual.shape[1] * 2))
            fig.subplots_adjust(hspace=0.5)

            # Step 변화 지점 계산
            step_change_indices = np.where(np.diff(step_num_list) != 0)[0] # step 변화가 발생

            for sensor_idx in range(time_series_actual.shape[1]):
                # Time Series Plot
                axes[sensor_idx * 2].plot(time_steps, time_series_actual[:, sensor_idx], label="Actual", color='blue', linewidth=2)
                axes[sensor_idx * 2].plot(time_steps, time_series_predicted[:, sensor_idx], '--', label="Predicted", color='magenta', linewidth=2)
                axes[sensor_idx * 2].set_ylim(-0.1, 1.1)
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)
                axes[sensor_idx * 2].set_title(f"{self.feature_map[sensor_idx]}")
                axes[sensor_idx * 2].set_xlabel("Time Steps")
                axes[sensor_idx * 2].set_ylabel("Values")
                axes[sensor_idx * 2].legend()

                # Anomaly Score Plot
                axes[sensor_idx * 2 + 1].plot(time_steps, wafer_anomaly_scores[sensor_idx, :], label="Anomaly Score", color='orange', linewidth=2)
                axes[sensor_idx * 2 + 1].axhline(self.threshold_based_val, color='cyan', linestyle='--', label="Threshold", linewidth=2)  # Threshold 수평선 추가
                axes[sensor_idx * 2 + 1].set_ylim(-1, 30)
                legend_labels = set()  # 이미 추가된 legend label을 추적하기 위한 set   
                for idx in step_change_indices:
                    if "Step Change" not in legend_labels:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")
                        legend_labels.add("Step Change")
                    else:
                        axes[sensor_idx * 2+1].axvline(x=idx, color='black', linestyle='--', linewidth=1.5)
                axes[sensor_idx * 2 + 1].set_title(f"{self.feature_map[sensor_idx]} - Anomaly Score")
                axes[sensor_idx * 2 + 1].set_xlabel("Time Steps")
                axes[sensor_idx * 2 + 1].set_ylabel("Score")
                axes[sensor_idx * 2 + 1].legend()

            # plt.suptitle(f"Time Series and Anomaly Scores for Fully Missed Anomalous Wafer Lotid {lotid}, WaferNum {wafer_number}")
            plt.tight_layout()

            # 폴더 생성 및 저장
            directory = os.path.join(f'./graph_figure_topk{args.topk}', f"Fully_Missed_Anomalous_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, "time_series_and_scores_fully_missed_anomalous.png"))
            plt.show()
            plt.close()

            # Step 4: 랜덤으로 선택한 시점의 그래프 생성
            random_windows = np.random.choice(wafer_indices, size=min(n_random_windows, len(wafer_indices)), replace=False)

            for idx in random_windows:
                feature, _, _, edge_index, _, _, _ = self.test_dataset[idx]
                feature = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model(feature, edge_index.to(self.device))
                    self.visualize_learned_graph(
                        self.env_config['load_model_path'],
                        f"Fully_Missed_Anomalous_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}_Window_{idx}"
                    )



    def visualize_step5_relationship_and_timeseries(self, test_result, n_random_wafers=5):
        """
        Step_num == 5인 데이터를 대상으로 Wall_Temp와 APC_Position 간의 관계를 시각화하고,
        두 변수의 시계열 데이터를 한 subplot에 그립니다.
        Args:
            test_result: 테스트 결과 (모델 출력, 실제 레이블)
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        # Step 1: 정상 레이블이면서 모델도 정상으로 예측한 웨이퍼 찾기
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(test_result[1], axis=1)  # 모델 예측 결과
        predicted_labels = (predicted_scores > self.threshold_based_val).astype(int)  # Threshold로 이상 여부 판단

        fully_normal_wafers = set(
            (self.test_dataset.lotids[idx], self.test_dataset.wafer_numbers[idx])
            for idx, label in enumerate(true_labels)
            if label == 0
        )

        correctly_predicted_wafers = []
        for lotid, wafer_number in fully_normal_wafers:
            wafer_indices = [
                idx for idx, (lid, wnum) in enumerate(zip(self.test_dataset.lotids, self.test_dataset.wafer_numbers))
                if lid == lotid and wnum == wafer_number
            ]
            wafer_predicted_labels = predicted_labels[wafer_indices]
            wafer_true_labels = true_labels[wafer_indices]
            if np.all(wafer_predicted_labels == 0) and np.all(wafer_true_labels == 0):
                correctly_predicted_wafers.append((lotid, wafer_number))

        if len(correctly_predicted_wafers) == 0:
            print("No fully normal wafers with correct predictions detected.")
            return

        correctly_predicted_wafers = np.array(correctly_predicted_wafers)
        selected_indices = np.random.choice(
            range(len(correctly_predicted_wafers)),
            size=min(n_random_wafers, len(correctly_predicted_wafers)),
            replace=False
        )
        selected_wafers = correctly_predicted_wafers[selected_indices]

        wall_temp_idx = self.feature_map.index("Wall_Temp_Monitor")
        apc_position_idx = self.feature_map.index("APC_Position")

        for lotid, wafer_number in selected_wafers:
            print(f"Visualizing Wafer Lotid {lotid}, WaferNum {wafer_number} for step_num == 5")

            wafer_data = self.get_wafer_data(lotid, wafer_number)

            time_series_actual = []
            time_series_predicted = []
            step_num_list = []

            for feature, y, _, edge_index, _, _, step_num in wafer_data:
                feature = feature.unsqueeze(0).to(self.device)
                edge_index = edge_index.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    predicted = self.model(feature, edge_index).cpu().detach().numpy()[0]
                time_series_actual.append(y.numpy())
                time_series_predicted.append(predicted)
                step_num_list.append(step_num)

            time_series_actual = np.array(time_series_actual)
            time_series_predicted = np.array(time_series_predicted)
            step_num_list = np.array(step_num_list)

            step_5_indices = np.where(step_num_list == 5)[0]

            if len(step_5_indices) == 0:
                print(f"No step_num == 5 data for Wafer Lotid {lotid}, WaferNum {wafer_number}")
                continue

            # Step 3: Graph 저장 및 시각화
            directory = os.path.join(f'./graph_figure_topk{args.topk}', f"Normal_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}")
            os.makedirs(directory, exist_ok=True)

            for step_idx in step_5_indices:
                att_edge_index = self.model.att_edge_index
                attention_weights = self.model.extract_attention_weights()[0].squeeze().cpu().numpy()

                fig, ax = plt.subplots(figsize=(10, 8))
                G = nx.DiGraph()

                G.add_node("Wall_Temp_Monitor", color='blue')
                G.add_node("APC_Position", color='green')

                for edge_idx in range(att_edge_index.shape[1]):
                    src, dst = att_edge_index[:, edge_idx]
                    weight = attention_weights[edge_idx]
                    if abs(weight) >= 0.05 and (src in [wall_temp_idx, apc_position_idx] or dst in [wall_temp_idx, apc_position_idx]):
                        G.add_edge(self.feature_map[src], self.feature_map[dst], weight=weight)

                pos = nx.spring_layout(G, k=1.2)
                edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

                nx.draw(
                    G, pos, with_labels=True, ax=ax, node_color='lightblue', font_size=10,
                    edge_color=weights, edge_cmap=plt.cm.Reds,
                    edge_vmin=min(weights), edge_vmax=max(weights), width=2, arrowstyle='->'
                )
                nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in zip(edges, weights)}, font_size=10)
                plt.title(f"Graph for Wafer Lotid {lotid}, WaferNum {wafer_number} (Window {step_idx})")
                graph_path = os.path.join(directory, f"relationship_graph_window_{step_idx}.png")
                plt.savefig(graph_path)
                plt.close()

            # Step 5: Time series plot 저장
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
            axes[0].plot(step_5_indices, time_series_actual[step_5_indices, wall_temp_idx], label="Actual", color='blue')
            axes[0].plot(step_5_indices, time_series_predicted[step_5_indices, wall_temp_idx], label="Predicted", color='green')
            axes[0].set_title("Wall_Temp_Monitor")
            axes[0].set_ylim(-0.1, 1.1)
            axes[0].legend()

            axes[1].plot(step_5_indices, time_series_actual[step_5_indices, apc_position_idx], label="Actual", color='blue')
            axes[1].plot(step_5_indices, time_series_predicted[step_5_indices, apc_position_idx], label="Predicted", color='green')
            axes[1].set_title("APC_Position")
            axes[1].set_ylim(-0.1, 1.1)
            axes[1].legend()

            # plt.suptitle(f"Time Series for Wafer Lotid {lotid}, WaferNum {wafer_number} (step_num == 5)")
            time_series_path = os.path.join(directory, "time_series_subplot_step_5.png")
            plt.savefig(time_series_path)
            plt.close()

            print(f"Saved plots for Wafer Lotid {lotid}, WaferNum {wafer_number}")
    def visualize_step3_4_relationship_and_timeseries(self, lotid, wafernum):
        """
        특정 lotid와 wafernum에 대해 step_num == 3 또는 4인 데이터를 대상으로 
        Wall_Temp와 APC_Position 간의 관계를 시각화하고,
        두 변수의 시계열 데이터를 한 subplot에 그립니다.
        Args:
            lotid: 분석할 웨이퍼의 Lot ID
            wafernum: 분석할 웨이퍼의 번호
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        print(f"Visualizing Wafer Lotid {lotid}, WaferNum {wafernum} for step_num == 3 or 4")

        wafer_data = self.get_wafer_data(lotid, wafernum)

        time_series_actual = []
        time_series_predicted = []
        step_num_list = []

        for feature, y, _, edge_index, _, _, step_num in wafer_data:
            feature = feature.unsqueeze(0).to(self.device)
            edge_index = edge_index.to(self.device)
            self.model.eval()
            with torch.no_grad():
                predicted = self.model(feature, edge_index).cpu().detach().numpy()[0]
            time_series_actual.append(y.numpy())
            time_series_predicted.append(predicted)
            step_num_list.append(step_num)

        time_series_actual = np.array(time_series_actual)
        time_series_predicted = np.array(time_series_predicted)
        step_num_list = np.array(step_num_list)

        step_3_4_indices = np.where((step_num_list == 3) | (step_num_list == 4))[0]

        if len(step_3_4_indices) == 0:
            print(f"No step_num == 3 or 4 data for Wafer Lotid {lotid}, WaferNum {wafernum}")
            return

        # Step 3: Graph 저장 및 시각화
        directory = os.path.join(f'./graph_figure_topk{args.topk}', f"Step3,4_Lotid_{lotid}_WaferNum_{wafernum}")
        os.makedirs(directory, exist_ok=True)

        wall_temp_idx = self.feature_map.index("Wall_Temp_Monitor")
        apc_position_idx = self.feature_map.index("APC_Position")

        for step_idx in step_3_4_indices:
            att_edge_index = self.model.att_edge_index
            attention_weights = self.model.extract_attention_weights()[0].squeeze().cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            G = nx.DiGraph()

            G.add_node("Wall_Temp_Monitor", color='blue')
            G.add_node("APC_Position", color='green')

            for edge_idx in range(att_edge_index.shape[1]):
                src, dst = att_edge_index[:, edge_idx]
                weight = attention_weights[edge_idx]
                if abs(weight) >= 0.05 and (src in [wall_temp_idx, apc_position_idx] or dst in [wall_temp_idx, apc_position_idx]):
                    G.add_edge(self.feature_map[src], self.feature_map[dst], weight=weight)

            pos = nx.spring_layout(G, k=1.2)
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

            nx.draw(
                G, pos, with_labels=True, ax=ax, node_color='lightblue', font_size=10,
                edge_color=weights, edge_cmap=plt.cm.Reds,
                edge_vmin=min(weights), edge_vmax=max(weights), width=2, arrowstyle='->'
            )
            nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in zip(edges, weights)}, font_size=10)
            plt.title(f"Graph for Wafer Lotid {lotid}, WaferNum {wafernum} (Window {step_idx})")
            graph_path = os.path.join(directory, f"relationship_graph_window_{step_idx}.png")
            plt.savefig(graph_path)
            plt.close()

        # Step 5: Time series plot 저장
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        axes[0].plot(step_3_4_indices, time_series_actual[step_3_4_indices, wall_temp_idx], label="Actual", color='blue')
        axes[0].plot(step_3_4_indices, time_series_predicted[step_3_4_indices, wall_temp_idx], label="Predicted", color='green')
        axes[0].set_title("Wall_Temp_Monitor")
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].legend()

        axes[1].plot(step_3_4_indices, time_series_actual[step_3_4_indices, apc_position_idx], label="Actual", color='blue')
        axes[1].plot(step_3_4_indices, time_series_predicted[step_3_4_indices, apc_position_idx], label="Predicted", color='green')
        axes[1].set_title("APC_Position")
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend()

        # plt.suptitle(f"Time Series for Wafer Lotid {lotid}, WaferNum {wafernum} (step_num == 3 or 4)")
        time_series_path = os.path.join(directory, "time_series_subplot_step_3_4.png")
        plt.savefig(time_series_path)
        plt.close()

        print(f"Saved plots for Wafer Lotid {lotid}, WaferNum {wafernum}")


    def visualize_correctly_predicted_normal_wafers(self, test_result, n_random_wafers=10):
        """
        정상 레이블을 가지고 있고, 모델도 정상으로 올바르게 예측한 랜덤 웨이퍼에 대해 시계열 플롯 생성
        Args:
            test_result: 테스트 결과 (모델 출력, 실제 레이블)
            n_random_wafers: 랜덤으로 선택할 웨이퍼의 수
        """

        # Step 1: 정상 레이블이면서 모델이 올바르게 예측한 웨이퍼 찾기
        true_labels = np.array(test_result[2])[:, 0]  # 윈도우 단위 레이블
        predicted_scores = np.max(test_result[1], axis=1)  # 모델 예측 결과
        predicted_labels = (predicted_scores > self.threshold_based_val).astype(int)  # Threshold로 이상 여부 판단

        # 정상 레이블이고 모델도 정상으로 예측한 경우
        correct_normal_indices = np.where((true_labels == 0) & (predicted_labels == 0))[0]

        # 웨이퍼별로 그룹화
        wafer_groups = {}
        for idx in correct_normal_indices:
            lotid = self.test_dataset.lotids[idx]
            wafer_number = self.test_dataset.wafer_numbers[idx]
            if (lotid, wafer_number) not in wafer_groups:
                wafer_groups[(lotid, wafer_number)] = []
            wafer_groups[(lotid, wafer_number)].append(idx)

        # 정상 웨이퍼만 추출
        correct_normal_wafers = list(wafer_groups.keys())

        if len(correct_normal_wafers) == 0:
            print("No correctly predicted normal wafers detected.")
            return

        # 랜덤으로 웨이퍼 선택
        selected_wafers = np.random.choice(correct_normal_wafers, size=min(n_random_wafers, len(correct_normal_wafers)), replace=False)
        print(f"Selected correctly predicted normal wafers: {selected_wafers}")

        for lotid, wafer_number in selected_wafers:
            print(f"Visualizing for Normal Wafer Lotid {lotid}, WaferNum {wafer_number}")

            # Wafer의 모든 윈도우 데이터 구성
            wafer_data = self.get_wafer_data(lotid, wafer_number)  # get_wafer_data 활용

            # Step 2: 실제값과 예측값 추출
            time_series_actual = []
            time_series_predicted = []
            step_num_list = []

            for feature, y, _, edge_index, _, _, step_num in wafer_data:
                feature = feature.unsqueeze(0).to(self.device)  # 입력 데이터
                edge_index = edge_index.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    predicted = self.model(feature, edge_index).cpu().detach().numpy()[0]  # 모델 예측값
                time_series_actual.append(y.numpy())  # 실제값
                time_series_predicted.append(predicted)
                step_num_list.append(step_num)

            time_series_actual = np.array(time_series_actual)
            time_series_predicted = np.array(time_series_predicted)
            step_num_list = np.array(step_num_list)

            # Step 3: 시계열 플롯 생성
            time_steps = np.arange(len(time_series_actual))
            fig, axes = plt.subplots(nrows=time_series_actual.shape[1], ncols=1, figsize=(15, 3 * time_series_actual.shape[1]))
            fig.subplots_adjust(hspace=0.5)

            # Step 변화 지점 계산
            step_change_indices = np.where(np.diff(step_num_list) != 0)[0]# step 변화가 발생

            for sensor_idx, ax in enumerate(axes):
                ax.plot(time_steps, time_series_actual[:, sensor_idx], label="Actual", color='blue', linewidth=2)
                ax.plot(time_steps, time_series_predicted[:, sensor_idx], '--', label="Predicted", color='magenta', linewidth=2)
                # y축 범위를 0~1로 설정
                ax.set_ylim(0-0.1, 1+0.1)

                # Step 변화 시점에서 수직선 그리기
                for idx in step_change_indices:
                    ax.axvline(x=idx, color='black', linestyle='--', linewidth=1.5, label="Step Change")

                ax.set_title(f"{self.feature_map[sensor_idx]}")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Values")
                ax.legend()


            # plt.suptitle(f"Time Series for Correctly Predicted Normal Wafer Lotid {lotid}, WaferNum {wafer_number}")
            plt.tight_layout()

            # 폴더 생성 및 저장
            directory = os.path.join(f'./graph_figure_topk{args.topk}', f"Correct_Normal_Wafer_Lotid_{lotid}_WaferNum_{wafer_number}")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, "time_series_correct_normal.png"))
            plt.show()
            plt.close()


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
    parser.add_argument('-load_model', help='0: train / 1: only test', type=int, default=0)
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
        





