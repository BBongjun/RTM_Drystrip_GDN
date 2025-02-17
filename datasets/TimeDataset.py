import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tqdm import tqdm
import os

# index mapping 단순한 버전
class WaferDataset(Dataset):
    def __init__(self, file_list, edge_index, label=True):
        """
        GDN에 적합하도록 WaferDataset을 변환한 데이터셋 클래스 (데이터 결합 방식 적용)
        Args:
            file_list: List of .npz file paths
            edge_index: Edge index for graph structure
            label: Whether to include labels
        """
        self.file_list = file_list  # npz 파일 목록
        self.edge_index = edge_index  # 그래프 구조 (노드 간 연결 정보)
        self.label = label  # 레이블 포함 여부

        # 데이터를 한 번에 로드하여 결합
        all_data, all_next_steps, all_labels = [], [], []
        self.lotids = []
        self.wafer_numbers = []
        self.step_nums = []

        for file_path in tqdm(file_list, desc="Loading and merging data"):
            data = np.load(file_path, allow_pickle=True)
            all_data.append(data['data'].astype(float))  # (n_windows, window_size, n_features)
            all_next_steps.append(data['next_step'].astype(float))  # (n_windows, n_features)
            all_labels.append(data['labels'])  # (n_windows,)

            # Lotid, WaferNum 정보도 저장 => 나중에 vis에서 활용
            self.lotids.extend(data['lotids'])
            self.wafer_numbers.extend(data['wafer_numbers'])
            self.step_nums.extend(data['step_num'])

        # 데이터 결합
        self.all_data = np.concatenate(all_data, axis=0)  # 전체 윈도우 결합
        self.all_next_steps = np.concatenate(all_next_steps, axis=0)
        self.all_labels = np.concatenate(all_labels, axis=0)
        self.n_sensor = self.all_data.shape[2]  # 센서(노드) 개수


    def __len__(self):
        """
        데이터셋의 총 샘플 개수를 반환
        """
        return self.all_data.shape[0]  # 전체 윈도우 개수 반환

    def __getitem__(self, idx):
        """
        GDN 모델에서 사용할 데이터 반환
        Args:
            idx: 데이터 인덱스
        Returns:
            feature: 입력 데이터 (node_num, window_size)
            target: 출력 데이터 (node_num,)
            label: 라벨 정보
            edge_index: 그래프 구조
        """
        # 데이터 로드
        window_data = self.all_data[idx]  # (window_size, n_features)
        y = self.all_next_steps[idx]  # (n_features,)
        label = self.all_labels[idx]  # Scalar
        lotid = self.lotids[idx]  # Lotid
        wafer_number = self.wafer_numbers[idx]  # WaferNum
        step_num = self.step_nums[idx]
        step_num = int(step_num)

        # 데이터 차원 조정 (window_size, n_features → n_features, window_size)
        feature = torch.FloatTensor(window_data.T)  # (n_features, window_size)
        y = torch.FloatTensor(y)  # 마지막 시점 값 (n_features,)

        return feature, y, label, self.edge_index.long(), lotid, wafer_number, step_num

def loader_wafer(root, edge_index, batch_size, label=True):
    """
    GDN 모델에 적합한 DataLoader 생성 함수
    Args:
        root: 데이터 디렉토리 경로
        edge_index: Edge index for graph structure
        batch_size: DataLoader 배치 크기
        label: 레이블 사용 여부
    Returns:
        train_loader, val_loader, test_loader, n_sensor
    """
    # 데이터 경로 설정
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    # 파일 리스트 생성
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".npz")]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".npz")]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".npz")]

    # WaferDataset 생성
    train_dataset = WaferDataset(train_files, edge_index, label=label)
    val_dataset = WaferDataset(val_files, edge_index, label=label)
    test_dataset = WaferDataset(test_files, edge_index, label=label)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    n_sensor = train_dataset.n_sensor  # 센서 개수 추출

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, n_sensor