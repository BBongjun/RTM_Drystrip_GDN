import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from torch.optim.lr_scheduler import CosineAnnealingLR



def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



# def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

#     seed = config['seed']

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

#     now = time.time()
    
#     train_loss_list = []
#     cmp_loss_list = []

#     device = get_device()


#     acu_loss = 0
#     min_loss = 1e+8
#     min_f1 = 0
#     min_pre = 0
#     best_prec = 0

#     i = 0
#     epoch = config['epoch']
#     early_stop_win = 100

#     model.train()

#     log_interval = 1000
#     stop_improve_count = 0

#     dataloader = train_dataloader

#     for i_epoch in range(epoch):

#         acu_loss = 0
#         model.train()

#         for x, labels, attack_labels, edge_index in dataloader:
#             _start = time.time()

#             x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

#             optimizer.zero_grad()
#             out = model(x, edge_index).float().to(device)
#             loss = loss_func(out, labels)
            
#             loss.backward()
#             optimizer.step()

            
#             train_loss_list.append(loss.item())
#             acu_loss += loss.item()
                
#             i += 1


#         # each epoch
#         print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
#                         i_epoch, epoch, 
#                         acu_loss/len(dataloader), acu_loss), flush=True
#             )

#         # use val dataset to judge
#         if val_dataloader is not None:

#             val_loss, val_result = test(model, val_dataloader)

#             if val_loss < min_loss:
#                 torch.save(model.state_dict(), save_path)

#                 min_loss = val_loss
#                 stop_improve_count = 0
#             else:
#                 stop_improve_count += 1


#             if stop_improve_count >= early_stop_win:
#                 break

#         else:
#             if acu_loss < min_loss :
#                 torch.save(model.state_dict(), save_path)
#                 min_loss = acu_loss



#     return train_loss_list


from tqdm import tqdm  # 학습 진행률 표시를 위한 tqdm
import matplotlib.pyplot as plt

def train(
    model=None, save_path='', config={}, train_dataloader=None, 
    val_dataloader=None, feature_map={}, test_dataloader=None, 
    test_dataset=None, dataset_name='swat', train_dataset=None
):
    seed = config['seed']
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    device = get_device()

    # 학습 손실 기록
    train_loss_list = []
    val_loss_list = []

    acu_loss = 0
    min_loss = float('inf')  # 최소 손실 초기화
    stop_improve_count = 0
    early_stop_win = config.get('early_stop_win', 200)  # 조기 종료 조건
    epoch = config['epoch']

    # 학습 시작
    for i_epoch in range(epoch):
        model.train()
        acu_loss = 0

        # tqdm으로 진행률 표시
        with tqdm(total=len(train_dataloader), desc=f"Epoch {i_epoch+1}/{epoch}") as pbar:
            for batch_idx, (x, labels, attack_labels, edge_index, lotid, wafer_number, step_num) in enumerate(train_dataloader):
                # 데이터 로드 및 디바이스 이동
                x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

                # Forward, Loss, Backward, Step
                optimizer.zero_grad()
                out = model(x, edge_index).float()
                loss = loss_func(out, labels)
                loss.backward()
                optimizer.step()

                # 손실 업데이트
                
                acu_loss += loss.item()

                # 진행률 업데이트
                pbar.set_postfix({'Batch Loss': f"{loss.item():.6f}"})
                pbar.update(1)

        # 에포크 손실 평균
        avg_train_loss = acu_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)
        # 에포크 로그 출력
        print(f"Epoch {i_epoch+1}/{epoch} | Train Loss: {avg_train_loss:.6f}")

        # 검증 데이터셋에서 평가
        if val_dataloader is not None:
            val_loss, _ = test(model, val_dataloader)
            val_loss_list.append(val_loss)

            print(f"Validation Loss: {val_loss:.6f}")

            # 최저 검증 손실 저장 및 조기 종료
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                print(f"Best Model Saved at Epoch {i_epoch+1}")
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                print(f"Early stopping at epoch {i_epoch+1}")
                break

        else:
            # 검증 데이터가 없으면 학습 손실 기준으로 저장
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                print(f"Best Model Saved at Epoch {i_epoch+1}")
                min_loss = acu_loss

    # 학습 종료 후 손실 시각화
    visualize_loss(train_loss_list, val_loss_list)

    return train_loss_list, val_loss_list


def train_sch(
    model=None, save_path='', config={}, train_dataloader=None, 
    val_dataloader=None, feature_map={}, test_dataloader=None, 
    test_dataset=None, dataset_name='swat', train_dataset=None
):
    seed = config['seed']
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    # Cosine Annealing Scheduler 추가
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    device = get_device()

    # 학습 손실 기록
    train_loss_list = []
    val_loss_list = []

    acu_loss = 0
    min_loss = float('inf')  # 최소 손실 초기화
    stop_improve_count = 0
    early_stop_win = config.get('early_stop_win', 200)  # 조기 종료 조건
    epoch = config['epoch']

    # 학습 시작
    for i_epoch in range(epoch):
        model.train()
        acu_loss = 0

        # tqdm으로 진행률 표시
        with tqdm(total=len(train_dataloader), desc=f"Epoch {i_epoch+1}/{epoch}") as pbar:
            for batch_idx, (x, labels, attack_labels, edge_index, lotid, wafer_number, step_num) in enumerate(train_dataloader):
                # 데이터 로드 및 디바이스 이동
                x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

                # Forward, Loss, Backward, Step
                optimizer.zero_grad()
                out = model(x, edge_index).float()
                loss = loss_func(out, labels)
                loss.backward()
                optimizer.step()

                # 손실 업데이트
                acu_loss += loss.item()

                # 진행률 업데이트
                pbar.set_postfix({'Batch Loss': f"{loss.item():.6f}"})
                pbar.update(1)

        # 에포크 손실 평균
        avg_train_loss = acu_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)

        # Scheduler Step
        scheduler.step()

        # 에포크 로그 출력
        print(f"Epoch {i_epoch+1}/{epoch} | Train Loss: {avg_train_loss:.6f} | Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # 검증 데이터셋에서 평가
        if val_dataloader is not None:
            val_loss, _ = test(model, val_dataloader)
            val_loss_list.append(val_loss)

            print(f"Validation Loss: {val_loss:.6f}")

            # 최저 검증 손실 저장 및 조기 종료
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                print(f"Best Model Saved at Epoch {i_epoch+1}")
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                print(f"Early stopping at epoch {i_epoch+1}")
                break

        else:
            # 검증 데이터가 없으면 학습 손실 기준으로 저장
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                print(f"Best Model Saved at Epoch {i_epoch+1}")
                min_loss = acu_loss

    # 학습 종료 후 손실 시각화
    visualize_loss(train_loss_list, val_loss_list)

    return train_loss_list, val_loss_list

def visualize_loss(train_loss_list, val_loss_list=None, save_path="loss_plot.png"):
    """
    학습 손실과 검증 손실 시각화 및 저장
    Args:
        train_loss_list (list): 학습 손실 리스트
        val_loss_list (list, optional): 검증 손실 리스트 (default: None)
        save_path (str): 저장할 파일 경로 및 이름 (default: "loss_plot.png")
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label="Train Loss", color="blue", alpha=0.7)
    if val_loss_list:
        plt.plot(range(len(val_loss_list)), val_loss_list, label="Validation Loss", color="orange", alpha=0.7)
    plt.title("Train and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    plt.savefig(save_path, format="png")
    print(f"Loss plot saved to {save_path}")

    # 그래프 표시 (원하는 경우 주석 처리 가능)
    plt.show()
