"""
HyDL-IDS: Hybrid Deep Learning-based Intrusion Detection System for Car-Hacking Detection

이 코드는 차량 내부 네트워크(CAN 버스) 침입 탐지를 위한 HyDL-IDS 모델을 구현합니다.
모델은 CNN과 LSTM을 결합하여 CAN 트래픽의 공간적 특징과 시간적 특징을 모두 추출합니다.

Car-Hacking 데이터셋(HCRL, 고려대학교)을 사용하며 다음과 같은 공격 유형을 탐지합니다:
- DoS (Denial of Service)
- Fuzzy
- Spoofing Gear
- Spoofing RPM

작성자: Scout AI
날짜: 2025-05-06
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
import logging
from typing import Tuple, Dict, List, Any, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyDL_IDS:
    """
    HyDL-IDS (Hybrid Deep Learning-based Intrusion Detection System) 클래스
    
    데이터 전처리, 모델 구축, 학습 및 평가 기능을 제공합니다.
    """
    
    def __init__(self):
        """
        HyDL-IDS 모델 초기화
        """
        self.model = None
        self.can_id_encoder = None
        self.dlc_scaler = None
        self.data_scaler = None
        self.history = None
        
    # def preprocess_can_id(self, can_ids: np.ndarray) -> np.ndarray:
    #     """
    #     CAN ID 전처리: 16진수 CAN ID를 정수로 인코딩
        
    #     Args:
    #         can_ids: 16진수 문자열 형태의 CAN ID 배열
            
    #     Returns:
    #         정수 인코딩된 CAN ID 배열
    #     """
    #     logger.info("CAN ID 전처리 중...")
        
    #     # LabelEncoder를 사용하여 고유한 CAN ID를 0부터 N-1까지의 정수로 매핑
    #     if self.can_id_encoder is None:
    #         self.can_id_encoder = LabelEncoder()
    #         encoded_can_ids = self.can_id_encoder.fit_transform(can_ids)
    #     else:
    #         encoded_can_ids = self.can_id_encoder.transform(can_ids)
            
    #     logger.info(f"고유 CAN ID 수: {len(self.can_id_encoder.classes_)}")
    #     return encoded_can_ids
    
    # def preprocess_dlc(self, dlc_values: np.ndarray) -> np.ndarray:
    #     """
    #     DLC(Data Length Code) 전처리: log(값 + 1) 변환 후 0-1 범위로 스케일링
        
    #     Args:
    #         dlc_values: 정수 형태의 DLC 값 배열 (0-8)
            
    #     Returns:
    #         정규화된 DLC 값 배열
    #     """
    #     logger.info("DLC 전처리 중...")
        
    #     # log(값 + 1) 변환
    #     log_dlc = np.log1p(dlc_values)
        
    #     # Min-Max 스케일링 (0-1 범위)
    #     if self.dlc_scaler is None:
    #         self.dlc_scaler = MinMaxScaler()
    #         normalized_dlc = self.dlc_scaler.fit_transform(log_dlc.reshape(-1, 1)).flatten()
    #     else:
    #         normalized_dlc = self.dlc_scaler.transform(log_dlc.reshape(-1, 1)).flatten()
            
    #     return normalized_dlc
    
    # def preprocess_data_bytes(self, data_bytes: np.ndarray, dlc_values: np.ndarray) -> np.ndarray:
    #     """
    #     DATA 바이트 전처리:
    #     1. 16진수 데이터 바이트를 10진수로 변환 (0-255)
    #     2. 존재하지 않는 바이트('M')는 256으로 변환
    #     3. log(값 + 1) 변환 후 0-1 범위로 스케일링
        
    #     Args:
    #         data_bytes: DATA[0]-DATA[7] 필드 값 배열 (8 바이트)
    #         dlc_values: 각 행의 DLC 값 배열
            
    #     Returns:
    #         정규화된 DATA 바이트 배열 (차원: [샘플 수, 8])
    #     """
    #     logger.info("DATA 바이트 전처리 중...")
        
    #     # 결과 배열 초기화 (샘플 수 x 8 바이트)
    #     n_samples = len(dlc_values)
    #     decimal_data = np.zeros((n_samples, 8))
        
    #     # 각 샘플에 대해 처리
    #     for i in range(n_samples):
    #         dlc = dlc_values[i]
            
    #         # DLC 값에 따라 실제 데이터 바이트 수 결정
    #         for j in range(8):
    #             if j < dlc:
    #                 # 16진수 데이터 바이트를 10진수로 변환
    #                 try:
    #                     if isinstance(data_bytes[i, j], str) and data_bytes[i, j].upper() != 'M':
    #                         decimal_data[i, j] = int(data_bytes[i, j], 16)
    #                     else:
    #                         # 'M' 또는 다른 비정상 값인 경우 256 할당
    #                         decimal_data[i, j] = 256
    #                 except ValueError:
    #                     # 변환 오류 시 256 할당
    #                     decimal_data[i, j] = 256
    #             else:
    #                 # DLC보다 인덱스가 큰 경우 (존재하지 않는 바이트) 256 할당
    #                 decimal_data[i, j] = 256
        
    #     # log(값 + 1) 변환
    #     log_data = np.log1p(decimal_data)
        
    #     # Min-Max 스케일링 (0-1 범위)
    #     if self.data_scaler is None:
    #         self.data_scaler = MinMaxScaler()
    #         normalized_data = self.data_scaler.fit_transform(log_data)
    #     else:
    #         normalized_data = self.data_scaler.transform(log_data)
            
    #     return normalized_data
    
    # def preprocess_flag(self, flags: np.ndarray) -> np.ndarray:
    #     """
    #     Flag/Tag 전처리: 'R'(정상)을 0으로, 'T'(공격)를 1로 인코딩
        
    #     Args:
    #         flags: 'R' 또는 'T' 문자열로 이루어진 Flag/Tag 값 배열
            
    #     Returns:
    #         인코딩된 라벨 배열 (0: 정상, 1: 공격)
    #     """
    #     logger.info("Flag/Tag 전처리 중...")
        
    #     # 'R'을 0으로, 'T'를 1로 변환
    #     encoded_flags = np.array([0 if flag == 'R' else 1 for flag in flags])
        
    #     return encoded_flags
    
    # def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     시계열 데이터를 모델 입력에 적합한 시퀀스 형태로 변환
        
    #     Args:
    #         features: 특성 배열 [샘플 수, 특성 수]
    #         labels: 라벨 배열 [샘플 수]
            
    #     Returns:
    #         시퀀스 형태의 특성 배열 [시퀀스 수, 윈도우 크기, 특성 수]
    #         시퀀스 형태의 라벨 배열 [시퀀스 수] - 각 시퀀스의 마지막 샘플 라벨 사용
    #     """
    #     logger.info(f"시퀀스 데이터 생성 중 (윈도우 크기: {self.window_size}, 스트라이드: {self.stride})...")
        
    #     n_samples, n_features = features.shape
        
    #     # 시퀀스 수 계산
    #     n_sequences = (n_samples - self.window_size) // self.stride + 1
        
    #     # 시퀀스 배열 초기화
    #     X = np.zeros((n_sequences, self.window_size, n_features), dtype=np.float32) # <--- dtype=np.float32 추가
    #     y = np.zeros(n_sequences)
        
    #     # 윈도우 슬라이딩 방식으로 시퀀스 생성
    #     for i in range(n_sequences):
    #         start_idx = i * self.stride
    #         end_idx = start_idx + self.window_size
            
    #         X[i] = features[start_idx:end_idx]
    #         # 각 시퀀스의 라벨은 마지막 샘플의 라벨로 설정
    #         # 만약 시퀀스 내에 공격 샘플이 하나라도 있다면 공격으로 간주하는 방식도 가능
    #         y[i] = labels[end_idx - 1]
        
    #     logger.info(f"생성된 시퀀스 수: {n_sequences}")
    #     return X, y
    
    # def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     전체 데이터 전처리 파이프라인
        
    #     Args:
    #         data: 원본 데이터프레임 (Timestamp, CAN ID, DLC, DATA[0]-DATA[7], Flag/Tag 포함)
            
    #     Returns:
    #         시퀀스 형태의 특성 배열 [시퀀스 수, 윈도우 크기, 특성 수]
    #         시퀀스 형태의 라벨 배열 [시퀀스 수]
    #     """
    #     logger.info("데이터 전처리 시작...")
        
    #     # 결측값 처리
    #     # (옵션) 범주형 필드는 최빈값으로, 수치형 필드는 평균/중앙값으로 대체할 수 있음
    #     # 여기서는 간단한 예시만 제공하며, 실제 데이터에 맞게 조정 필요
    #     if data.isnull().any().any():
    #         logger.warning("결측값이 발견되었습니다.")
            
    #         # CAN ID (범주형) - 최빈값으로 대체
    #         if data['CAN ID'].isnull().any():
    #             can_id_mode = data['CAN ID'].mode()[0]
    #             data['CAN ID'].fillna(can_id_mode, inplace=True)
                
    #         # DLC (수치형) - 중앙값으로 대체
    #         if data['DLC'].isnull().any():
    #             dlc_median = data['DLC'].median()
    #             data['DLC'].fillna(dlc_median, inplace=True)
                
    #         # Flag/Tag (범주형) - 결측값이 있는 경우 'R'(정상)으로 간주
    #         if data['Flag/Tag'].isnull().any():
    #             data['Flag/Tag'].fillna('R', inplace=True)
        
    #     # 전처리 단계별 수행
    #     encoded_can_ids = self.preprocess_can_id(data['CAN ID'].values)
    #     normalized_dlc = self.preprocess_dlc(data['DLC'].values)
        
    #     # DATA[0]-DATA[7] 추출 및 전처리
    #     data_columns = [f'DATA[{i}]' for i in range(8)]
    #     data_bytes = data[data_columns].values
    #     normalized_data = self.preprocess_data_bytes(data_bytes, data['DLC'].values)
        
    #     # Flag/Tag 전처리
    #     encoded_flags = self.preprocess_flag(data['Flag/Tag'].values)
        
    #     # 전처리된 특성 결합
    #     # [CAN ID(1), DLC(1), DATA 바이트(8)] = 총 10개 특성
    #     features = np.column_stack([encoded_can_ids, normalized_dlc, normalized_data])
        
    #     # 시퀀스 데이터 생성
    #     X_sequences, y_sequences = self.create_sequences(features, encoded_flags)
        
    #     logger.info("데이터 전처리 완료")
    #     return X_sequences, y_sequences
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        HyDL-IDS 모델 구축 (CNN + LSTM 아키텍처)
        
        Args:
            input_shape: 입력 데이터 형태 (시퀀스 길이, 특성 수)
        """
        logger.info("HyDL-IDS 모델 구축 중...")
        
        # 모델 아키텍처 정의
        model = Sequential()
        
        # 입력층
        model.add(Input(shape=input_shape))
        
        # Convo Unit I
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(Dropout(0.2))
        
        # Convo Unit II
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(Dropout(0.2))
        
        # LSTM 레이어
        model.add(LSTM(128, activation='tanh'))
        
        # Flatten 레이어
        model.add(Flatten())
        
        # Dense 레이어 1
        model.add(Dense(128))
        
        # Dropout 레이어
        model.add(Dropout(0.2))
        
        # 출력 레이어 (Dense 레이어 2)
        model.add(Dense(1, activation='sigmoid'))
        
        # 모델 요약 출력
        model.summary()
        
        self.model = model
        logger.info("모델 구축 완료")
        
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """
        모델 컴파일
        
        Args:
            learning_rate: Adam 옵티마이저의 학습률
        """
        if self.model is None:
            raise ValueError("모델이 아직 구축되지 않았습니다. build_model()을 먼저 호출하세요.")
        
        logger.info("모델 컴파일 중...")
        
        # 옵티마이저 설정
        optimizer = Adam(learning_rate=learning_rate)
        
        # 모델 컴파일
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae', 'mse']
        )
        
        logger.info("모델 컴파일 완료")
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray,
                  batch_size: int = 256, epochs: int = 10,
                  use_early_stopping: bool = True, verbose: int = 2) -> Dict[str, List[float]]:
        """
        모델 학습
        
        Args:
            X_train: 학습 데이터 특성
            y_train: 학습 데이터 라벨
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 라벨
            batch_size: 배치 크기
            epochs: 에포크 수
            use_early_stopping: EarlyStopping 콜백 사용 여부
            verbose: 진행 출력 레벨 (0: 출력 없음, 1: 진행 막대, 2: 에포크당 한 줄)
            
        Returns:
            학습 이력 (손실, 정확도 등)
        """
        if self.model is None:
            raise ValueError("모델이 아직 컴파일되지 않았습니다. compile_model()을 먼저 호출하세요.")
        
        logger.info(f"모델 학습 시작 (배치 크기: {batch_size}, 에포크: {epochs})")
        
        # 콜백 정의
        callbacks = []
        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                min_delta=0.01,
                patience=3,
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            logger.info("EarlyStopping 콜백 활성화")
        
        # 모델 학습
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose  # 에포크당 한 줄만 출력
        )
        
        self.history = history.history
        logger.info("모델 학습 완료")
        
        return self.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, verbose: int = 2) -> Dict[str, float]:
        """
        테스트 데이터로 모델 평가
        
        Args:
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 라벨
            verbose: 진행 출력 레벨 (0: 출력 없음, 1: 진행 막대, 2: 한 줄 출력)
            
        Returns:
            평가 지표 딕셔너리 (손실, 정확도, MAE, MSE)
        """
        if self.model is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        
        logger.info("테스트 데이터에서 모델 평가 중...")
        
        # 기본 평가 지표
        loss, accuracy, mae, mse = self.model.evaluate(X_test, y_test, verbose=verbose)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'mae': mae,
            'mse': mse
        }
        
        logger.info(f"평가 결과: 손실={loss:.4f}, 정확도={accuracy:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
        
        return metrics
    
    def compute_advanced_metrics(self, X_test: np.ndarray, y_test: np.ndarray, verbose: int = 0) -> Dict[str, float]:
        """
        고급 평가 지표 계산 (Precision, Recall, F1-score, FPR, FNR)
        
        Args:
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 라벨
            verbose: 예측 진행 출력 레벨
            
        Returns:
            고급 평가 지표 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        
        logger.info("고급 평가 지표 계산 중...")
        
        # 예측 확률
        y_pred_prob = self.model.predict(X_test, verbose=verbose)
        
        # 확률을 바이너리 클래스로 변환 (임계값 0.5)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 혼동 행렬
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # 평가 지표 계산
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr = fp / (fp + tn)  # False Positive Rate
        fnr = fn / (fn + tp)  # False Negative Rate
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'fnr': fnr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        logger.info(f"고급 지표: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}")
        
        return metrics
    
    def plot_learning_curves(self) -> Tuple[plt.Figure, plt.Figure]:
        """
        학습 곡선(손실 및 정확도) 시각화
        
        Returns:
            손실 곡선 그래프, 정확도 곡선 그래프
        """
        if self.history is None:
            raise ValueError("모델 학습 이력이 없습니다. train_model()을 먼저 호출하세요.")
        
        logger.info("학습 곡선 시각화 중...")
        
        # 손실 곡선
        loss_fig, loss_ax = plt.subplots(figsize=(10, 6))
        loss_ax.plot(self.history['loss'], label='Training Loss')
        loss_ax.plot(self.history['val_loss'], label='Validation Loss')
        loss_ax.set_xlabel('Epoch')
        loss_ax.set_ylabel('Loss')
        loss_ax.set_title('Training and Validation Loss')
        loss_ax.legend()
        loss_ax.grid(True)
        
        # 정확도 곡선
        acc_fig, acc_ax = plt.subplots(figsize=(10, 6))
        acc_ax.plot(self.history['accuracy'], label='Training Accuracy')
        acc_ax.plot(self.history['val_accuracy'], label='Validation Accuracy')
        acc_ax.set_xlabel('Epoch')
        acc_ax.set_ylabel('Accuracy')
        acc_ax.set_title('Training and Validation Accuracy')
        acc_ax.legend()
        acc_ax.grid(True)
        
        return loss_fig, acc_fig
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        혼동 행렬 시각화
        
        Args:
            y_test: 실제 라벨
            y_pred: 예측 라벨
            
        Returns:
            혼동 행렬 그래프
        """
        logger.info("혼동 행렬 시각화 중...")
        
        # 혼동 행렬 계산
        cm = confusion_matrix(y_test, y_pred)
        
        # 시각화
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Normal', 'Attack'])
        ax.set_yticklabels(['Normal', 'Attack'])
        
        return fig
    
    def save_model(self, model_path: str) -> None:
        """
        모델 저장
        
        Args:
            model_path: 모델 저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        logger.info(f"모델 저장 중: {model_path}")
        self.model.save(model_path)
        logger.info("모델 저장 완료")
    
    def load_model(self, model_path: str) -> None:
        """
        모델 로드
        
        Args:
            model_path: 모델 로드 경로
        """
        logger.info(f"모델 로드 중: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("모델 로드 완료")


def usage_example():
    """
    HyDL-IDS 모델 사용 예시
    """
    print("===== HyDL-IDS 모델 사용 예시 =====")
    print("이 예시는 Car-Hacking 데이터셋에 HyDL-IDS 모델을 적용하는 방법을 보여줍니다.")
    print()
    
    # 1. 데이터 로드 (사용자가 실제 Car-Hacking 데이터셋 로드 필요)
    print("1. 데이터 로드")
    print("아래와 같은 형식으로 Car-Hacking 데이터셋을 로드합니다:")
    print("   - Timestamp: float (초)")
    print("   - CAN ID: 16진수 문자열")
    print("   - DLC: int (0-8)")
    print("   - DATA[0]-DATA[7]: 각 바이트에 대한 16진수 문자열")
    print("   - Flag/Tag: 문자열 ('R': 정상, 'T': 공격)")
    print()
    
    # 2. 데이터 분할 (학습/검증/테스트)
    print("2. 데이터 분할")
    print("데이터를 학습, 검증, 테스트 세트로 분할합니다:")
    print("   - 학습 세트: 67%")
    print("   - 검증 세트: 13%")
    print("   - 테스트 세트: 20%")
    print()
    print("# 코드 예시:")
    print("from sklearn.model_selection import train_test_split")
    print()
    print("# 먼저 테스트 세트 분리")
    print("X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)")
    print()
    print("# 나머지 데이터를 학습 및 검증 세트로 분할 (학습:검증 = 67:13 = 약 5:1)")
    print("X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)")
    print()
    
    # 3. HyDL-IDS 모델 초기화 및 데이터 전처리
    print("3. HyDL-IDS 모델 초기화 및 데이터 전처리")
    print("# 코드 예시:")
    print("# HyDL-IDS 모델 초기화 (윈도우 크기: 10, 스트라이드: 1)")
    print("hydl_ids = HyDL_IDS()")
    print()
    print("# 데이터 전처리")
    print("X_sequences, y_sequences = hydl_ids.preprocess_data(data)")
    print()
    print("# 전처리된 데이터 분할")
    print("# (위 2단계의 분할 코드 적용)")
    print()
    
    # 4. 모델 구축 및 컴파일
    print("4. 모델 구축 및 컴파일")
    print("# 코드 예시:")
    print("# 모델 구축")
    print("input_shape = (X_train.shape[1], X_train.shape[2])  # (윈도우 크기, 특성 수)")
    print("hydl_ids.build_model(input_shape)")
    print()
    print("# 모델 컴파일")
    print("hydl_ids.compile_model(learning_rate=0.001)")
    print()
    
    # 5. 모델 학습
    print("5. 모델 학습")
    print("# 코드 예시:")
    print("# 모델 학습 (배치 크기: 256, 에포크: 10, EarlyStopping 콜백 사용)")
    print("history = hydl_ids.train_model(")
    print("    X_train, y_train,")
    print("    X_val, y_val,")
    print("    batch_size=256,")
    print("    epochs=10,")
    print("    use_early_stopping=True")
    print(")")
    print()
    
    # 6. 모델 평가
    print("6. 모델 평가")
    print("# 코드 예시:")
    print("# 기본 평가 지표 (손실, 정확도, MAE, MSE)")
    print("metrics = hydl_ids.evaluate_model(X_test, y_test)")
    print()
    print("# 고급 평가 지표 (Precision, Recall, F1-score, FPR, FNR)")
    print("advanced_metrics = hydl_ids.compute_advanced_metrics(X_test, y_test)")
    print()
    
    # 7. 학습 곡선 시각화
    print("7. 학습 곡선 시각화")
    print("# 코드 예시:")
    print("# 손실 및 정확도 곡선 시각화")
    print("loss_fig, acc_fig = hydl_ids.plot_learning_curves()")
    print()
    print("# 혼동 행렬 시각화")
    print("y_pred = (hydl_ids.model.predict(X_test) > 0.5).astype(int)")
    print("cm_fig = hydl_ids.plot_confusion_matrix(y_test, y_pred)")
    print()
    
    # 8. 모델 저장 및 로드
    print("8. 모델 저장 및 로드")
    print("# 코드 예시:")
    print("# 모델 저장")
    print("hydl_ids.save_model('hydl_ids_model.h5')")
    print()
    print("# 모델 로드")
    print("new_hydl_ids = HyDL_IDS()")
    print("new_hydl_ids.load_model('hydl_ids_model.h5')")
    print()
    
    print("===== 예시 종료 =====")
    print("위 코드를 참고하여 실제 Car-Hacking 데이터셋에 HyDL-IDS 모델을 적용해 보세요.")


if __name__ == "__main__":
    # 사용 예시 출력
    usage_example()