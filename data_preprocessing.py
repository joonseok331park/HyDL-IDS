"""
HyDL-IDS: 차량 내부 네트워크 침입 탐지를 위한 데이터 전처리 모듈

이 모듈은 Car-Hacking 데이터셋에 대한 상세한 전처리 로직을 구현합니다.
주요 기능:
- CAN ID 처리: 16진수 CAN ID를 정수로 인코딩
- DLC 처리: DLC 값의 로그 변환 및 정규화
- DATA 필드 처리: 16진수 데이터 바이트 변환 및 정규화
- Flag/Tag 처리: 레이블 인코딩
- 데이터 재구성: 시퀀스 데이터로 변환
- 결측값 처리: 필드별 적절한 대체값 적용

작성자: Scout AI
날짜: 2025-05-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging
from typing import Tuple, Dict, List, Any, Union, Optional
import os


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CANDataPreprocessor:
    """
    CAN 버스 데이터 전처리를 위한 클래스
    
    CAN 데이터 필드별 전처리 및 시퀀스 생성 기능을 제공합니다.
    """
    
    def __init__(self, window_size: int = 10, stride: int = 1):
        """
        전처리기 초기화
        
        Args:
            window_size: 시퀀스 데이터 생성을 위한 윈도우 크기
            stride: 윈도우 이동 단위
        """
        self.window_size = window_size
        self.stride = stride
        
        # 인코더 및 스케일러 초기화
        self.can_id_encoder = None
        self.dlc_scaler = None
        self.data_scaler = None
        
        # 통계 정보 저장용
        self.stats = {}
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터프레임에 전체 전처리 파이프라인을 적용하고 학습
        
        Args:
            df: 원본 CAN 데이터프레임
            
        Returns:
            시퀀스 형태의 특성 배열, 라벨 배열
        """
        # 결측값이 있는 경우 처리 (훈련 전 필수)
        df_clean = self.handle_missing_values(df)
        
        # 필드별 전처리 (학습 모드)
        encoded_can_ids = self.process_can_id(df_clean['CAN ID'].values, fit=True)
        normalized_dlc = self.process_dlc(df_clean['DLC'].values, fit=True)
        normalized_data = self.process_data_bytes(
            df_clean[[f'DATA[{i}]' for i in range(8)]].values, 
            df_clean['DLC'].values, 
            fit=True
        )
        encoded_flags = self.process_flag(df_clean['Flag/Tag'].values)
        
        # 특성 결합
        features = np.column_stack([encoded_can_ids, normalized_dlc, normalized_data])
        
        # 시계열 시퀀스 생성
        X_sequences, y_sequences = self.create_sequences(features, encoded_flags)
        
        # 통계 정보 업데이트
        self._update_statistics(df_clean, features, encoded_flags)
        
        return X_sequences, y_sequences
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습된 전처리기를 사용하여 새 데이터 변환 (훈련 후 테스트/예측용)
        
        Args:
            df: 원본 CAN 데이터프레임
            
        Returns:
            시퀀스 형태의 특성 배열, 라벨 배열
        """
        if self.can_id_encoder is None or self.dlc_scaler is None or self.data_scaler is None:
            raise ValueError("전처리기가 학습되지 않았습니다. fit_transform()을 먼저 호출하세요.")
        
        # 결측값 처리
        df_clean = self.handle_missing_values(df)
        
        # 필드별 전처리 (변환 모드)
        encoded_can_ids = self.process_can_id(df_clean['CAN ID'].values, fit=False)
        normalized_dlc = self.process_dlc(df_clean['DLC'].values, fit=False)
        normalized_data = self.process_data_bytes(
            df_clean[[f'DATA[{i}]' for i in range(8)]].values, 
            df_clean['DLC'].values, 
            fit=False
        )
        encoded_flags = self.process_flag(df_clean['Flag/Tag'].values)
        
        # 특성 결합
        features = np.column_stack([encoded_can_ids, normalized_dlc, normalized_data])
        
        # 시계열 시퀀스 생성
        X_sequences, y_sequences = self.create_sequences(features, encoded_flags)
        
        return X_sequences, y_sequences
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CAN 데이터의 결측값 처리
        
        Args:
            df: 원본 CAN 데이터프레임
            
        Returns:
            결측값이 처리된 데이터프레임
        """
        # 복사본 생성하여 원본 데이터 보존
        df_clean = df.copy()
        
        # 결측값 유무 확인
        has_missing = df_clean.isnull().any().any()
        
        if has_missing:
            logger.warning("결측값이 발견되었습니다. 필드별 적절한 대체값을 적용합니다.")
            
            # 각 필드별 처리
            # 1. CAN ID (범주형) - 최빈값으로 대체
            if df_clean['CAN ID'].isnull().any():
                can_id_mode = df_clean['CAN ID'].mode()[0]
                # 수정: chained assignment 경고 해결
                df_clean = df_clean.fillna({'CAN ID': can_id_mode})
                logger.info(f"CAN ID 결측값을 최빈값({can_id_mode})으로 대체했습니다.")
            
            # 2. DLC (수치형) - 중앙값으로 대체
            if df_clean['DLC'].isnull().any():
                dlc_median = df_clean['DLC'].median()
                # 수정: chained assignment 경고 해결
                df_clean = df_clean.fillna({'DLC': dlc_median})
                logger.info(f"DLC 결측값을 중앙값({dlc_median})으로 대체했습니다.")
            
            # 3. DATA 필드 (16진수 문자열)
            # 각 DATA 필드에 대해 최빈값으로 대체하거나, DLC 값에 따라 'M' 할당
            for i in range(8):
                col = f'DATA[{i}]'
                if df_clean[col].isnull().any():
                    # 수정: chained assignment 경고 해결
                    fill_dict = {col: 'M'}
                    df_clean = df_clean.fillna(fill_dict)
                    logger.info(f"DATA[{i}] 결측값을 'M'으로 대체했습니다.")
            
            # 4. Flag/Tag (범주형) - 결측값이 있는 경우 'R'(정상)으로 간주
            # (보안 관점에서는 의심스러운 경우 공격으로 간주할 수도 있으나, 
            # 여기서는 일반적인 상황에서 대부분 정상 트래픽이므로 'R'로 기본값 설정)
            if df_clean['Flag/Tag'].isnull().any():
                df_clean = df_clean.fillna({'Flag/Tag': 'R'})
                logger.info("Flag/Tag 결측값을 'R'(정상)으로 대체했습니다.")
            
            # 결측값 처리 후 확인
            if df_clean.isnull().any().any():
                logger.warning("일부 결측값이 여전히 남아 있습니다. 확인이 필요합니다.")
            else:
                logger.info("모든 결측값이 성공적으로 처리되었습니다.")
        else:
            logger.info("결측값이 없습니다. 추가 처리 없이 진행합니다.")
        
        return df_clean
    
    def process_can_id(self, can_ids: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        CAN ID 전처리: 16진수 CAN ID를 정수로 인코딩
        
        Args:
            can_ids: 16진수 문자열 형태의 CAN ID 배열
            fit: 인코더 학습 여부 (True: 학습+변환, False: 변환만)
            
        Returns:
            정수 인코딩된 CAN ID 배열
        """
        logger.info("CAN ID 전처리 중...")
        
        # CAN ID를 문자열로 변환 (혼합 유형 방지)
        can_ids_str = np.array([str(cid) for cid in can_ids])
        
        # 16진수 형식 정규화 (0x 접두사 제거 등)
        normalized_ids = []
        for cid in can_ids_str:
            # 0x 접두사 제거
            if cid.lower().startswith('0x'):
                cid = cid[2:]
            # 빈 문자열 체크
            if not cid:
                cid = '0'
            normalized_ids.append(cid.upper())
        
        normalized_ids = np.array(normalized_ids)
        
        # LabelEncoder를 사용하여 고유한 CAN ID를 0부터 N-1까지의 정수로 매핑
        if fit or self.can_id_encoder is None:
            self.can_id_encoder = LabelEncoder()
            encoded_can_ids = self.can_id_encoder.fit_transform(normalized_ids)
            logger.info(f"CAN ID 인코더 학습 완료. 고유 CAN ID 수: {len(self.can_id_encoder.classes_)}")
        else:
            try:
                encoded_can_ids = self.can_id_encoder.transform(normalized_ids)
            except ValueError as e:
                # 학습 데이터에 없는 새로운 CAN ID 처리
                logger.warning(f"학습되지 않은 CAN ID가 발견되었습니다: {e}")
                # 알려지지 않은 값을 0으로 처리 (또는 다른 전략 선택 가능)
                encoded_can_ids = np.zeros_like(normalized_ids, dtype=int)
                for i, cid in enumerate(normalized_ids):
                    try:
                        encoded_can_ids[i] = self.can_id_encoder.transform([cid])[0]
                    except ValueError:
                        encoded_can_ids[i] = 0  # 또는 다른 기본값
                        logger.warning(f"알려지지 않은 CAN ID '{cid}' 인덱스 {i}를 0으로 대체")
        
        return encoded_can_ids
    
    def process_dlc(self, dlc_values: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        DLC(Data Length Code) 전처리: log(값 + 1) 변환 후 0-1 범위로 정규화
        
        논문 제시 수식: normalize(x) = ln(val_i+1)/ln(9) 사용
        
        Args:
            dlc_values: 정수 형태의 DLC 값 배열 (0-8)
            fit: 호환성을 위해 유지하지만 실제로는 사용되지 않음
            
        Returns:
            정규화된 DLC 값 배열
        """
        logger.info("DLC 전처리 중...")
        
        # DLC 값을 정수로 변환
        try:
            dlc_int = np.array(dlc_values, dtype=int)
        except (ValueError, TypeError):
            logger.warning("DLC 값을 정수로 변환하는 중 오류가 발생했습니다. 문자열 또는 누락된 값이 있는지 확인합니다.")
            # 개별 요소 변환 시도
            dlc_int = np.zeros_like(dlc_values, dtype=int)
            for i, value in enumerate(dlc_values):
                try:
                    dlc_int[i] = int(value)
                except (ValueError, TypeError):
                    dlc_int[i] = 0  # 변환 불가능한 값은 0으로 대체
                    logger.warning(f"변환 불가능한 DLC 값 '{value}' 인덱스 {i}를 0으로 대체")
        
        # 유효범위 확인 및 조정 (0-8)
        dlc_clipped = np.clip(dlc_int, 0, 8)
        if not np.array_equal(dlc_int, dlc_clipped):
            logger.warning("일부 DLC 값이 유효 범위(0-8)를 벗어나 조정되었습니다.")
        
        # log(값 + 1) 변환
        log_dlc = np.log1p(dlc_clipped)
        
        # 논문 제시 수식을 사용한 정규화: ln(val_i+1)/ln(9)
        normalized_dlc = log_dlc / np.log(9)
        logger.info("DLC 정규화 완료 (논문 제시 수식 ln(val_i+1)/ln(9) 적용)")
        
        return normalized_dlc
    
    def process_data_bytes(self, data_bytes: np.ndarray, dlc_values: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        DATA 바이트 전처리:
        1. 16진수 데이터 바이트를 10진수로 변환 (0-255)
        2. 존재하지 않는 바이트('M')는 256으로 변환
        3. log(값 + 1) 변환 후 0-1 범위로 정규화
        
        논문 제시 수식: normalize(x) = ln(val_i+1)/ln(257) 사용
        
        Args:
            data_bytes: DATA[0]-DATA[7] 필드 값 배열 (8 바이트)
            dlc_values: 각 행의 DLC 값 배열
            fit: 호환성을 위해 유지하지만 실제로는 사용되지 않음
            
        Returns:
            정규화된 DATA 바이트 배열 (차원: [샘플 수, 8])
        """
        logger.info("DATA 바이트 전처리 중...")
        
        # 결과 배열 초기화 (샘플 수 x 8 바이트)
        n_samples = len(dlc_values)
        decimal_data = np.zeros((n_samples, 8))
        
        # DLC 값을 정수로 변환
        dlc_int = np.zeros_like(dlc_values, dtype=int)
        for i, value in enumerate(dlc_values):
            try:
                dlc_int[i] = int(value)
            except (ValueError, TypeError):
                dlc_int[i] = 0
                logger.warning(f"변환 불가능한 DLC 값 '{value}' 인덱스 {i}를 0으로 대체")
        
        # DLC 값 클리핑 (0-8 범위로 제한)
        dlc_int = np.clip(dlc_int, 0, 8)
        
        # 각 샘플에 대해 처리
        for i in range(n_samples):
            dlc = dlc_int[i]
            
            # DLC 값에 따라 실제 데이터 바이트 수 결정
            for j in range(8):
                if j < dlc:
                    # 16진수 데이터 바이트를 10진수로 변환
                    try:
                        data_byte = data_bytes[i, j]
                        
                        # None, nan 체크
                        if data_byte is None or (isinstance(data_byte, float) and np.isnan(data_byte)):
                            decimal_data[i, j] = 256
                        # 'M' 체크 (결측값 표시)
                        elif isinstance(data_byte, str) and data_byte.upper() == 'M':
                            decimal_data[i, j] = 256
                        # 16진수 변환 시도
                        elif isinstance(data_byte, str):
                            # 0x 접두사 제거
                            if data_byte.lower().startswith('0x'):
                                data_byte = data_byte[2:]
                            # 빈 문자열 체크 - 수정: 빈 문자열도 유효하지 않은 값으로 처리
                            if not data_byte:
                                decimal_data[i, j] = 256
                                logger.warning(f"빈 문자열 데이터 값 인덱스 [{i},{j}]를 256으로 대체")
                            else:
                                try:
                                    decimal_data[i, j] = int(data_byte, 16)
                                except ValueError:
                                    decimal_data[i, j] = 256
                                    logger.warning(f"유효하지 않은 16진수 값 '{data_byte}' 인덱스 [{i},{j}]를 256으로 대체")
                        # 직접 숫자인 경우 - 수정: 정수형으로 변환
                        else:
                            try:
                                # 정수형으로 변환 (소수점이 있는 경우 버림)
                                decimal_data[i, j] = int(float(data_byte))
                            except (ValueError, TypeError):
                                decimal_data[i, j] = 256
                                logger.warning(f"변환 불가능한 데이터 바이트 값 '{data_byte}' 인덱스 [{i},{j}]를 256으로 대체")
                    except (IndexError, TypeError) as e:
                        decimal_data[i, j] = 256
                        logger.warning(f"데이터 바이트 처리 중 오류 발생: {e}. 인덱스 [{i},{j}]를 256으로 대체")
                else:
                    # DLC보다 인덱스가 큰 경우 (존재하지 않는 바이트) 256 할당
                    decimal_data[i, j] = 256
        
        # 값 클리핑 (0-256 범위로 제한)
        decimal_data = np.clip(decimal_data, 0, 256)
        
        # log(값 + 1) 변환
        log_data = np.log1p(decimal_data)
        
        # 논문 제시 수식을 사용한 정규화: ln(val_i+1)/ln(257)
        normalized_data = log_data / np.log(257)
        logger.info("DATA 바이트 정규화 완료 (논문 제시 수식 ln(val_i+1)/ln(257) 적용)")
        
        return normalized_data
    
    def process_flag(self, flags: np.ndarray) -> np.ndarray:
        """
        Flag/Tag 전처리: 'R'(정상)을 0으로, 'T'(공격)를 1로 인코딩
        
        Args:
            flags: 'R' 또는 'T' 문자열로 이루어진 Flag/Tag 값 배열
            
        Returns:
            인코딩된 라벨 배열 (0: 정상, 1: 공격)
        """
        logger.info("Flag/Tag 전처리 중...")
        
        # 'R'을 0으로, 'T'를 1로 변환
        encoded_flags = np.zeros(len(flags), dtype=int)
        
        for i, flag in enumerate(flags):
            if flag is None:
                encoded_flags[i] = 0  # 결측값은 정상(0)으로 처리
                logger.warning(f"Flag/Tag 인덱스 {i}의 결측값을 0(정상)으로 대체")
            elif isinstance(flag, str):
                if flag.upper() == 'T':
                    encoded_flags[i] = 1
                else:
                    # 'R' 또는 기타 문자열은 0으로 처리
                    encoded_flags[i] = 0
                    if flag.upper() != 'R':
                        logger.warning(f"예상치 못한 Flag/Tag 값 '{flag}' 인덱스 {i}를 0(정상)으로 대체")
            else:
                # 문자열이 아닌 경우
                try:
                    # 숫자를 이진값으로 변환 (0은 0, 0 이외는 1)
                    encoded_flags[i] = 1 if float(flag) != 0 else 0
                    logger.warning(f"숫자 형태의 Flag/Tag 값 {flag} 인덱스 {i}를 {encoded_flags[i]}로 변환")
                except (ValueError, TypeError):
                    encoded_flags[i] = 0
                    logger.warning(f"변환 불가능한 Flag/Tag 값 '{flag}' 인덱스 {i}를 0(정상)으로 대체")
        
        # 클래스 분포 확인
        n_normal = np.sum(encoded_flags == 0)
        n_attack = np.sum(encoded_flags == 1)
        logger.info(f"클래스 분포: 정상(0)={n_normal}, 공격(1)={n_attack}")
        
        return encoded_flags
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터를 모델 입력에 적합한 시퀀스 형태로 변환
        
        Args:
            features: 특성 배열 [샘플 수, 특성 수]
            labels: 라벨 배열 [샘플 수]
            
        Returns:
            시퀀스 형태의 특성 배열 [시퀀스 수, 윈도우 크기, 특성 수]
            시퀀스 형태의 라벨 배열 [시퀀스 수] - 각 시퀀스의 마지막 샘플 라벨 사용
        """
        logger.info(f"시퀀스 데이터 생성 중 (윈도우 크기: {self.window_size}, 스트라이드: {self.stride})...")
        
        n_samples, n_features = features.shape
        
        # 시퀀스 수 계산
        n_sequences = max(0, (n_samples - self.window_size) // self.stride + 1)
        
        if n_sequences <= 0:
            raise ValueError(f"데이터 샘플 수({n_samples})가 윈도우 크기({self.window_size})보다 작아 시퀀스를 생성할 수 없습니다.")
        
        # 시퀀스 배열 초기화
        X = np.zeros((n_sequences, self.window_size, n_features))
        y = np.zeros(n_sequences)
        
        # 윈도우 슬라이딩 방식으로 시퀀스 생성
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # 범위 체크
            if end_idx > n_samples:
                logger.warning(f"시퀀스 {i}의 끝 인덱스({end_idx})가 데이터 범위({n_samples})를 초과합니다. 시퀀스를 건너뜁니다.")
                continue
            
            X[i] = features[start_idx:end_idx]
            
            # 각 시퀀스의 라벨은 마지막 샘플의 라벨로 설정
            y[i] = labels[end_idx - 1]
            
            # 공격 탐지를 위해 시퀀스 내 공격이 하나라도 있으면 공격으로 간주하는 대안적 접근법
            # (사용자 요구사항에 따라 주석 해제하여 사용)
            # if np.any(labels[start_idx:end_idx] == 1):
            #     y[i] = 1
        
        logger.info(f"생성된 시퀀스 수: {n_sequences}")
        return X, y
    
    def _update_statistics(self, df: pd.DataFrame, features: np.ndarray, labels: np.ndarray) -> None:
        """
        전처리 관련 통계 정보 업데이트
        
        Args:
            df: 원본 데이터프레임
            features: 전처리된 특성 배열
            labels: 인코딩된 라벨 배열
        """
        # CAN ID 통계
        if self.can_id_encoder is not None:
            self.stats['can_id'] = {
                'unique_count': len(self.can_id_encoder.classes_),
                'classes': self.can_id_encoder.classes_.tolist(),
                'distribution': {
                    can_id: np.sum(df['CAN ID'] == can_id) for can_id in self.can_id_encoder.classes_
                }
            }
        
        # DLC 통계
        dlc_values = df['DLC'].values
        self.stats['dlc'] = {
            'min': np.min(dlc_values),
            'max': np.max(dlc_values),
            'mean': np.mean(dlc_values),
            'median': np.median(dlc_values),
            'distribution': {
                value: np.sum(dlc_values == value) for value in range(9)  # 0-8
            }
        }
        
        # 라벨 통계
        self.stats['labels'] = {
            'normal_count': np.sum(labels == 0),
            'attack_count': np.sum(labels == 1),
            'normal_percentage': np.mean(labels == 0) * 100,
            'attack_percentage': np.mean(labels == 1) * 100
        }
        
        # 전처리된 특성 통계
        self.stats['features'] = {
            'shape': features.shape,
            'min': np.min(features),
            'max': np.max(features),
            'mean': np.mean(features),
            'std': np.std(features)
        }


class CANDataVisualizer:
    """
    CAN 데이터 시각화를 위한 클래스
    """
    
    @staticmethod
    def plot_can_id_distribution(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
        """
        CAN ID 분포 시각화
        
        Args:
            df: CAN 데이터프레임
            top_n: 표시할 상위 CAN ID 수
            
        Returns:
            분포 그래프
        """
        # CAN ID 빈도 계산
        can_id_counts = df['CAN ID'].value_counts()
        
        # 상위 N개 선택
        top_can_ids = can_id_counts.head(top_n)
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(15, 8))
        top_can_ids.plot(kind='bar', ax=ax)
        
        ax.set_title(f'Top {top_n} CAN ID Distribution')
        ax.set_xlabel('CAN ID')
        ax.set_ylabel('Count')
        ax.grid(axis='y')
        
        # 막대 위에 개수 표시
        for i, count in enumerate(top_can_ids):
            ax.text(i, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_dlc_distribution(df: pd.DataFrame) -> plt.Figure:
        """
        DLC 값 분포 시각화
        
        Args:
            df: CAN 데이터프레임
            
        Returns:
            분포 그래프
        """
        # DLC 빈도 계산
        dlc_counts = df['DLC'].value_counts().sort_index()
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        dlc_counts.plot(kind='bar', ax=ax, color='skyblue')
        
        ax.set_title('DLC Value Distribution')
        ax.set_xlabel('DLC Value')
        ax.set_ylabel('Count')
        ax.grid(axis='y')
        
        # 막대 위에 개수 표시
        for i, count in enumerate(dlc_counts):
            ax.text(i, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_flag_distribution(df: pd.DataFrame) -> plt.Figure:
        """
        Flag/Tag 분포 시각화
        
        Args:
            df: CAN 데이터프레임
            
        Returns:
            분포 그래프
        """
        # Flag/Tag 빈도 계산
        flag_counts = df['Flag/Tag'].value_counts()
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(8, 8))
        flag_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, 
                         colors=['lightgreen', 'salmon'] if 'T' in flag_counts.index else ['lightgreen'])
        
        ax.set_title('Normal vs Attack Distribution')
        ax.set_ylabel('')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_attack_by_can_id(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
        """
        CAN ID별 공격 비율 시각화
        
        Args:
            df: CAN 데이터프레임
            top_n: 표시할 상위 CAN ID 수
            
        Returns:
            분포 그래프
        """
        # CAN ID별 공격 및 정상 메시지 수 계산
        attack_counts = df[df['Flag/Tag'] == 'T']['CAN ID'].value_counts()
        normal_counts = df[df['Flag/Tag'] == 'R']['CAN ID'].value_counts()
        
        # 공격이 발생한 상위 CAN ID 선택
        top_attack_ids = attack_counts.head(top_n).index
        
        # 데이터 준비
        attack_data = []
        normal_data = []
        
        for can_id in top_attack_ids:
            attack_data.append(attack_counts.get(can_id, 0))
            normal_data.append(normal_counts.get(can_id, 0))
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(top_attack_ids))
        width = 0.35
        
        ax.bar(x - width/2, normal_data, width, label='Normal', color='lightgreen')
        ax.bar(x + width/2, attack_data, width, label='Attack', color='salmon')
        
        ax.set_title('Attack vs Normal by CAN ID')
        ax.set_xlabel('CAN ID')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(top_attack_ids)
        ax.legend()
        ax.grid(axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_attack_timeline(df: pd.DataFrame, bin_width: float = 1.0) -> plt.Figure:
        """
        공격 타임라인 시각화
        
        Args:
            df: CAN 데이터프레임
            bin_width: 히스토그램 빈 크기 (초)
            
        Returns:
            타임라인 그래프
        """
        # 타임스탬프를 기준으로 정렬
        sorted_df = df.sort_values('Timestamp')
        
        # 시작 시간과 종료 시간
        start_time = sorted_df['Timestamp'].min()
        end_time = sorted_df['Timestamp'].max()
        
        # 시간 범위 및 빈 생성
        bins = np.arange(start_time, end_time + bin_width, bin_width)
        
        # 공격 및 정상 메시지 분리
        attack_df = sorted_df[sorted_df['Flag/Tag'] == 'T']
        normal_df = sorted_df[sorted_df['Flag/Tag'] == 'R']
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # 히스토그램 그리기
        ax.hist(normal_df['Timestamp'], bins=bins, alpha=0.5, label='Normal', color='lightgreen')
        ax.hist(attack_df['Timestamp'], bins=bins, alpha=0.5, label='Attack', color='salmon')
        
        ax.set_title('CAN Traffic Timeline (Normal vs Attack)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Number of Messages')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig


def visualize_preprocessed_data(X: np.ndarray, y: np.ndarray, sample_idx: int = 0) -> Tuple[plt.Figure, plt.Figure]:
    """
    전처리된 시퀀스 데이터 시각화
    
    Args:
        X: 전처리된 시퀀스 데이터 [시퀀스 수, 윈도우 크기, 특성 수]
        y: 시퀀스 라벨 [시퀀스 수]
        sample_idx: 시각화할 샘플 인덱스
        
    Returns:
        특성 히트맵, 라벨 분포 그래프 튜플
    """
    # 샘플 검증
    if sample_idx >= len(X):
        sample_idx = 0
        logger.warning(f"요청한 샘플 인덱스({sample_idx})가 범위를 벗어났습니다. 첫 번째 샘플을 사용합니다.")
    
    # 특성 히트맵
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.heatmap(X[sample_idx], cmap='viridis', ax=ax1)
    
    ax1.set_title(f'Features Heatmap (Sample {sample_idx}, Label: {"Attack" if y[sample_idx] == 1 else "Normal"})')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Time Step')
    
    # 예시 특성 인덱스 설명 추가
    feature_labels = ['CAN ID', 'DLC']
    for i in range(8):
        feature_labels.append(f'DATA[{i}]')
    
    # 라벨 개수가 특성 개수와 맞지 않으면 조정
    if len(feature_labels) != X.shape[2]:
        feature_labels = [f'Feature {i}' for i in range(X.shape[2])]
    
    # 라벨 분포
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    labels = ['Normal', 'Attack']
    counts = [np.sum(y == 0), np.sum(y == 1)]
    
    ax2.bar(labels, counts, color=['lightgreen', 'salmon'])
    ax2.set_title('Label Distribution')
    ax2.set_ylabel('Count')
    
    # 막대 위에 개수 및 비율 표시
    total = len(y)
    for i, count in enumerate(counts):
        percentage = count / total * 100
        ax2.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    ax2.grid(axis='y')
    
    plt.tight_layout()
    return fig1, fig2


def example_usage():
    """
    CANDataPreprocessor 및 CANDataVisualizer 사용 예시
    """
    print("===== CAN 데이터 전처리 및 시각화 예시 =====")
    print("이 예시는 Car-Hacking 데이터셋에 전처리 및 시각화 모듈을 적용하는 방법을 보여줍니다.")
    print()
    
    # 1. 데이터 로드 (사용자가 실제 Car-Hacking 데이터셋 로드 필요)
    print("1. 데이터 로드")
    print("# 코드 예시:")
    print("import pandas as pd")
    print("from utils import load_can_data")
    print()
    print("# CSV 파일 로드")
    print("data = load_can_data('your_can_data.csv')")
    print("print(f'데이터 형태: {data.shape}')")
    print()
    
    # 2. 데이터 시각화
    print("2. 데이터 시각화")
    print("# 코드 예시:")
    print("from data_preprocessing import CANDataVisualizer")
    print()
    print("# 시각화 객체 생성")
    print("visualizer = CANDataVisualizer()")
    print()
    print("# CAN ID 분포 시각화")
    print("can_id_fig = visualizer.plot_can_id_distribution(data)")
    print("can_id_fig.savefig('can_id_distribution.png')")
    print()
    print("# DLC 분포 시각화")
    print("dlc_fig = visualizer.plot_dlc_distribution(data)")
    print("dlc_fig.savefig('dlc_distribution.png')")
    print()
    print("# Flag/Tag 분포 시각화")
    print("flag_fig = visualizer.plot_flag_distribution(data)")
    print("flag_fig.savefig('flag_distribution.png')")
    print()
    print("# CAN ID별 공격 비율 시각화")
    print("attack_by_id_fig = visualizer.plot_attack_by_can_id(data)")
    print("attack_by_id_fig.savefig('attack_by_can_id.png')")
    print()
    print("# 공격 타임라인 시각화")
    print("timeline_fig = visualizer.plot_attack_timeline(data)")
    print("timeline_fig.savefig('attack_timeline.png')")
    print()
    
    # 3. 데이터 전처리
    print("3. 데이터 전처리")
    print("# 코드 예시:")
    print("from data_preprocessing import CANDataPreprocessor")
    print()
    print("# 전처리기 초기화 (윈도우 크기: 10, 스트라이드: 1)")
    print("preprocessor = CANDataPreprocessor(window_size=10, stride=1)")
    print()
    print("# 데이터 전처리 및 인코더/스케일러 학습")
    print("X_sequences, y_sequences = preprocessor.fit_transform(data)")
    print("print(f'전처리된 데이터 형태: X={X_sequences.shape}, y={y_sequences.shape}')")
    print()
    
    # 4. 전처리된 데이터 시각화
    print("4. 전처리된 데이터 시각화")
    print("# 코드 예시:")
    print("from data_preprocessing import visualize_preprocessed_data")
    print()
    print("# 첫 번째 샘플 시각화")
    print("feature_fig, label_fig = visualize_preprocessed_data(X_sequences, y_sequences, sample_idx=0)")
    print("feature_fig.savefig('preprocessed_features.png')")
    print("label_fig.savefig('preprocessed_labels.png')")
    print()
    
    # 5. 새로운 데이터에 전처리 적용
    print("5. 새로운 데이터에 전처리 적용")
    print("# 코드 예시:")
    print("# 이미 학습된 전처리기를 새 데이터에 적용")
    print("new_data = load_can_data('new_can_data.csv')")
    print("X_new, y_new = preprocessor.transform(new_data)")
    print()
    
    print("===== 예시 종료 =====")
    print("위 코드를 참고하여 실제 Car-Hacking 데이터셋에 전처리 모듈을 적용해 보세요.")


def main(dataset_path=None, window_size=10, stride=1, results_dir="preprocessing_results"):
    """
    데이터 전처리 파이프라인의 메인 함수
    
    Args:
        dataset_path: 데이터셋 파일 경로, 기본값은 "Car-Hacking Dataset/Fuzzy_dataset.csv"
        window_size: 시퀀스 윈도우 크기, 기본값은 10
        stride: 시퀀스 스트라이드, 기본값은 1
        results_dir: 결과물 저장 경로, 기본값은 "preprocessing_results"
    """
    if dataset_path is None:
        dataset_path = os.path.join("Car-Hacking Dataset", "Fuzzy_dataset.csv")
    
    try:
        # 데이터 로드 - 컬럼명 없이 모든 행이 데이터인 경우를 처리
        logger.info(f"데이터셋 로드 중: {dataset_path}")
        
        # pandas 버전에 따라 다른 옵션 사용 (호환성 고려)
        try:
            # 최신 pandas 버전 (1.3.0 이상)
            data = pd.read_csv(
                dataset_path, 
                header=None,
                engine='python',  # 파이썬 엔진 사용으로 더 유연하게 처리
                sep=None,         # 구분자를 자동 감지
                on_bad_lines='skip'  # 잘못된 형식의 행 건너뛰기
            )
        except TypeError:
            # 이전 pandas 버전
            data = pd.read_csv(
                dataset_path, 
                header=None,
                engine='python',  # 파이썬 엔진 사용으로 더 유연하게 처리
                sep=None,         # 구분자를 자동 감지
                error_bad_lines=False,  # 이전 버전 옵션
                warn_bad_lines=True     # 이전 버전 옵션
            )
        
        logger.info(f"데이터 로드 완료. 형태: {data.shape}")
        
        # 컬럼 수 확인
        num_columns = data.shape[1]
        logger.info(f"데이터셋 컬럼 수: {num_columns}")
        
        # 기본 컬럼 구조 매핑 (Fuzzy_dataset.csv의 일반적 구조 기준)
        # 컬럼명 직접 할당
        if num_columns >= 12:  # 예상되는 컬럼 수 (타임스탬프 + CAN ID + DLC + 8개 DATA + Flag)
            # 새로운 컬럼명 목록 생성
            new_columns = ['Timestamp', 'CAN ID', 'DLC'] 
            for i in range(8):
                new_columns.append(f'DATA[{i}]')
            new_columns.append('Flag/Tag')
            
            # 컬럼 수가 예상보다 많거나 적은 경우 처리
            if len(new_columns) > num_columns:
                new_columns = new_columns[:num_columns]
                logger.warning(f"컬럼 수가 예상보다 적습니다. 처음 {num_columns}개 컬럼만 매핑합니다.")
            elif len(new_columns) < num_columns:
                for i in range(len(new_columns), num_columns):
                    new_columns.append(f'Extra_{i}')
                logger.warning(f"컬럼 수가 예상보다 많습니다. 추가 컬럼은 'Extra_N'으로 명명합니다.")
            
            # 컬럼명 할당
            data.columns = new_columns
            logger.info(f"할당된 컬럼명: {new_columns}")
        else:
            logger.warning(f"예상보다 컬럼 수가 적습니다 ({num_columns}개). 기본 컬럼명을 할당합니다.")
            # 컬럼이 적은 경우 기본 이름 할당
            basic_columns = []
            for i in range(num_columns):
                basic_columns.append(f'Column_{i}')
            data.columns = basic_columns
            logger.info(f"할당된 기본 컬럼명: {basic_columns}")
            
            # 필요한 컬럼 매핑 (최소한의 처리를 위해)
            if num_columns >= 3:
                data = data.rename(columns={
                    'Column_0': 'CAN ID',
                    'Column_1': 'DLC',
                    'Column_' + str(num_columns-1): 'Flag/Tag'
                })
                # DATA 컬럼 매핑
                for i in range(min(8, num_columns-3)):  # -3은 CAN ID, DLC, Flag/Tag를 고려
                    data = data.rename(columns={f'Column_{i+2}': f'DATA[{i}]'})
                logger.info("최소한의 컬럼 매핑을 적용했습니다.")
        
        # DATA 컬럼 확인
        data_byte_columns = [col for col in data.columns if 'DATA[' in col]
        logger.info(f"DATA 컬럼: {data_byte_columns}")
        
        # 데이터 시각화
        logger.info("데이터 시각화 중...")
        visualizer = CANDataVisualizer()
        
        # 결과물 저장 폴더 생성
        os.makedirs(results_dir, exist_ok=True)
        
        # CAN ID 분포 시각화
        can_id_fig = visualizer.plot_can_id_distribution(data)
        can_id_fig.savefig(os.path.join(results_dir, 'can_id_distribution.png'))
        
        # DLC 분포 시각화
        dlc_fig = visualizer.plot_dlc_distribution(data)
        dlc_fig.savefig(os.path.join(results_dir, 'dlc_distribution.png'))
        
        # Flag/Tag 분포 시각화
        flag_fig = visualizer.plot_flag_distribution(data)
        flag_fig.savefig(os.path.join(results_dir, 'flag_distribution.png'))
        
        # CAN ID별 공격 비율 시각화
        attack_by_id_fig = visualizer.plot_attack_by_can_id(data)
        attack_by_id_fig.savefig(os.path.join(results_dir, 'attack_by_can_id.png'))
        
        # 공격 타임라인 시각화 (Timestamp 컬럼이 있는 경우에만)
        if 'Timestamp' in data.columns:
            timeline_fig = visualizer.plot_attack_timeline(data)
            timeline_fig.savefig(os.path.join(results_dir, 'attack_timeline.png'))
        
        # 데이터 전처리
        logger.info("데이터 전처리 중...")
        preprocessor = CANDataPreprocessor(window_size=window_size, stride=stride)
        X_sequences, y_sequences = preprocessor.fit_transform(data)
        logger.info(f"전처리된 데이터 형태: X={X_sequences.shape}, y={y_sequences.shape}")
        
        # 모델 통계 출력
        logger.info("전처리 통계:")
        for key, value in preprocessor.stats.items():
            if key == 'can_id':
                logger.info(f"  고유 CAN ID 수: {value['unique_count']}")
            elif key == 'labels':
                logger.info(f"  정상 샘플: {value['normal_count']} ({value['normal_percentage']:.2f}%)")
                logger.info(f"  공격 샘플: {value['attack_count']} ({value['attack_percentage']:.2f}%)")
        
        # 전처리된 데이터 시각화
        logger.info("전처리된 데이터 시각화 중...")
        feature_fig, label_fig = visualize_preprocessed_data(X_sequences, y_sequences, sample_idx=0)
        feature_fig.savefig(os.path.join(results_dir, 'preprocessed_features.png'))
        label_fig.savefig(os.path.join(results_dir, 'preprocessed_labels.png'))
        
        # 추가 샘플 시각화 (데이터 크기에 따라 몇 개 더 시각화)
        for i in range(1, min(5, len(X_sequences))):
            feature_fig, _ = visualize_preprocessed_data(X_sequences, y_sequences, sample_idx=i)
            feature_fig.savefig(os.path.join(results_dir, f'preprocessed_features_sample_{i}.png'))
        
        # 전처리된 데이터 저장
        logger.info("전처리된 데이터 저장 중...")
        np.save(os.path.join(results_dir, 'X_preprocessed.npy'), X_sequences)
        np.save(os.path.join(results_dir, 'y_preprocessed.npy'), y_sequences)
        
        logger.info(f"처리 완료. 결과가 '{results_dir}' 폴더에 저장되었습니다.")
        
        # 처리된 데이터 및 전처리기 반환 (다른 모듈에서 사용 가능하도록)
        return data, preprocessor, X_sequences, y_sequences
        
    except FileNotFoundError:
        logger.error(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        logger.info("데이터셋 파일이 지정된 경로에 있는지 확인하세요.")
        return None, None, None, None
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None


if __name__ == "__main__":
    # 실제 데이터셋으로 테스트 실행
    main()
    
    # 필요한 경우 사용 예시 출력
    # example_usage()
    