"""
HyDL-IDS 모델 유틸리티 함수

이 모듈은 Car-Hacking 데이터셋 처리 및 모델 평가를 위한 유틸리티 함수를 제공합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Dict, List, Any, Union
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_can_data(file_path: str) -> pd.DataFrame:
    """
    Car-Hacking 데이터셋 CSV 파일 로드
    
    Args:
        file_path: 데이터셋 CSV 파일 경로
        
    Returns:
        로드된 데이터프레임
    """
    logger.info(f"데이터셋 로드 중: {file_path}")
    
    try:
        # 헤더 없이 CSV 파일 로드 (data_preprocessing.py와 일관성 유지)
        try:
            # 최신 pandas 버전 (1.3.0 이상)
            df = pd.read_csv(
                file_path, 
                header=None,
                engine='python',  # 파이썬 엔진 사용으로 더 유연하게 처리
                sep=None,         # 구분자를 자동 감지
                on_bad_lines='skip'  # 잘못된 형식의 행 건너뛰기
            )
        except TypeError:
            # 이전 pandas 버전
            df = pd.read_csv(
                file_path, 
                header=None,
                engine='python',
                sep=None,
                error_bad_lines=False,
                warn_bad_lines=True
            )
        
        # 컬럼 수 확인
        num_columns = df.shape[1]
        logger.info(f"데이터셋 컬럼 수: {num_columns}")
        
        # 기본 컬럼 구조 매핑 (표준 컬럼 구조 가정)
        if num_columns >= 11:  # 타임스탬프 + CAN ID + DLC + 8개 DATA + Flag
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
            df.columns = new_columns
            logger.info(f"할당된 컬럼명: {new_columns}")
        else:
            logger.warning(f"예상보다 컬럼 수가 적습니다 ({num_columns}개). 기본 컬럼명을 할당합니다.")
            # 컬럼이 적은 경우 기본 이름 할당
            basic_columns = []
            for i in range(num_columns):
                basic_columns.append(f'Column_{i}')
            df.columns = basic_columns
            logger.info(f"할당된 기본 컬럼명: {basic_columns}")
            
            # 필요한 컬럼 매핑 (최소한의 처리를 위해)
            if num_columns >= 3:
                df = df.rename(columns={
                    'Column_0': 'CAN ID',
                    'Column_1': 'DLC',
                    'Column_' + str(num_columns-1): 'Flag/Tag'
                })
                # DATA 컬럼 매핑
                for i in range(min(8, num_columns-3)):  # -3은 CAN ID, DLC, Flag/Tag를 고려
                    df = df.rename(columns={f'Column_{i+2}': f'DATA[{i}]'})
                logger.info("최소한의 컬럼 매핑을 적용했습니다.")
        
        logger.info(f"데이터 로드 완료. 형태: {df.shape}")
        return df
        
    except FileNotFoundError:
        logger.error(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")
        raise
    except Exception as e:
        logger.error(f"데이터셋 로드 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    원시 데이터 전처리 (데이터 타입 변환 및 기본 정리)
    
    Args:
        df: 원시 데이터프레임
        
    Returns:
        전처리된 데이터프레임
    """
    # 데이터 복사본 생성
    processed_df = df.copy()
    
    # 데이터 타입 변환
    # Timestamp를 float로 변환
    processed_df['Timestamp'] = processed_df['Timestamp'].astype(float)
    
    # CAN ID를 문자열로 통일 (이미 문자열인 경우 변환 없음)
    processed_df['CAN ID'] = processed_df['CAN ID'].astype(str)
    
    # DLC를 정수로 변환 (0-8 범위)
    processed_df['DLC'] = processed_df['DLC'].astype(int)
    
    # DATA 필드를 문자열로 통일
    for i in range(8):
        col = f'DATA[{i}]'
        processed_df[col] = processed_df[col].astype(str)
    
    # Flag/Tag를 문자열로 통일
    processed_df['Flag/Tag'] = processed_df['Flag/Tag'].astype(str)
    
    return processed_df


def split_data(X: np.ndarray, y: np.ndarray, val_size: float = 0.33, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터를 학습과 검증 세트로 분할 (논문의 데이터 분할 방식에 따른 2-way 분할)
    
    Args:
        X: 특성 데이터
        y: 라벨 데이터
        val_size: 전체 데이터 중 검증 세트 비율 (논문 기준 0.33 = 33%)
        random_state: 랜덤 시드
        
    Returns:
        (X_train, X_val, y_train, y_val) 튜플 - 학습 세트(67%)와 검증 세트(33%)
    """
    # 학습(67%)과 검증(33%) 세트로 직접 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


def plot_class_distribution(y_train: np.ndarray, y_val: np.ndarray) -> plt.Figure:
    """
    데이터셋의 클래스 분포 시각화 (학습 및 검증 세트)
    
    Args:
        y_train: 학습 세트 라벨
        y_val: 검증 세트 라벨
        
    Returns:
        클래스 분포 그래프
    """
    # 각 세트별 클래스 개수 계산
    train_normal = np.sum(y_train == 0)
    train_attack = np.sum(y_train == 1)
    
    val_normal = np.sum(y_val == 0)
    val_attack = np.sum(y_val == 1)
    
    # 데이터 구성
    datasets = ['Training (67%)', 'Validation (33%)']
    normal_counts = [train_normal, val_normal]
    attack_counts = [train_attack, val_attack]
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(datasets))
    
    ax.bar(index, normal_counts, bar_width, label='Normal (0)')
    ax.bar(index + bar_width, attack_counts, bar_width, label='Attack (1)')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Datasets')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    # 각 막대 위에 수치 표시
    for i, count in enumerate(normal_counts):
        ax.text(i, count, str(count), ha='center', va='bottom')
    
    for i, count in enumerate(attack_counts):
        ax.text(i + bar_width, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def print_model_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray = None) -> Dict[str, float]:
    """
    모델 평가 지표 출력 및 반환
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_prob: 예측 확률 (옵션)
        
    Returns:
        평가 지표 딕셔너리
    """
    # 혼동 행렬 계산
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 평가 지표 계산
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # 분류 리포트 출력
    print("\n분류 보고서:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
    
    # 주요 지표 출력
    print("\n주요 평가 지표:")
    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print(f"F1 점수: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    
    # 혼동 행렬 출력
    print("\n혼동 행렬:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    
    # 지표 딕셔너리 반환
    metrics = {
        'accuracy': accuracy,
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
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> plt.Figure:
    """
    ROC 곡선 시각화
    
    Args:
        y_true: 실제 라벨
        y_pred_prob: 예측 확률
        
    Returns:
        ROC 곡선 그래프
    """
    from sklearn.metrics import roc_curve, auc
    
    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> plt.Figure:
    """
    정밀도-재현율 곡선 시각화
    
    Args:
        y_true: 실제 라벨
        y_pred_prob: 예측 확률
        
    Returns:
        정밀도-재현율 곡선 그래프
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # 정밀도-재현율 곡선 계산
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True)
    
    return fig