"""
HyDL-IDS 모델 테스트 및 평가 메인 스크립트

이 스크립트는 Car-Hacking 데이터셋에 HyDL-IDS 모델을 적용하여 학습, 평가 및 결과 시각화를 수행합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import logging
from datetime import datetime
import ntpath

from hydl_ids_model import HyDL_IDS
from data_preprocessing import CANDataPreprocessor
from utils import (
    load_can_data, split_data, plot_class_distribution,
    print_model_evaluation_metrics, plot_roc_curve, plot_precision_recall_curve
)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    명령줄 인수 파싱
    """
    parser = argparse.ArgumentParser(description='HyDL-IDS 모델 학습 및 평가')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Car-Hacking 데이터셋 CSV 파일 경로')
    
    parser.add_argument('--window_size', type=int, default=10,
                        help='시퀀스 생성을 위한 윈도우 크기 (기본값: 10)')
    
    parser.add_argument('--stride', type=int, default=1,
                        help='윈도우 슬라이딩 단위 (기본값: 1)')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='학습 배치 크기 (기본값: 256)')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 에포크 수 (기본값: 10)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='학습률 (기본값: 0.001)')
    
    parser.add_argument('--early_stopping', action='store_true',
                        help='EarlyStopping 콜백 사용 여부')
    
    parser.add_argument('--val_size', type=float, default=0.33,
                        help='검증 세트 비율 (기본값: 0.33, 논문 기준)')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='결과 저장 디렉토리 (기본값: results)')
    
    parser.add_argument('--save_model', action='store_true',
                        help='학습된 모델 저장 여부')
    
    parser.add_argument('--preprocess_dir', type=str, default='.',
                        help='전처리 결과 저장 디렉토리 (기본값: 현재 디렉토리)')
    
    parser.add_argument('--verbose', type=int, default=2,
                        help='진행 출력 레벨 (0: 출력 없음, 1: 진행 막대, 2: 에포크당 한 줄) (기본값: 2)')
    
    # 추가된 인자: 전처리된 파일 사용 여부
    parser.add_argument('--use_preprocessed', action='store_true',
                        help='이미 전처리된 데이터 파일 사용 여부')
    
    # 추가된 인자: 전처리된 파일 경로
    parser.add_argument('--preprocessed_data_path', type=str, default='.',
                        help='전처리된 데이터 파일이 저장된 디렉토리 경로 (기본값: 현재 디렉토리)')
    
    return parser.parse_args()


def get_dataset_name(data_path):
    """
    데이터 경로에서 파일 이름(확장자 제외)을 추출합니다.
    
    Args:
        data_path: 데이터 파일 경로
        
    Returns:
        파일 이름(확장자 제외)
    """
    # 경로에서 파일 이름 추출
    file_name = ntpath.basename(data_path)
    # 확장자 제거
    dataset_name = os.path.splitext(file_name)[0]
    return dataset_name


def main():
    """
    메인 실행 함수
    """
    # 인수 파싱
    args = parse_arguments()
    
    # 데이터셋 이름 추출
    dataset_name = get_dataset_name(args.data_path)
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    # 전처리 파일 이름 생성 (데이터셋 이름 포함)
    x_preprocessed_file = f"X_preprocessed_{dataset_name}.npy"
    y_preprocessed_file = f"y_preprocessed_{dataset_name}.npy"
    
    # 전처리 결과 디렉토리 설정
    preprocess_dir = args.preprocess_dir
    os.makedirs(preprocess_dir, exist_ok=True)
    
    # 전처리된 데이터를 사용할지 또는 새로 전처리를 수행할지 결정
    if args.use_preprocessed:
        # 전처리된 데이터 파일 로드
        logger.info(f"전처리된 데이터 파일 로드 중: {args.preprocessed_data_path}")
        X_sequences_path = os.path.join(args.preprocessed_data_path, x_preprocessed_file)
        y_sequences_path = os.path.join(args.preprocessed_data_path, y_preprocessed_file)
        
        if os.path.exists(X_sequences_path) and os.path.exists(y_sequences_path):
            X_sequences = np.load(X_sequences_path)
            y_sequences = np.load(y_sequences_path)
            logger.info(f"전처리된 데이터 로드 완료: X={X_sequences.shape}, y={y_sequences.shape}")
            
            # 전처리 통계를 로드할 수 없기 때문에 기본 통계 정보만 생성
            preprocessor = CANDataPreprocessor(window_size=args.window_size, stride=args.stride)
            preprocessor.stats = {
                'labels': {
                    'normal_count': int((y_sequences == 0).sum()),
                    'attack_count': int((y_sequences == 1).sum()),
                    'normal_percentage': (y_sequences == 0).mean() * 100,
                    'attack_percentage': (y_sequences == 1).mean() * 100
                }
            }
        else:
            logger.warning(f"전처리된 데이터 파일을 찾을 수 없습니다: {args.preprocessed_data_path}")
            logger.warning("새로 전처리를 수행합니다.")
            args.use_preprocessed = False
    
    # 전처리된 데이터가 없거나 사용하지 않는 경우 새로 전처리 수행
    if not args.use_preprocessed:
        # 데이터 로드
        logger.info(f"데이터 로드 중: {args.data_path}")
        raw_data = load_can_data(args.data_path)
        logger.info(f"로드된 데이터 형태: {raw_data.shape}")
        
        # 논문 방식의 전처리 수행 (CANDataPreprocessor 사용)
        logger.info(f"논문 방식의 전처리 수행 중 (윈도우 크기: {args.window_size}, 스트라이드: {args.stride})...")
        preprocessor = CANDataPreprocessor(window_size=args.window_size, stride=args.stride)
        X_sequences, y_sequences = preprocessor.fit_transform(raw_data)
        logger.info(f"전처리된 데이터 형태: X={X_sequences.shape}, y={y_sequences.shape}")
        
        # 전처리 결과 저장 (메인 함수 위치에)
        np.save(os.path.join(preprocess_dir, x_preprocessed_file), X_sequences)
        np.save(os.path.join(preprocess_dir, y_preprocessed_file), y_sequences)
        logger.info(f"전처리 결과가 저장되었습니다: {os.path.join(preprocess_dir, x_preprocessed_file)}")
    
    # 전처리 통계 출력
    logger.info("전처리 통계:")
    for key, value in preprocessor.stats.items():
        if key == 'can_id' and 'unique_count' in value:
            logger.info(f"  고유 CAN ID 수: {value['unique_count']}")
        elif key == 'labels':
            logger.info(f"  정상 샘플: {value['normal_count']} ({value['normal_percentage']:.2f}%)")
            logger.info(f"  공격 샘플: {value['attack_count']} ({value['attack_percentage']:.2f}%)")
    
    # 데이터 분할 (학습 67%, 검증 33% - 논문 기준)
    logger.info(f"데이터 분할 중 (검증: {args.val_size})")
    X_train, X_val, y_train, y_val = split_data(
        X_sequences, y_sequences, 
        val_size=args.val_size
    )
    logger.info(f"학습 세트: {X_train.shape} (67%), 검증 세트: {X_val.shape} (33%)")
    
    # 클래스 분포 시각화
    logger.info("클래스 분포 시각화 중...")
    class_dist_fig = plot_class_distribution(y_train, y_val)
    class_dist_fig.savefig(os.path.join(result_dir, 'class_distribution.png'))
    
    # HyDL-IDS 모델 인스턴스 생성
    logger.info(f"HyDL-IDS 모델 초기화")
    hydl_ids = HyDL_IDS()
    
    # 모델 구축
    logger.info("모델 구축 중...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    hydl_ids.build_model(input_shape)
    
    # 모델 컴파일
    logger.info(f"모델 컴파일 중 (학습률: {args.learning_rate})")
    hydl_ids.compile_model(learning_rate=args.learning_rate)
    
    # 모델 학습
    logger.info(f"모델 학습 시작 (배치 크기: {args.batch_size}, 에포크: {args.epochs})")
    history = hydl_ids.train_model(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_early_stopping=args.early_stopping,
        verbose=args.verbose
    )
    
    # 학습 곡선 시각화
    logger.info("학습 곡선 시각화 중...")
    loss_fig, acc_fig = hydl_ids.plot_learning_curves()
    loss_fig.savefig(os.path.join(result_dir, 'learning_curve_loss.png'))
    acc_fig.savefig(os.path.join(result_dir, 'learning_curve_accuracy.png'))
    
    # 모델 평가 - 검증 세트로 평가
    logger.info("검증 세트 평가 중...")
    metrics = hydl_ids.evaluate_model(X_val, y_val, verbose=args.verbose)
    
    # 고급 평가 지표 계산
    logger.info("고급 평가 지표 계산 중...")
    advanced_metrics = hydl_ids.compute_advanced_metrics(X_val, y_val, verbose=0)
    
    # 예측 수행
    logger.info("검증 세트 예측 중...")
    y_pred_prob = hydl_ids.model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 평가 지표 출력 및 저장
    metrics_combined = print_model_evaluation_metrics(y_val, y_pred, y_pred_prob)
    
    # 지표 저장 (텍스트 파일)
    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write("HyDL-IDS 모델 평가 지표\n")
        f.write(f"날짜: {timestamp}\n\n")
        
        f.write("모델 파라미터:\n")
        f.write(f"- 윈도우 크기: {args.window_size}\n")
        f.write(f"- 스트라이드: {args.stride}\n")
        f.write(f"- 배치 크기: {args.batch_size}\n")
        f.write(f"- 학습률: {args.learning_rate}\n")
        f.write(f"- 에포크 수: {args.epochs}\n")
        f.write(f"- EarlyStopping: {args.early_stopping}\n\n")
        f.write(f"- 전처리된 데이터 사용: {args.use_preprocessed}\n")
        if args.use_preprocessed:
            f.write(f"- 전처리된 데이터 경로: {args.preprocessed_data_path}\n\n")
        
        f.write("평가 지표:\n")
        for metric, value in metrics_combined.items():
            f.write(f"- {metric}: {value:.6f}\n")
    
    # 지표 저장 (마크다운 파일)
    with open(os.path.join(result_dir, 'evaluation_results.md'), 'w') as f:
        # 마크다운 헤더와 기본 정보
        f.write("# HyDL-IDS 모델 평가 결과\n\n")
        f.write(f"**날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**데이터셋**: `{os.path.basename(args.data_path)}`\n")
        if args.use_preprocessed:
            f.write(f"**전처리된 데이터 사용**: 예 (`{args.preprocessed_data_path}`)\n\n")
        else:
            f.write(f"**전처리된 데이터 사용**: 아니오\n\n")
        
        # 모델 파라미터 정보 테이블
        f.write("## 모델 파라미터\n\n")
        f.write("| 파라미터 | 값 |\n")
        f.write("|---------|----|\n")
        f.write(f"| 윈도우 크기 | {args.window_size} |\n")
        f.write(f"| 스트라이드 | {args.stride} |\n")
        f.write(f"| 배치 크기 | {args.batch_size} |\n")
        f.write(f"| 학습률 | {args.learning_rate} |\n") 
        f.write(f"| 에포크 수 | {args.epochs} |\n")
        f.write(f"| EarlyStopping | {args.early_stopping} |\n\n")
        
        # 데이터 통계 정보 테이블
        f.write("## 데이터 통계\n\n")
        f.write("| 통계 | 값 |\n")
        f.write("|------|----|\n")
        for key, value in preprocessor.stats.items():
            if key == 'can_id' and 'unique_count' in value:
                f.write(f"| 고유 CAN ID 수 | {value['unique_count']} |\n")
            elif key == 'labels':
                f.write(f"| 정상 샘플 | {value['normal_count']} ({value['normal_percentage']:.2f}%) |\n")
                f.write(f"| 공격 샘플 | {value['attack_count']} ({value['attack_percentage']:.2f}%) |\n")
        f.write(f"| 총 시퀀스 수 | {len(y_sequences)} |\n")
        f.write(f"| 학습 세트 크기 | {len(y_train)} |\n")
        f.write(f"| 검증 세트 크기 | {len(y_val)} |\n\n")
        
        # 평가 지표 테이블
        f.write("## 평가 지표\n\n")
        f.write("| 지표 | 값 |\n")
        f.write("|------|----|\n")
        for metric, value in metrics_combined.items():
            f.write(f"| {metric} | {value:.6f} |\n")
        
        # 결과 시각화 링크
        f.write("\n## 결과 시각화\n\n")
        f.write("### 학습 곡선\n\n")
        f.write("![손실 학습 곡선](learning_curve_loss.png)\n\n")
        f.write("![정확도 학습 곡선](learning_curve_accuracy.png)\n\n")
        
        f.write("### 성능 평가\n\n")
        f.write("![혼동 행렬](confusion_matrix.png)\n\n")
        f.write("![ROC 곡선](roc_curve.png)\n\n")
        f.write("![정밀도-재현율 곡선](precision_recall_curve.png)\n\n")
        
        f.write("### 클래스 분포\n\n")
        f.write("![클래스 분포](class_distribution.png)\n\n")
    
    # 혼동 행렬 시각화
    cm_fig = hydl_ids.plot_confusion_matrix(y_val, y_pred)
    cm_fig.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    
    # ROC 곡선 시각화
    roc_fig = plot_roc_curve(y_val, y_pred_prob)
    roc_fig.savefig(os.path.join(result_dir, 'roc_curve.png'))
    
    # 정밀도-재현율 곡선 시각화
    pr_fig = plot_precision_recall_curve(y_val, y_pred_prob)
    pr_fig.savefig(os.path.join(result_dir, 'precision_recall_curve.png'))
    
    # 모델 저장
    if args.save_model:
        model_path = os.path.join(result_dir, 'hydl_ids_model.h5')
        logger.info(f"모델 저장 중: {model_path}")
        hydl_ids.save_model(model_path)
    
    logger.info(f"모든 결과가 저장되었습니다: {result_dir}")
    logger.info("완료")


if __name__ == "__main__":
    main()