"""
HyDL-IDS 모델 학습 및 평가 스크립트

이 스크립트는 전처리된 데이터를 사용하여 HyDL-IDS 모델을 학습하고 평가합니다.
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import json

from model_architecture import HyDLIDSModel, create_class_weights, get_model_summary


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    명령줄 인수 파싱
    """
    parser = argparse.ArgumentParser(description='HyDL-IDS 모델 학습 및 평가')
    
    # 데이터 관련 인수
    parser.add_argument('--data_dir', type=str, required=True,
                        help='전처리된 데이터가 저장된 디렉토리')
    
    # 모델 아키텍처 관련 인수
    parser.add_argument('--conv1_filters', type=int, default=32,
                       help='첫 번째 컨볼루션 레이어의 필터 수 (기본값: 32)')
    
    parser.add_argument('--conv2_filters', type=int, default=64,
                       help='두 번째 컨볼루션 레이어의 필터 수 (기본값: 64)')
    
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='컨볼루션 커널 크기 (기본값: 3)')
    
    parser.add_argument('--lstm_units', type=int, default=128,
                       help='LSTM 유닛 수 (기본값: 128)')
    
    parser.add_argument('--dense_units', type=int, default=128,
                       help='Dense 레이어 유닛 수 (기본값: 128)')
    
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='드롭아웃 비율 (기본값: 0.2)')
    
    # 학습 관련 인수
    parser.add_argument('--batch_size', type=int, default=256,
                       help='배치 크기 (기본값: 256)')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='최대 에포크 수 (기본값: 10)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='학습률 (기본값: 0.001)')
    
    parser.add_argument('--patience', type=int, default=3,
                       help='EarlyStopping 인내심 (기본값: 3)')
    
    parser.add_argument('--min_delta', type=float, default=0.01,
                       help='EarlyStopping 최소 변화량 (기본값: 0.01)')
    
    parser.add_argument('--monitor', type=str, default='val_accuracy',
                       help='모니터링할 지표 (기본값: val_accuracy)')
    
    parser.add_argument('--use_class_weights', action='store_true',
                       help='클래스 가중치 사용 여부')
    
    parser.add_argument('--reduce_lr', action='store_true',
                       help='학습률 감소 콜백 사용 여부')
    
    # 출력 관련 인수
    parser.add_argument('--output_dir', type=str, default='results',
                       help='결과 저장 디렉토리 (기본값: results)')
    
    parser.add_argument('--model_name', type=str, default='hydl_ids',
                       help='모델 이름 (기본값: hydl_ids)')
    
    parser.add_argument('--save_model', action='store_true',
                       help='학습된 모델 저장 여부')
    
    # GPU 관련 인수
    parser.add_argument('--gpu', type=int, default=0,
                       help='사용할 GPU 번호 (기본값: 0, -1: CPU)')
    
    parser.add_argument('--gpu_memory_limit', type=int, default=None,
                       help='GPU 메모리 제한 (MB)')
    
    return parser.parse_args()


def setup_gpu(gpu_id, memory_limit=None):
    """
    GPU 설정
    
    Args:
        gpu_id: 사용할 GPU 번호 (-1: CPU)
        memory_limit: GPU 메모리 제한 (MB)
    """
    if gpu_id < 0:
        # CPU만 사용
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("CPU만 사용하도록 설정되었습니다.")
        return
    
    # GPU 메모리 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        logger.warning("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
        return
    
    try:
        # 특정 GPU 선택
        if gpu_id < len(gpus):
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            logger.info(f"GPU {gpu_id}를 사용합니다.")
            
            # 메모리 제한 설정
            if memory_limit is not None:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_id],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"GPU 메모리를 {memory_limit}MB로 제한합니다.")
            else:
                # 메모리 증가 설정
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                logger.info("GPU 메모리 증가 모드를 활성화했습니다.")
        else:
            logger.warning(f"GPU {gpu_id}가 존재하지 않습니다. GPU 0을 대신 사용합니다.")
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        logger.error(f"GPU 설정 중 오류 발생: {e}")


def main():
    """
    메인 실행 함수
    """
    # 인수 파싱
    args = parse_arguments()
    
    # GPU 설정
    setup_gpu(args.gpu, args.gpu_memory_limit)
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output_dir, f'{args.model_name}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    # 로그 저장 디렉토리
    log_dir = os.path.join(result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 체크포인트 저장 디렉토리
    checkpoint_dir = os.path.join(result_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 그래프 저장 디렉토리
    graph_dir = os.path.join(result_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    
    # 데이터 로드
    logger.info(f"전처리된 데이터 로드 중: {args.data_dir}")
    
    try:
        X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
        
        logger.info(f"데이터 로드 완료:")
        logger.info(f"학습 데이터: {X_train.shape}, {y_train.shape}")
        logger.info(f"검증 데이터: {X_val.shape}, {y_val.shape}")
        logger.info(f"테스트 데이터: {X_test.shape}, {y_test.shape}")
    except FileNotFoundError as e:
        logger.error(f"데이터 로드 실패: {e}")
        logger.error(f"디렉토리 '{args.data_dir}'에 'X_train.npy', 'y_train.npy' 등의 파일이 있는지 확인하세요.")
        return
    
    # 입력 형태 가져오기
    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"입력 형태: {input_shape}")
    
    # 모델 초기화 및 구축
    logger.info("모델 초기화 및 구축 중...")
    model = HyDLIDSModel(input_shape=input_shape, model_name=args.model_name)
    
    model.build_model(
        conv1_filters=args.conv1_filters,
        conv2_filters=args.conv2_filters,
        kernel_size=args.kernel_size,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate
    )
    
    # 모델 요약 정보 저장
    model_summary = get_model_summary(model.model)
    with open(os.path.join(result_dir, 'model_summary.txt'), 'w') as f:
        f.write(model_summary)
    
    # 모델 컴파일
    logger.info(f"모델 컴파일 중... (학습률: {args.learning_rate})")
    model.compile_model(
        learning_rate=args.learning_rate,
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae', 'mse']
    )
    
    # 콜백 설정
    logger.info("콜백 설정 중...")
    callbacks = model.get_callbacks(
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
        logdir=log_dir,
        checkpoint_path=os.path.join(checkpoint_dir, f'{args.model_name}_best.h5'),
        reduce_lr=args.reduce_lr
    )
    
    # 클래스 가중치 계산 (선택사항)
    class_weights = None
    if args.use_class_weights:
        logger.info("클래스 가중치 계산 중...")
        class_weights = create_class_weights(y_train)
    
    # 하이퍼파라미터 저장
    hyperparams = vars(args)
    with open(os.path.join(result_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # 모델 학습
    logger.info(f"모델 학습 시작 (배치 크기: {args.batch_size}, 최대 에포크: {args.epochs})")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # 학습 곡선 시각화
    logger.info("학습 곡선 시각화 중...")
    loss_fig, acc_fig = model.plot_learning_curves()
    loss_fig.savefig(os.path.join(graph_dir, 'learning_curve_loss.png'))
    acc_fig.savefig(os.path.join(graph_dir, 'learning_curve_accuracy.png'))
    
    # 모델 평가
    logger.info("모델 평가 중...")
    metrics = model.evaluate(X_test, y_test, batch_size=args.batch_size)
    
    # 예측 및 고급 지표 계산
    logger.info("예측 및 고급 지표 계산 중...")
    y_pred_prob = model.predict(X_test, batch_size=args.batch_size)
    
    # 분류 지표 계산
    classification_metrics = model.compute_classification_metrics(y_test, y_pred_prob)
    
    # 결과 시각화
    logger.info("결과 시각화 중...")
    
    # 혼동 행렬 시각화
    cm_fig = model.plot_confusion_matrix(y_test, y_pred_prob)
    cm_fig.savefig(os.path.join(graph_dir, 'confusion_matrix.png'))
    
    # ROC 곡선 시각화
    roc_fig = model.plot_roc_curve(y_test, y_pred_prob)
    roc_fig.savefig(os.path.join(graph_dir, 'roc_curve.png'))
    
    # 정밀도-재현율 곡선 시각화
    pr_fig = model.plot_precision_recall_curve(y_test, y_pred_prob)
    pr_fig.savefig(os.path.join(graph_dir, 'precision_recall_curve.png'))
    
    # 메트릭 저장
    logger.info("메트릭 저장 중...")
    model.save_metrics(os.path.join(result_dir, 'metrics.json'))
    
    # 평가 지표 요약 저장
    with open(os.path.join(result_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("===== HyDL-IDS 모델 평가 요약 =====\n\n")
        
        f.write("모델 아키텍처:\n")
        f.write(f"- 입력 형태: {input_shape}\n")
        f.write(f"- 첫 번째 컨볼루션 필터: {args.conv1_filters}\n")
        f.write(f"- 두 번째 컨볼루션 필터: {args.conv2_filters}\n")
        f.write(f"- 커널 크기: {args.kernel_size}\n")
        f.write(f"- LSTM 유닛: {args.lstm_units}\n")
        f.write(f"- Dense 유닛: {args.dense_units}\n")
        f.write(f"- 드롭아웃 비율: {args.dropout_rate}\n\n")
        
        f.write("학습 파라미터:\n")
        f.write(f"- 배치 크기: {args.batch_size}\n")
        f.write(f"- 최대 에포크: {args.epochs}\n")
        f.write(f"- 학습률: {args.learning_rate}\n")
        f.write(f"- 클래스 가중치 사용: {args.use_class_weights}\n\n")
        
        f.write("평가 지표:\n")
        f.write(f"- 손실: {metrics['loss']:.6f}\n")
        f.write(f"- 정확도: {metrics['accuracy']:.6f}\n")
        f.write(f"- MAE: {metrics['mae']:.6f}\n")
        f.write(f"- MSE: {metrics['mse']:.6f}\n\n")
        
        f.write("분류 지표:\n")
        f.write(f"- 정밀도: {classification_metrics['precision']:.6f}\n")
        f.write(f"- 재현율: {classification_metrics['recall']:.6f}\n")
        f.write(f"- F1 점수: {classification_metrics['f1_score']:.6f}\n")
        f.write(f"- FPR: {classification_metrics['fpr']:.6f}\n")
        f.write(f"- FNR: {classification_metrics['fnr']:.6f}\n\n")
        
        f.write("혼동 행렬:\n")
        f.write(f"- 진양성(TP): {classification_metrics['true_positives']}\n")
        f.write(f"- 진음성(TN): {classification_metrics['true_negatives']}\n")
        f.write(f"- 위양성(FP): {classification_metrics['false_positives']}\n")
        f.write(f"- 위음성(FN): {classification_metrics['false_negatives']}\n")
    
    # 최종 모델 저장 (선택사항)
    if args.save_model:
        logger.info("최종 모델 저장 중...")
        model.save_model(os.path.join(result_dir, f'{args.model_name}_final.h5'))
    
    logger.info(f"모든 결과가 저장되었습니다: {result_dir}")
    logger.info("완료")


if __name__ == "__main__":
    main()