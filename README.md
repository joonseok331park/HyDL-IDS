# HyDL-IDS

차량 해킹 데이터셋을 위한 하이브리드 딥러닝 기반 침입 탐지 시스템(Hybrid Deep Learning-based Intrusion Detection System)

## 프로젝트 개요

이 프로젝트는 차량 내부 네트워크(CAN 버스)에서 발생하는 사이버 공격을 탐지하기 위한 하이브리드 딥러닝 기반 침입 탐지 시스템을 구현합니다. 

## 주요 기능

- 차량 CAN 데이터 전처리
- 시퀀스 기반 특성 추출
- 하이브리드 딥러닝 모델 학습 및 평가
- 성능 지표 시각화

## 설치 방법

```bash
# 저장소 복제
git clone <repository_url>
cd HyDL-IDS

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

```bash
# 모델 학습 및 평가
python main.py --data_path "Car-Hacking Dataset/DoS_dataset.csv" --window_size 10 --stride 1 --epochs 10
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 