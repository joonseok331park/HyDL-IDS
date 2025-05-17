# HyDL-IDS: 차량 내부 네트워크 침입 탐지 시스템

이 프로젝트는 차량 내부 네트워크(CAN 버스) 침입 탐지를 위한 하이브리드 딥러닝 기반 침입 탐지 시스템(HyDL-IDS)을 구현합니다. CNN과 LSTM을 결합하여 CAN 트래픽의 공간적 특징과 시간적 특징을 모두 추출하는 모델입니다.

## 개요

HyDL-IDS 모델은 다음과 같은 핵심 컴포넌트로 구성됩니다:

1. 데이터 전처리 모듈: CAN ID, DLC, DATA 바이트, Flag/Tag 등의 필드를 적절히 전처리
2. CNN-LSTM 하이브리드 네트워크: 공간적 특징 추출을 위한 CNN과 시간적 특징 추출을 위한 LSTM
3. 이진 분류기: 정상 트래픽과 공격 트래픽을 분류

## 파일 구조

```
.
├── README.md                   # 프로젝트 설명
├── hydl_ids_model.py           # HyDL-IDS 모델 클래스 및 함수
├── utils.py                    # 유틸리티 함수 (데이터 로드, 평가 등)
├── main.py                     # 메인 실행 스크립트
└── requirements.txt            # 필요 패키지
```

## 설치 방법

1. 필요 패키지 설치:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 데이터 준비

Car-Hacking 데이터셋(HCRL, 고려대학교)을 CSV 형식으로 준비해야 합니다. 데이터는 다음과 같은 형식이어야 합니다:

- Timestamp (float, 초)
- CAN ID (16진수 문자열)
- DLC (int, 0-8)
- DATA[0]-DATA[7] (각 바이트에 대한 16진수 문자열)
- Flag/Tag (문자열, 정상 'R', 공격 'T')

### 모델 학습 및 평가

다음 명령으로 모델을 학습하고 평가할 수 있습니다:

```bash
python main.py --data_path your_data.csv --output_dir results --save_model
```

주요 매개변수:

- `--data_path`: 데이터셋 CSV 파일 경로 (필수)
- `--window_size`: 시퀀스 윈도우 크기 (기본값: 10)
- `--stride`: 윈도우 슬라이딩 단위 (기본값: 1)
- `--batch_size`: 학습 배치 크기 (기본값: 256)
- `--epochs`: 학습 에포크 수 (기본값: 10)
- `--learning_rate`: 학습률 (기본값: 0.001)
- `--early_stopping`: EarlyStopping 콜백 사용 여부
- `--test_size`: 테스트 세트 비율 (기본값: 0.2)
- `--val_size`: 검증 세트 비율 (기본값: 0.2)
- `--output_dir`: 결과 저장 디렉토리 (기본값: results)
- `--save_model`: 학습된 모델 저장 여부
- `--use_preprocessed`: 이미 전처리된 데이터 파일 사용 여부
- `--preprocessed_data_path`: 전처리된 데이터 파일이 저장된 디렉토리 경로 (기본값: preprocessing)

### Early Stopping 사용하기

Early Stopping은 모델이 검증 데이터에 대해 더 이상 개선되지 않을 때 학습을 조기에 중단하는 기법입니다. 이 기능을 사용하면 과적합을 방지하고 학습 시간을 단축할 수 있습니다. 

Early Stopping을 적용하려면 명령에 `--early_stopping` 플래그를 추가하세요:

```bash
python main.py --data_path your_data.csv --output_dir results --early_stopping
```

### 전처리된 파일 재사용하기

데이터 전처리는 시간이 많이 소요되는 작업입니다. 이미 전처리된 데이터 파일을 재사용하면 모델 튜닝 시 전처리 과정을 건너뛰고 학습 시간을 크게 단축할 수 있습니다.

전처리된 파일을 사용하려면 다음과 같이 `--use_preprocessed` 플래그를 추가하고, 필요에 따라 `--preprocessed_data_path`로 저장 위치를 지정하세요:

```bash
python main.py --data_path your_data.csv --output_dir results --use_preprocessed --preprocessed_data_path "preprocessing"
```

이렇게 하면 "preprocessing" 디렉토리에 저장된 X_preprocessed.npy와 y_preprocessed.npy 파일을 로드하여 전처리 과정을 건너뜁니다.

여러 데이터셋을 연속으로 처리할 때는 다음과 같이 명령을 조합할 수 있습니다:

```bash
python main.py --data_path "dataset1.csv" --output_dir "results_dataset1" --early_stopping --use_preprocessed --preprocessed_data_path "preprocessing"; python main.py --data_path "dataset2.csv" --output_dir "results_dataset2" --early_stopping --use_preprocessed --preprocessed_data_path "preprocessing"
```

### 직접 클래스 사용

Python 코드에서 직접 HyDL-IDS 클래스를 사용할 수도 있습니다:

```python
from hydl_ids_model import HyDL_IDS
import pandas as pd

# 데이터 로드
data = pd.read_csv('your_data.csv')

# HyDL-IDS 모델 초기화
hydl_ids = HyDL_IDS(window_size=10, stride=1)

# 데이터 전처리
X_sequences, y_sequences = hydl_ids.preprocess_data(data)

# 데이터 분할 (사용자 코드)
# ...

# 모델 구축 및 컴파일
input_shape = (X_train.shape[1], X_train.shape[2])
hydl_ids.build_model(input_shape)
hydl_ids.compile_model()

# 모델 학습
history = hydl_ids.train_model(
    X_train, y_train,
    X_val, y_val,
    batch_size=256,
    epochs=10,
    use_early_stopping=True  # Early Stopping 적용
)

# 모델 평가
metrics = hydl_ids.evaluate_model(X_test, y_test)
advanced_metrics = hydl_ids.compute_advanced_metrics(X_test, y_test)
```

## 결과

학습 및 평가 후 다음과 같은 결과물이 생성됩니다:

1. 학습 곡선 (손실 및 정확도)
2. 혼동 행렬
3. ROC 곡선 및 정밀도-재현율 곡선
4. 평가 지표 (정확도, 정밀도, 재현율, F1 점수, FPR, FNR)
5. 학습된 모델 파일 (H5 형식)
6. 평가 결과 마크다운 파일 (evaluation_results.md)

## 모델 아키텍처

HyDL-IDS 모델은 다음과 같은 레이어로 구성됩니다:

1. **입력층**: (윈도우 크기, 특성 수)
2. **Convo Unit I**:
   - Conv1D (필터 32개, 커널 크기 3, 활성화 함수 'relu')
   - BatchNormalization
   - MaxPooling1D (풀 크기 2)
   - Dropout (비율 0.2)
3. **Convo Unit II**:
   - Conv1D (필터 64개, 커널 크기 3, 활성화 함수 'relu')
   - BatchNormalization
   - MaxPooling1D (풀 크기 2)
   - Dropout (비율 0.2)
4. **LSTM 레이어**: 128 유닛, 활성화 함수 'tanh'
5. **Flatten 레이어**
6. **Dense 레이어 1**: 128 유닛
7. **Dropout 레이어**: 비율 0.2
8. **출력 레이어**: 1 유닛, 활성화 함수 'sigmoid'

## 라이센스

이 프로젝트는 MIT 라이센스에 따라 배포됩니다.