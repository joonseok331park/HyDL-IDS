# HyDL-IDS 데이터 전처리

이 문서는 HyDL-IDS 모델에서 사용되는 데이터 전처리 방식에 대해 설명합니다.

## 논문의 전처리 방식

HyDL-IDS 논문에서는 다음과 같은 전처리 방식을 제안합니다:

1. **CAN ID 처리**: 16진수 CAN ID를 정수로 인코딩
2. **DLC 처리**: DLC 값의 로그 변환 및 정규화 - `ln(val_i+1)/ln(9)` 수식 적용
3. **DATA 필드 처리**: 16진수 데이터 바이트 변환 및 정규화 - `ln(val_i+1)/ln(257)` 수식 적용
   - 존재하지 않는 바이트는 256으로 처리
4. **Flag/Tag 처리**: 레이블 인코딩 ('R'은 0, 'T'는 1)
5. **시퀀스 데이터 생성**: 윈도우 슬라이딩 방식으로 시계열 데이터 생성

## 전처리 클래스 사용법

`CANDataPreprocessor` 클래스를 사용하여 논문의 전처리 방식을 그대로 구현할 수 있습니다:

```python
from data_preprocessing import CANDataPreprocessor

# 전처리기 초기화 (윈도우 크기: 10, 스트라이드: 1)
preprocessor = CANDataPreprocessor(window_size=10, stride=1)

# 데이터 전처리 및 시퀀스 생성
X_sequences, y_sequences = preprocessor.fit_transform(data)

# 전처리된 데이터 형태 확인
print(f"X_sequences 형태: {X_sequences.shape}")  # [시퀀스 수, 윈도우 크기, 특성 수]
print(f"y_sequences 형태: {y_sequences.shape}")  # [시퀀스 수]
```

## 전처리 통계 정보

`CANDataPreprocessor`는 전처리 과정에서 다양한 통계 정보를 수집하며, 이를 `stats` 속성을 통해 확인할 수 있습니다:

```python
# CAN ID 통계
print(f"고유 CAN ID 수: {preprocessor.stats['can_id']['unique_count']}")

# 라벨 통계
print(f"정상 샘플: {preprocessor.stats['labels']['normal_count']}")
print(f"공격 샘플: {preprocessor.stats['labels']['attack_count']}")
print(f"정상 비율: {preprocessor.stats['labels']['normal_percentage']:.2f}%")
print(f"공격 비율: {preprocessor.stats['labels']['attack_percentage']:.2f}%")
```

## 모델 실행 방법

논문의 전처리 방식을 그대로 사용하여 모델을 학습하려면 다음 명령을 실행하세요:

```bash
python run_model.py --data_path [데이터셋 경로] --window_size 10 --stride 1 --batch_size 256 --epochs 10 --save_model
```

또는 main.py를 직접 실행할 수도 있습니다:

```bash
python main.py --data_path [데이터셋 경로] --window_size 10 --stride 1 --batch_size 256 --epochs 10 --save_model
```

## 전처리 결과 저장

전처리된 데이터는 지정된 출력 디렉토리의 'preprocessing' 폴더에 자동으로 저장됩니다:
- `X_preprocessed.npy`: 전처리된 특성 데이터
- `y_preprocessed.npy`: 전처리된 라벨 데이터

## 참고 사항

- 전처리 과정은 현재 실행 환경 및 데이터에 따라 시간이 오래 걸릴 수 있습니다.
- 데이터셋에 결측값이 있는 경우 적절한 대체값이 자동으로 적용됩니다.
- 커스텀 전처리 설정을 적용하려면 `CANDataPreprocessor` 클래스의 매개변수를 조정하세요. 