# HyDL-IDS 모델 결과 시각화 가이드

이 문서는 HyDL-IDS 모델의 결과를 효과적으로 시각화하는 방법에 대한 상세한 가이드라인을 제공합니다. 결과 해석과 보고서 작성을 위한 다양한 시각화 기법과 Python 코드 예시를 제공합니다.

## 목차

1. [시각화의 중요성](#1-시각화의-중요성)
2. [학습 과정 시각화](#2-학습-과정-시각화)
3. [모델 성능 시각화](#3-모델-성능-시각화)
4. [특성 중요도 시각화](#4-특성-중요도-시각화)
5. [시간적 패턴 시각화](#5-시간적-패턴-시각화)
6. [시각화 모범 사례](#6-시각화-모범-사례)
7. [보고서 통합 가이드](#7-보고서-통합-가이드)

## 1. 시각화의 중요성

시각화는 복잡한 모델 결과를 직관적으로 이해하고 효과적으로 전달하는 데 핵심적인 역할을 합니다:

- **패턴 식별**: 데이터와 결과의 패턴을 쉽게 파악할 수 있습니다.
- **직관적 이해**: 수치 데이터보다 시각적 정보가 더 빠르게 이해됩니다.
- **결과 전달**: 복잡한 모델 성능을 비전문가도 이해할 수 있게 전달합니다.
- **문제 진단**: 모델의 강점과 약점을 시각적으로 식별하여 개선점을 찾을 수 있습니다.

HyDL-IDS와 같은 이진 분류 모델의 경우, 다양한 시각화 기법을 통해 모델의 학습 과정과 성능을 효과적으로 표현할 수 있습니다.

## 2. 학습 과정 시각화

### 2.1 손실 및 정확도 곡선

학습 과정에서의 손실과 정확도 변화를 시각화하여 모델의 학습 패턴을 분석합니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 가정: history는 model.fit()에서 반환된 학습 이력 객체 또는 저장된 이력 데이터
# history_dict = history.history 또는 미리 저장된 이력 데이터

# 데이터 준비
epochs = range(1, len(history_dict['loss']) + 1)
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# 그래프 설정
plt.figure(figsize=(15, 6))

# 손실 곡선
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 마지막 에포크 정보
plt.annotate(f'Final: {loss[-1]:.4f}', 
             xy=(epochs[-1], loss[-1]), 
             xytext=(epochs[-1]-2, loss[-1]+0.1),
             arrowprops=dict(facecolor='blue', shrink=0.05))
plt.annotate(f'Final: {val_loss[-1]:.4f}', 
             xy=(epochs[-1], val_loss[-1]), 
             xytext=(epochs[-1]-2, val_loss[-1]+0.1),
             arrowprops=dict(facecolor='red', shrink=0.05))

# 정확도 곡선
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 마지막 에포크 정보
plt.annotate(f'Final: {acc[-1]:.4f}', 
             xy=(epochs[-1], acc[-1]), 
             xytext=(epochs[-1]-2, acc[-1]-0.1),
             arrowprops=dict(facecolor='blue', shrink=0.05))
plt.annotate(f'Final: {val_acc[-1]:.4f}', 
             xy=(epochs[-1], val_acc[-1]), 
             xytext=(epochs[-1]-2, val_acc[-1]-0.1),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **학습-검증 격차**: 두 곡선 간의 큰 격차는 과적합을 나타냅니다.
- **수렴 패턴**: 빠른 초기 수렴 후 안정화되는 패턴은 좋은 학습 과정을 나타냅니다.
- **불안정성**: 검증 곡선의 큰 변동은 학습률이 너무 높거나 배치 크기가 너무 작을 수 있음을 시사합니다.
- **조기 수렴**: 몇 에포크 후 쉽게 수렴하면 학습률을 높이거나 모델 복잡성을 증가시키는 것을 고려할 수 있습니다.

### 2.2 Early Stopping 시각화

Early Stopping이 발생한 지점을 시각화하여 최적 모델 선택 시점을 분석합니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt

# 가정: history는 학습 이력 데이터, best_epoch는 Early Stopping에서 선택된 최적 에포크
best_epoch = 5  # 예시값, 실제 best epoch로 대체 필요

epochs = range(1, len(history_dict['loss']) + 1)
val_acc = history_dict['val_accuracy']

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_acc, 'bo-', label='Validation Accuracy')
plt.axvline(x=best_epoch, color='r', linestyle='--', 
            label=f'Best Epoch ({best_epoch}): {val_acc[best_epoch-1]:.4f}')

# 최적 지점 강조
plt.plot(best_epoch, val_acc[best_epoch-1], 'ro', ms=10)

plt.title('Validation Accuracy with Early Stopping', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('early_stopping.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **최적 지점**: Early Stopping이 발생한 에포크는 과적합과 과소적합 사이의 최적 지점을 나타냅니다.
- **후속 패턴**: 최적 지점 이후의 정확도 감소 패턴은 과적합 발생을 확인할 수 있습니다.
- **Patience 설정**: Early Stopping 발생 패턴은 patience 파라미터의 적절성을 평가하는 데 도움이 됩니다.

## 3. 모델 성능 시각화

### 3.1 혼동 행렬 (Confusion Matrix)

혼동 행렬은 이진 분류 모델의 성능을 직관적으로 이해하는 데 가장 기본적인 시각화 도구입니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 가정: y_true는 실제 라벨, y_pred는 예측 라벨
# y_pred = (model.predict(X_test) > 0.5).astype(int)

# 혼동 행렬 계산
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# 비율 계산을 위한 행별 합계
row_sums = cm.sum(axis=1)
norm_cm = cm / row_sums[:, np.newaxis]

# 그래프 생성
plt.figure(figsize=(10, 8))
ax = plt.subplot()

# 히트맵 스타일 혼동 행렬
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

# 각 셀에 비율 추가
annot = np.empty_like(cm, dtype=str)
for i in range(2):
    for j in range(2):
        annot[i, j] = f'{cm[i, j]} ({norm_cm[i, j]:.1%})'

# 두 번째 히트맵으로 비율 표시 (투명도 설정으로 겹쳐 보이게)
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, 
            alpha=0, annot_kws={"size": 12}, ax=ax)

# 추가 정보
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title('Confusion Matrix', fontsize=16)
ax.xaxis.set_ticklabels(['Normal (0)', 'Attack (1)'], fontsize=12)
ax.yaxis.set_ticklabels(['Normal (0)', 'Attack (1)'], fontsize=12)

# 정확도, 정밀도, 재현율 정보 추가
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

info_text = (f'Accuracy: {accuracy:.4f}\n'
             f'Precision: {precision:.4f}\n'
             f'Recall: {recall:.4f}\n'
             f'F1-score: {f1:.4f}')

plt.figtext(0.02, 0.02, info_text, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **대각선 값**: 높은 TN(좌상단)과 TP(우하단)은 좋은 분류 성능을 나타냅니다.
- **오분류**: FP(우상단)는 오탐지를, FN(좌하단)은 미탐지를 나타냅니다.
- **비율 분석**: 각 클래스 내에서의 비율을 통해 클래스별 성능을 분석할 수 있습니다.
- **불균형 데이터**: 클래스별 샘플 수 차이가 클 경우, 단순 개수보다 비율이 더 중요할 수 있습니다.

### 3.2 ROC 곡선

ROC(Receiver Operating Characteristic) 곡선은 다양한 임계값에 따른 TPR(재현율)과 FPR의 관계를 보여줍니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 가정: y_true는 실제 라벨, y_pred_prob는 예측 확률
# y_pred_prob = model.predict(X_test)

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 특정 임계값에서의 점 표시 (예: 0.3, 0.5, 0.7)
threshold_points = [0.3, 0.5, 0.7]
threshold_idxs = []
for t in threshold_points:
    # 가장 가까운 임계값 찾기
    idx = (np.abs(thresholds - t)).argmin()
    threshold_idxs.append(idx)

# 그래프 생성
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.5)')

# 특정 임계값 점 표시
for i, idx in enumerate(threshold_idxs):
    plt.plot(fpr[idx], tpr[idx], 'ro', ms=8, 
             label=f'Threshold = {threshold_points[i]:.1f} (TPR={tpr[idx]:.2f}, FPR={fpr[idx]:.2f})')

# 그래프 스타일 설정
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **AUC 값**: 1에 가까울수록 좋은 성능을 나타냅니다 (1 = 완벽한 분류, 0.5 = 무작위 추측)
- **곡선 형태**: 좌상단에 가깝게 휘어질수록 더 좋은 분류 성능을 나타냅니다.
- **임계값 선택**: 곡선 상의 점들은 다양한 임계값에서의 성능을 보여줍니다. 애플리케이션 요구에 맞는 TPR과 FPR의 균형점을 찾는 데 활용할 수 있습니다.
- **무작위 선과의 비교**: 대각선(y=x)은 무작위 추측 성능을 나타내며, 곡선이 이 선에서 멀리 떨어질수록 좋은 모델입니다.

### 3.3 정밀도-재현율 곡선

정밀도-재현율 곡선은 다양한 임계값에 따른 정밀도와 재현율의 관계를 보여줍니다. 불균형 데이터셋에서 특히 유용합니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 가정: y_true는 실제 라벨, y_pred_prob는 예측 확률

# 정밀도-재현율 곡선 계산
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
avg_precision = average_precision_score(y_true, y_pred_prob)

# F1 점수가 최대가 되는 임계값 찾기
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_idx]
best_precision = precision[best_f1_idx]
best_recall = recall[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

# 그래프 생성
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, 'b-', lw=2, 
         label=f'PR Curve (AP = {avg_precision:.4f})')

# 기본 임계값(0.5) 및 최적 F1 임계값 표시
# 0.5에 가장 가까운 임계값 찾기
default_idx = (np.abs(thresholds - 0.5)).argmin()
plt.plot(recall[default_idx], precision[default_idx], 'go', ms=8, 
         label=f'Threshold = 0.5 (P={precision[default_idx]:.2f}, R={recall[default_idx]:.2f})')

# 최적 F1 점수 임계값 표시
plt.plot(best_recall, best_precision, 'ro', ms=8, 
         label=f'Best F1 Threshold = {best_threshold:.2f} (F1={best_f1:.2f})')

# 그래프 스타일 설정
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 등고선 - 같은 F1 점수를 가진 지점들
f1_levels = [0.2, 0.4, 0.6, 0.8, 0.9]
for f1_level in f1_levels:
    # F1 = 2*P*R/(P+R) => P = F1*R/(2*R-F1)
    r = np.linspace(f1_level, 1.0, 100)
    p = (f1_level * r) / (2 * r - f1_level)
    plt.plot(r, p, 'k--', alpha=0.3)
    plt.annotate(f'F1={f1_level}', xy=(r[-1], p[-1]), 
                 xytext=(r[-1]+0.02, p[-1]), alpha=0.5)

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **AP 값**: Average Precision은 곡선 아래 면적을 나타내며, 1에 가까울수록 좋은 성능입니다.
- **곡선 형태**: 우상단에 가깝게 유지될수록 더 좋은 성능을 나타냅니다.
- **임계값 선택**: 애플리케이션 요구에 따라 정밀도와 재현율의 적절한 균형을 제공하는 임계값을 선택할 수 있습니다.
- **F1 등고선**: 동일한 F1 점수를 가진 정밀도-재현율 쌍을 연결한 선으로, 균형 잡힌 성능 지점을 찾는 데 도움이 됩니다.

### 3.4 공격 유형별 성능 비교 (해당하는 경우)

다양한 공격 유형에 대한 모델의 성능을 비교 시각화합니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 가정: 다양한 공격 유형별 성능 데이터가 있다고 가정
attack_types = ['DoS', 'Fuzzy', 'Spoofing Gear', 'Spoofing RPM']
metrics = {
    'Accuracy': [0.95, 0.92, 0.88, 0.90],
    'Precision': [0.94, 0.90, 0.85, 0.87],
    'Recall': [0.96, 0.89, 0.80, 0.92],
    'F1-score': [0.95, 0.89, 0.82, 0.89]
}

# 데이터프레임 생성
df = pd.DataFrame(metrics, index=attack_types)

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 8))

# 막대 그래프 기본 설정
bar_width = 0.2
opacity = 0.8
index = np.arange(len(attack_types))

# 각 지표별 막대 그래프
for i, column in enumerate(df.columns):
    ax.bar(index + bar_width * (i - 1.5), df[column], bar_width,
           alpha=opacity, label=column)

# 그래프 스타일 설정
ax.set_xlabel('Attack Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Metrics by Attack Type', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(attack_types)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3, axis='y')

# 각 막대 위에 값 표시
for i, col in enumerate(df.columns):
    for j, value in enumerate(df[col]):
        ax.text(j + bar_width * (i - 1.5), value + 0.01, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('attack_type_performance.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **공격 유형별 차이**: 어떤 공격 유형에 모델이 더 강하고 약한지 분석할 수 있습니다.
- **균형적 성능**: 모든 공격 유형에서 고른 성능은 모델의 견고함을 나타냅니다.
- **지표 간 차이**: 특정 공격 유형에서 정밀도와 재현율의 큰 차이는 해당 유형에 대한 탐지 특성을 시사합니다.
- **개선 방향**: 성능이 낮은 공격 유형을 식별하여 모델 개선의 방향을 설정할 수 있습니다.

## 4. 특성 중요도 시각화

모델의 결정에 영향을 미치는 주요 특성을 시각화하여 모델의 작동 방식을 이해합니다.

### 4.1 특성 중요도 분석

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

# 가정: model은 학습된 HyDL-IDS 모델, X_test는 테스트 데이터
# 특성 이름 정의
feature_names = ['CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)]

# 특성 중요도 분석을 위한 접근법
# 1. 특성 제외 방법 - 한 번에 하나의 특성을 0으로 설정하고 성능 변화 측정
def feature_importance_by_zeroing(model, X_test, y_test):
    # 기준 성능
    baseline_pred = model.predict(X_test)
    baseline_loss = tf.keras.losses.binary_crossentropy(y_test, baseline_pred).numpy().mean()
    
    importance_scores = []
    
    for feature_idx in range(X_test.shape[2]):  # 특성 차원
        # 특성을 0으로 설정한 데이터 복사
        X_modified = X_test.copy()
        X_modified[:, :, feature_idx] = 0
        
        # 수정된 데이터로 예측
        modified_pred = model.predict(X_modified)
        modified_loss = tf.keras.losses.binary_crossentropy(y_test, modified_pred).numpy().mean()
        
        # 손실 증가 = 중요도
        importance = modified_loss - baseline_loss
        importance_scores.append(importance)
    
    return importance_scores

# 특성 중요도 계산
importance_scores = feature_importance_by_zeroing(model, X_test, y_test)

# 중요도를 기준으로 특성 정렬
sorted_idx = np.argsort(importance_scores)
sorted_scores = np.array(importance_scores)[sorted_idx]
sorted_names = np.array(feature_names)[sorted_idx]

# 그래프 생성
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_names)), sorted_scores, align='center')
plt.yticks(range(len(sorted_names)), sorted_names)
plt.xlabel('Importance (Loss Increase when Feature is Zeroed)', fontsize=12)
plt.title('Feature Importance Analysis', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **중요 특성 식별**: 모델의 결정에 가장 큰 영향을 미치는 특성을 식별할 수 있습니다.
- **도메인 지식 연결**: 중요 특성을 도메인 지식과 연결하여 모델 작동의 이유를 이해할 수 있습니다.
- **특성 엔지니어링 방향**: 중요도가 낮은 특성은 제거하거나 변형하여 모델을 개선할 수 있습니다.
- **공격 유형과의 관계**: 특정 특성의 중요도는 특정 공격 유형과 연관될 수 있습니다.

### 4.2 활성화 맵 시각화

CNN 레이어의 활성화를 시각화하여 모델이 CAN 메시지의 어떤 부분에 주목하는지 이해합니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

# 가정: model은 학습된 HyDL-IDS 모델, sample은 시각화할 단일 샘플

# 첫 번째 컨볼루션 레이어의 출력을 추출하는 모델 생성
conv_layer_name = 'conv1d_1'  # 실제 레이어 이름으로 대체
conv_model = Model(inputs=model.input, 
                  outputs=model.get_layer(conv_layer_name).output)

# 샘플에 대한 활성화 맵 계산
activation = conv_model.predict(np.expand_dims(sample, axis=0))[0]

# 그래프 생성
plt.figure(figsize=(15, 10))

# 원본 특성 값 시각화
plt.subplot(2, 1, 1)
plt.imshow(sample.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Feature Value')
plt.title('Original Sample Features', fontsize=14)
plt.ylabel('Feature Index', fontsize=12)
plt.yticks(range(sample.shape[1]), 
           ['CAN ID', 'DLC'] + [f'DATA[{i}]' for i in range(8)])
plt.xlabel('Time Step', fontsize=12)

# 활성화 맵 시각화
plt.subplot(2, 1, 2)
plt.imshow(activation.T, aspect='auto', cmap='hot')
plt.colorbar(label='Activation')
plt.title(f'{conv_layer_name} Activations', fontsize=14)
plt.ylabel('Filter Index', fontsize=12)
plt.xlabel('Time Step', fontsize=12)

plt.tight_layout()
plt.savefig('activation_map.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **패턴 인식**: 활성화가 강한 영역은 모델이 중요하게 여기는 패턴을 나타냅니다.
- **시간적 패턴**: 시간 단계에 따른 활성화 변화는 시간적 패턴에 대한 모델의 반응을 보여줍니다.
- **필터 역할**: 다양한 필터가 다른 패턴을 인식하는 방식을 이해할 수 있습니다.
- **정상 vs 공격**: 정상 샘플과 공격 샘플의 활성화 패턴 비교를 통해 모델이 공격을 어떻게 구분하는지 이해할 수 있습니다.

## 5. 시간적 패턴 시각화

HyDL-IDS 모델은 시간적 패턴을 활용하므로, 이를 시각화하여 공격 탐지 메커니즘을 이해합니다.

### 5.1 LSTM 상태 시각화

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

# 가정: model은 학습된 HyDL-IDS 모델, sample은 시각화할 단일 샘플

# LSTM 레이어의 출력을 추출하는 모델 생성
lstm_layer_name = 'lstm'  # 실제 레이어 이름으로 대체
lstm_model = Model(inputs=model.input, 
                  outputs=model.get_layer(lstm_layer_name).output)

# 샘플에 대한 LSTM 출력 계산
lstm_output = lstm_model.predict(np.expand_dims(sample, axis=0))[0]

# 상위 10개 상태 선택
top_k = 10
top_indices = np.argsort(np.abs(lstm_output))[-top_k:]
top_values = lstm_output[top_indices]

# 그래프 생성
plt.figure(figsize=(12, 6))
plt.bar(range(top_k), top_values, align='center')
plt.xticks(range(top_k), [f'State {i}' for i in top_indices])
plt.xlabel('LSTM State Index', fontsize=12)
plt.ylabel('State Value', fontsize=12)
plt.title('Top LSTM States for Sample', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.tight_layout()
plt.savefig('lstm_states.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **활성화된 상태**: 값이 크게 활성화된 LSTM 상태는 중요한 패턴을 포착했음을 의미합니다.
- **정상 vs 공격**: 다양한 샘플 유형에 대한 상태 차이를 비교하여 공격 시그니처를 이해할 수 있습니다.
- **시계열 특성**: 특정 LSTM 상태의 활성화는 특정 시퀀스 패턴의 존재를 나타낼 수 있습니다.

### 5.2 시퀀스 예측 확률 변화

시퀀스 내 스텝별 예측 확률 변화를 시각화하여 공격 탐지 시점을 이해합니다.

#### 코드 예시:

```python
import matplotlib.pyplot as plt
import numpy as np

# 가정: 시퀀스 데이터에 대한 스텝별 예측 확률
# 추론을 위한 함수 정의
def predict_sequence_steps(model, sequence, window_size):
    """시퀀스의 각 스텝에서 예측 확률을 계산합니다."""
    probs = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        prob = model.predict(np.expand_dims(window, axis=0))[0][0]
        probs.append(prob)
    return probs

# 예시 시퀀스 데이터
sample_sequence = X_test[0]  # 예시 시퀀스 데이터
window_size = 10  # 윈도우 크기

# 각 스텝에서의 예측 확률 계산
step_probs = predict_sequence_steps(model, sample_sequence, window_size)

# 그래프 생성
plt.figure(figsize=(12, 6))
plt.plot(range(len(step_probs)), step_probs, 'b-', lw=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')

# 공격 레이블이 있는 경우 표시
attack_steps = [10, 15, 20]  # 예시 - 공격이 발생한 스텝
for step in attack_steps:
    if step < len(step_probs):
        plt.axvspan(step, step+1, alpha=0.3, color='red', label='Actual Attack' if step == attack_steps[0] else '')

plt.xlabel('Sequence Step', fontsize=12)
plt.ylabel('Attack Probability', fontsize=12)
plt.title('Attack Probability Evolution Over Sequence Steps', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('sequence_probability.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 해석 포인트:

- **확률 변화**: 시간에 따른 예측 확률의 변화는 공격의 진행 패턴을 보여줍니다.
- **탐지 시점**: 확률이 임계값을 초과하는 시점은 공격 탐지 시점을 나타냅니다.
- **경고 속도**: 공격 시작부터 탐지까지의 지연 시간을 평가할 수 있습니다.
- **오탐지 패턴**: 확률이 잘못 상승하는 패턴은 오탐지의 원인을 이해하는 데 도움이 됩니다.

## 6. 시각화 모범 사례

효과적인 시각화를 위한 모범 사례를 따르세요:

### 6.1 일반 시각화 지침

- **일관된 색상 체계**: 동일한 의미를 가진 요소에는 일관된 색상을 사용하세요(예: 학습 = 파란색, 검증 = 빨간색).
- **적절한 해상도**: 최소 300 DPI 이상의 해상도로 저장하여 보고서에 품질 좋은 그래프를 포함할 수 있게 하세요.
- **의미 있는 제목과 레이블**: 그래프의 제목, 축 레이블, 범례를 명확하게 작성하세요.
- **그리드 및 참조선**: 필요한 경우 그리드와 참조선을 추가하여 값을 더 쉽게 읽을 수 있게 하세요.
- **적절한 크기**: 그래프의 크기를 정보 양에 맞게 조정하세요.

### 6.2 차량 보안 도메인 특화 지침

- **임계값 표시**: 모델의 실제 사용 임계값(보통 0.5)을 참조선으로 표시하세요.
- **오탐지/미탐지 강조**: 차량 보안에서 중요한 오탐지 및 미탐지를 특별히 강조하세요.
- **실시간 성능 지표**: 차량 환경에서 중요한 추론 시간, 지연 시간 등의 성능 지표를 포함하세요.
- **공격 유형 구분**: 가능한 경우 다양한 공격 유형을 시각적으로 구분하여 표시하세요.

## 7. 보고서 통합 가이드

시각화 결과를 효과적으로 보고서에 통합하는 방법:

### 7.1 이미지 삽입

마크다운 보고서에 이미지를 삽입하는 기본 문법:

```markdown
![제목](경로/파일명.png)

**그림 X.** 설명 텍스트.
```

### 7.2 이미지 그룹화

관련 이미지를 효과적으로 그룹화하는 방법:

```markdown
<div style="display: flex; justify-content: center;">
    <div style="margin-right: 10px;">
        <img src="path/to/image1.png" width="400" alt="Image 1">
        <p><strong>그림 X(a).</strong> 첫 번째 이미지 설명.</p>
    </div>
    <div>
        <img src="path/to/image2.png" width="400" alt="Image 2">
        <p><strong>그림 X(b).</strong> 두 번째 이미지 설명.</p>
    </div>
</div>
```

### 7.3 이미지 캡션 작성 가이드라인

효과적인 이미지 캡션 작성 방법:

1. **간결하고 명확하게**: 핵심 내용을 명확하게 전달하되 불필요한 세부 사항은 생략하세요.
2. **결과 해석 포함**: 단순한 설명보다 그래프가 보여주는 중요한 통찰을 포함하세요.
3. **그래프 특징 참조**: "곡선의 가파른 초기 상승은 모델이 빠르게 학습함을 보여줍니다"와 같이 특정 특징을 설명하세요.
4. **전체 이야기와 연결**: 캡션이 전체 보고서 내러티브와 일관되게 연결되도록 하세요.
5. **질문에 답하기**: "이 그래프는 모델이 DoS 공격을 다른 유형보다 더 잘 탐지하는 이유를 보여줍니다"와 같이 독자의 궁금증에 답하는 설명을 포함하세요.

### 7.4 결과 논의 연결

시각화 결과를 보고서의 논의 섹션과 효과적으로 연결하세요:

```markdown
그림 X에서 볼 수 있듯이, HyDL-IDS 모델은 DoS 공격에 대해 95%의 F1 점수를 달성했지만, Spoofing Gear 공격에 대해서는 82%로 상대적으로 낮은 성능을 보였습니다. 이러한 차이는 DoS 공격의 명확한 트래픽 패턴과 달리, Spoofing 공격이 정상 트래픽과 유사한 패턴을 보이기 때문으로 분석됩니다. 이는 앞서 그림 Y의 혼동 행렬에서도 확인되는데, Spoofing 공격에 대한 위음성(FN) 비율이 다른 공격 유형보다 높게 나타났습니다.

이러한 결과는 향후 개선 방향에 대한 중요한 시사점을 제공합니다. Spoofing 공격에 대한 탐지 성능을 향상시키기 위해, 다음과 같은 전략을 고려할 수 있습니다:

1. Spoofing 공격에 특화된 특성 추출 방법 개발
2. 클래스 가중치 조정을 통한 Spoofing 공격 탐지 강화
3. 앙상블 기법을 활용하여 공격 유형별 전문 모델 결합
```