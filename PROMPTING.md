# Prompting Experiment Report

## 1. Experimental Results (Accuracy Table)

| Method | 0-shot | 3-shot | 5-shot |
| :--- | :---: | :---: | :---: |
| **Direct Prompting** | 30.00% | 52.00% | 36.00% |
| **CoT Prompting** | 68.00% | 64.00% | 64.00% |
| **My Prompting** | 68.00% | 68.00% | 68.00% |

---

## 2. CoT Prompting이 Direct Prompting보다 뛰어난 이유

1. **중간 추론 연산(Computation Budget)의 확보**:
   Direct Prompting은 문제 제시 직후 정답 토큰만을 즉시 출력하도록 요구하므로, 복잡한 다단계 연산이나 수식 전개를 수행할 모델 내 작업 공간이 부족합니다. 반면 CoT Prompting은 문제를 여러 단계로 분해하여 사고 과정을 먼저 서술하게 함으로써 정답에 도달하는 데 필요한 계산 자원을 확보합니다.

2. **오류 추적 및 논리 오류 예방 (Error Mitigation)**:
   수학 문제 해결 과정에서 한 번의 직관적 도약으로 답을 구하려 할 경우 환각(Hallucination) 현상이 발생하기 쉽습니다. CoT 기법은 연산 단계를 순차적으로 풀어냄으로써 이전 스텝의 결과를 기반으로 다음 스텝을 유도해 논리적 오류 발생 확률을 유의미하게 낮춰줍니다.

---

## 3. 본인이 설계한 프롬프트(My Prompting) 기법 및 CoT 대비 우수성

### 프롬프트 설계 전략
기존 CoT의 단순 단계별 풀이에 **역할 부여** 및 **명시적 검증 과정**을 결합한 기법입니다.

- **Role Assignment**: `"Role: You are an expert mathematician and competitive math tutor."`를 부여하여 모델의 수학 도메인 특화 추론 파라미터 활성화를 유도했습니다.
- **Structured Plan Instruction**:
  1. 문제 조건 해석 및 변수 설정
  2. 논리적/단계별 연산 진행
  3. 계산 검증 및 단위/조건 교차 검증 (Double-check)
  4. 최종 답안 `\boxed{}` 표기

### CoT 대비 더 우수한 이유
1. **단순 풀이 나열 시 발생할 수 있는 연산 누적 에러 방지**:
   표준 CoT는 풀이 중간 단계에서 발생한 미세한 계산 오류가 최종 정답까지 그대로 이어지는 치명적인 단점이 있습니다. My Prompting은 연산 후 **자체 검증 단계**를 수행하도록 지시하여 중간 과정의 단순 계산 실수를 자체 수정한 후 최종 답을 생성하도록 유도합니다.
2. **출력 형식의 명확성과 환각(Hallucination) 통제**:
   명시적인 구조화 가이드라인(1, 2, 3 단계)을 제공함으로써 모델이 사족을 붙이거나 불필요한 설명을 늘어놓는 대신 핵심 계산 및 정답 도출에 concentrated 할 수 있도록 만들어 정확도를 끌어올립니다.