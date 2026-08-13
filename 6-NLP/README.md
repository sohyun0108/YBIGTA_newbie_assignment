## 가상환경 설정

아래 세 가지 방법 중 **하나만 선택**하여 환경을 설정하면 됩니다. 모두 동일한 패키지를 설치합니다.

### Option 1: venv (standard Python)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate kornli
```

### Option 3: uv

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
# or simply: uv sync
```


## Natural Language Inference

### Dataset Overview

| Column                     | Total   | Train      | Dev.  | Test  |
| -------------------------- | ------- | ---------- | ----- | ----- |
| Translated by              | -       | Machine    | Human | Human |
| \# Examples                | 950,354 | 942,854    | 2,490 | 5,010 |
| Avg. \# words (premise)    | 13.6    | 13.6       | 13.0  | 13.1  |
| Avg. \# words (hypothesis) | 7.1     | 7.2        | 6.8   | 6.8   |


### Examples

| Example                                                      | English Translation                                          | Label         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------- |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 이해하려고 노력하고 있었어요. | I was just there just trying to figure it out.<br />I was trying to understand. | Entailment    |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 나는 처음부터 그것을 잘 이해했다. | I was just there just trying to figure it out.<br />I understood it well from the beginning. | Contradiction |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 나는 돈이 어디로 갔는지 이해하려고 했어요. | I was just there just trying to figure it out.<br />I was trying to understand where the money went. | Neutral       |



### Examples

| Example                                                      | English Translation                                      | Label |
| ------------------------------------------------------------ | -------------------------------------------------------- | ----- |
| 한 남자가 음식을 먹고 있다.<br />한 남자가 뭔가를 먹고 있다. | A man is eating food.<br />A man is eating something.    | 4.2   |
| 한 비행기가 착륙하고 있다.<br />애니메이션화된 비행기 하나가 착륙하고 있다. | A plane is landing.<br />A animated airplane is landing. | 2.8   |
| 한 여성이 고기를 요리하고 있다.<br />한 남자가 말하고 있다. | A woman is cooking meat.<br />A man is speaking.      | 0.0   |