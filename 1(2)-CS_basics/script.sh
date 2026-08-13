
#!/bin/bash

# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if ! command -v conda &> /dev/null; then
    echo "[INFO] Conda가 설치되어 있지 않습니다. Miniconda 설치를 시작합니다..."
    # macOS용 Miniconda 다운로드 및 설치
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-MacOSX-x86_64.sh
    source "$HOME/miniconda/etc/profile.d/conda.sh"
else
    # 맥북 아나콘다 환경의 실제 베이스 경로를 추적하여 명확하게 불러옵니다
    CONDA_BASE_PATH=$(conda info --base)
    if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
    fi
fi

# Conda 환셩 생성 및 활성화
## TODO
# 이미 설치되어 있다면 conda 명령어 활성화 준비
if ! conda info --envs | grep -q "myenv"; then
    conda create -y -n myenv python=3.10
fi
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
# mypy 테스트를 수행해야 하므로 mypy 패키지를 설치합니다
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    # 파일 이름에서 문제 번호만 추출합니다 (예: 5_17408.py -> 17408)
    prob_num=$(echo "$file" | cut -d'_' -f2 | cut -d'.' -f1)
    
    # input과 output은 submission의 상위 폴더인 1(2)-CS_basics에 있으므로 ../ 경로를 사용합니다
    if [ -f "../input/${prob_num}_input" ]; then
        python "$file" < "../input/${prob_num}_input" > "../output/${prob_num}_output"
        echo "[INFO] 실행 완료: $file -> output/${prob_num}_output"
    else
        echo "[WARN] 입력 파일을 찾을 수 없습니다: input/${prob_num}_input"
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
# TODO
# 현재 위치가 submission 폴더이므로 상위 폴더에 로그를 저장합니다
mypy *.py > ../mypy_log.txt 2>&1
echo "[INFO] mypy 테스트 결과 저장 완료: mypy_log.txt"

# conda.yml 파일 생성
## TODO
conda env export > ../conda.yml
echo "[INFO] 가상환경 정보 저장 완료: conda.yml"

# 원래 디렉토리로 복귀
cd ..

# 가상환경 비활성화
## TODO
conda deactivate
echo "[INFO] 가상환경 비활성화 완료"