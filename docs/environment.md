# Experiment 2 Environment

## 1. Environment Policy

- Experiment 2는 프로젝트 루트의 전용 `.venv`를 사용한다.
- Experiment 1에서 사용한 Conda environment와 분리하여 두 실험의 dependency가 서로 영향을 주지 않게 한다.
- Windows local development environment와 향후 Linux Docker environment는 별도로 검증하고 관리한다.
- 원본 dataset과 생성 데이터는 Python environment에 포함하지 않는다.
- 현재 설치 상태의 기록인 `environment_snapshot.txt`와 의도된 dependency specification은 서로 다른 목적으로 관리한다.

## 2. Current Local Environment

2026-09-18에 프로젝트의 `.venv`에서 직접 확인한 결과다.

| Item | Value |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.14 |
| Python executable | `C:\Users\AISW_203_113\Documents\GitHub\cnn-lstm-drowsiness-detection\.venv\Scripts\python.exe` |
| pip | 26.2.1 |
| NumPy | 2.5.3 |
| Pandas | 3.0.6 |
| OpenCV import | 5.0.0 |
| OpenCV installed package | `opencv-python==5.0.0.93` |
| dlib import | 20.0.1 |
| dlib installed package | `dlib-bin==20.0.1` |
| PyYAML | 6.0.3 |
| tqdm | 4.70.1 |
| pytest | 9.1.1 |
| PyTorch | NOT INSTALLED |
| torchvision | NOT INSTALLED |
| scikit-learn | NOT INSTALLED |
| GPU | NVIDIA GeForce RTX 3080 |
| NVIDIA driver | 595.95 |
| Driver-reported CUDA | 13.2 |
| PyTorch CUDA runtime | NOT AVAILABLE — PyTorch가 설치되지 않음 |

전체 package 목록과 확인 명령의 결과는 [environment_snapshot.txt](environment_snapshot.txt)에 보존한다.

## 3. Why Python 3.12

현재 local environment는 Python 3.12.14로 고정한다. 기존 Experiment 1 검증도 Python 3.12 계열에서 수행됐으며, 현재 STEP 1에서 Dlib, OpenCV 등 computer vision dependency의 실제 import와 asset load가 확인됐기 때문이다. 이는 현재 검증된 기준을 유지하기 위한 선택이며 Python 3.13이 절대 지원되지 않는다는 의미는 아니다. Python 계열을 변경하려면 동일한 source validation과 이후 preprocessing audit을 다시 실행해 호환성을 확인한다.

## 4. Dlib Installation

초기에는 다음 source package 설치를 시도했다.

```powershell
python -m pip install dlib
```

Windows와 Python 3.12 환경에서 source distribution을 처리하던 중 Windows `cp949` decoding 과정의 `UnicodeDecodeError`가 발생했다. 따라서 현재 Windows local development environment에는 prebuilt wheel을 제공하는 다음 package를 사용했다.

```powershell
python -m pip install dlib-bin==20.0.1
```

설치 package 이름은 `dlib-bin`이지만 Python 코드에서는 다음과 같이 사용한다.

```python
import dlib
```

현재 확인된 package와 import 버전은 모두 20.0.1이며 STEP 1에서 Dlib68 predictor load가 성공했다. 이 선택은 Windows local environment에만 적용한다. 향후 Linux Docker image에서는 build 방식, 공식 `dlib` package 사용 가능성, image 크기 및 runtime 호환성을 별도로 검증한 후 dependency 정책을 결정한다.

## 5. Environment Activation

Windows CMD에서 다음 명령으로 활성화한다.

```bat
.venv\Scripts\activate
python --version
where python
```

`where python`의 첫 번째 결과가 다음 프로젝트 전용 executable인지 확인한다.

```text
C:\Users\AISW_203_113\Documents\GitHub\cnn-lstm-drowsiness-detection\.venv\Scripts\python.exe
```

PowerShell에서는 다음과 같이 활성화할 수 있다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 6. Dependency Policy

현재 repository에는 `requirements.txt`와 `pyproject.toml`이 없다. 현 단계에서는 transitive dependency까지 포함하는 `pip freeze` 결과를 공식 dependency specification으로 사용하지 않는다. 특히 학습용 PyTorch와 CUDA 조합이 아직 결정되지 않았기 때문이다.

### 현재 preprocessing 및 검증 environment에 설치됨

- `numpy==2.5.3`
- `pandas==3.0.6`
- `opencv-python==5.0.0.93` (`cv2.__version__ == 5.0.0`)
- `dlib-bin==20.0.1` (`dlib.__version__ == 20.0.1`)
- `PyYAML==6.0.3`
- `tqdm==4.70.1`
- `pytest==9.1.1`

### 향후 training 단계 후보이며 현재 설치되지 않음

- PyTorch
- torchvision
- scikit-learn

`docs/environment_snapshot.txt`는 특정 시점의 실제 local environment 기록이다. 향후 추가할 `requirements.txt` 또는 `pyproject.toml`은 재현을 위해 프로젝트가 의도적으로 지원하는 direct dependency와 version 정책을 정의해야 한다.

## 7. OpenCV / YuNet Warning

STEP 1 실행 중 OpenCV에서 다음 warning이 관찰됐다.

```text
Targets are not supported by the new graph engine for now
```

공식 STEP 1 summary의 warning 배열은 비어 있고, YuNet model load 상태는 `OK`, 최종 판정은 `READY_FOR_PREPROCESSING_AUDIT`이다. 따라서 이 warning은 현재 asset load validation의 성공 여부에는 영향을 주지 않았다. warning을 숨기거나 OpenCV 설정을 임의로 변경하지 않으며, STEP 2의 실제 detection audit에서 runtime 동작과 latency를 다시 확인한다.

## 8. Docker Plan

현재 단계에서는 Docker image를 만들지 않는다. 다음 순서로 local pipeline을 먼저 안정화한 뒤 container 환경을 구성한다.

1. Local `.venv`에서 STEP 2 preprocessing audit 수행
2. preprocessing 및 model pipeline 안정화
3. direct dependency와 version 정책 확정
4. Dockerfile 작성
5. GPU, NVIDIA driver 및 CUDA/PyTorch 조합 검증
6. container 내부에서 STEP 1과 STEP 2 validation 재실행

Local Windows에서 사용한 `dlib-bin`을 Linux Docker에도 그대로 적용한다고 미리 결정하지 않는다.
