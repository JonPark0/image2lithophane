# Image2Lithophane

본 프로젝트는 PyQt5와 VTK를 사용하여 이미지를 3D 리소페인(lithophane) 모델로 변환하는 데스크톱 애플리케이션입니다. 이미지를 불러오고, 다양한 매개변수를 조정하며, 생성된 3D 모델을 미리보고, 3D 프린팅을 위한 STL 파일로 내보내는 기능이 있습니다.

## 주요 기능

*   **이미지 불러오기**: 다양한 이미지 포맷 지원 (PNG, JPG, JPEG, BMP, TIFF)
*   **이미지 전처리**:
    *   이미지를 그레이스케일로 자동 변환
    *   자르기, 패딩, 늘리기 방식으로 이미지 크기 조정
*   **리소페인 매개변수 커스터마이징**: 기본 높이, 최대 높이, 최소 두께 조정
*   **실시간 3D 미리보기**: VTK를 사용한 생성된 리소페인 모델 실시간 시각화
*   **STL 내보내기**: 대부분의 3D 프린터와 호환되는 STL 파일로 3D 모델 저장
*   **User Friendly GUI 환경**: 직관적인 GUI와 진행률 표시
*   **자동 디바운싱**: 매개변수 조정 시 부드러운 미리보기 업데이트

## 시스템 요구사항

*   **운영체제**: Windows 10/11 (추천), macOS, Linux
*   **Python**: 3.7 이상
*   **메모리**: 최소 4GB RAM (8GB 추천)
*   **디스크 공간**: 500MB 이상

## 빠른 시작 (Windows 사용자 추천)

### 방법 1: 배치 스크립트 사용 (가장 간단)

1.  **설치 (처음 한 번만 실행)**
    *   `setup.bat` 파일을 더블클릭하여 실행
    *   자동으로 가상환경이 생성되고 필요한 패키지가 설치됩니다

2.  **실행**
    *   `run.bat` 파일을 더블클릭하여 프로그램 실행

### 방법 2: PowerShell 스크립트 사용

1.  **설치**
    *   `setup.ps1` 파일을 마우스 우클릭 → "PowerShell로 실행"

2.  **실행**
    *   `run.ps1` 파일을 마우스 우클릭 → "PowerShell로 실행"

> **참고**: PowerShell 스크립트 실행 시 오류가 발생하면:
>
> PowerShell을 관리자 권한으로 실행하고 다음 명령을 입력하세요:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 상세 설치 가이드 (수동 설치)

모든 플랫폼(Windows, macOS, Linux)에서 사용 가능한 수동 설치 방법입니다.

### 1. 리포지토리 복제 (해당되는 경우)

Git이 설치되어 있지 않은 경우 [여기를 클릭하여](https://github.com/JonPark0/image2lithophane/archive/refs/heads/main.zip) 압축 파일을 바로 다운로드 할 수 있습니다.

```bash
git clone <repository_url>
cd image2lithophane
```

### 2. Python 가상환경 생성

```bash
python -m venv venv
```

### 3. 가상환경 활성화

*   **Windows (CMD):**
    ```cmd
    venv\Scripts\activate
    ```

*   **Windows (PowerShell):**
    ```powershell
    venv\Scripts\Activate.ps1
    ```

*   **macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

설치되는 패키지:
*   `PyQt5` - GUI 프레임워크
*   `numpy` - 수치 계산
*   `Pillow` - 이미지 처리
*   `numpy-stl` - STL 파일 생성
*   `vtk` - 3D 시각화

---

## 프로그램 실행

### Windows 빠른 실행

*   `run.bat` 더블클릭 (Command Prompt)
*   또는 `run.ps1` 더블클릭 (PowerShell)

### 수동 실행 (모든 플랫폼)

```bash
# 가상환경 활성화 (아직 활성화하지 않은 경우)
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 프로그램 실행
python viewer.py
```

---

## 사용 방법

### 1. 이미지 불러오기
*   "Load Image" 버튼을 클릭하여 이미지 파일 선택
*   지원 포맷: PNG, JPG, JPEG, BMP, TIFF

### 2. 크기 조정 옵션 설정
*   **Width (가로)**: 리소페인의 가로 크기 (픽셀)
*   **Height (세로)**: 리소페인의 세로 크기 (픽셀)
*   **Resize Method (크기 조정 방법)** 선택:
    *   **Crop (Centered)**: 이미지를 중앙 기준으로 잘라내기
    *   **Pad (Centered)**: 이미지를 중앙에 배치하고 패딩 추가
    *   **Stretch**: 이미지를 늘려서 맞추기

### 3. 리소페인 매개변수 설정
*   **Base Height (기본 높이, mm)**: 리소페인의 최소 두께 (기본값: 2.0mm)
*   **Max Height (최대 높이, mm)**: 기본 높이에서 추가되는 최대 릴리프 높이 (기본값: 4.0mm)
*   **Min Thickness (최소 두께, mm)**: 가장 얇은 부분의 절대 최소 두께 (기본값: 0.5mm)

### 4. 리소페인으로 변환
*   "Convert to Lithophane" 버튼 클릭
*   변환 진행 중 진행률 대화상자가 표시됩니다
*   우측 패널에서 3D 모델 미리보기가 표시됩니다

### 5. STL 파일 내보내기
*   "Export STL" 버튼을 클릭하여 생성된 3D 모델을 원하는 위치에 저장

### 미리보기 기능
*   크기 조정 옵션을 변경하면 자동으로 처리된 이미지 미리보기가 업데이트됩니다
*   디바운싱 기능으로 부드러운 사용자 경험 제공 (500ms 지연)

---

## 프로젝트 구조

```
image2lithophane/
├── viewer.py              # 메인 애플리케이션 (리팩토링 완료)
├── requirements.txt       # Python 의존성 목록
├── README.md              # 프로젝트 문서 (이 파일)
├── .gitignore             # Git 무시 파일 목록
├── setup.bat              # Windows CMD 자동 설치 스크립트
├── setup.ps1              # PowerShell 자동 설치 스크립트
├── run.bat                # Windows CMD 실행 스크립트
├── run.ps1                # PowerShell 실행 스크립트
└── venv/                  # Python 가상환경 (자동 생성)
```

---

## 기술 스택

### 프론트엔드 (GUI)
*   **PyQt5**: 데스크톱 애플리케이션 프레임워크
*   **VTK (Visualization Toolkit)**: 3D 모델 렌더링 및 상호작용

### 백엔드 (이미지 처리 및 메시 생성)
*   **Pillow (PIL)**: 이미지 로딩, 변환, 크기 조정
*   **NumPy**: 고성능 벡터화 연산
*   **numpy-stl**: STL 파일 생성 및 메시 관리

### 코드 품질
*   **Type Hints**: 전체 코드베이스에 타입 주석 적용
*   **Logging**: Python logging 모듈을 사용한 체계적인 로깅
*   **Error Handling**: 견고한 예외 처리 및 사용자 친화적 오류 메시지

---

## FAQ

### Python 설치 오류
**증상**: "Python is not installed or not in PATH" 오류

**해결 방법**:
1.  [Python 공식 웹사이트](https://www.python.org/downloads/)에서 Python 설치
2.  설치 시 **"Add Python to PATH"** 옵션 반드시 체크
3.  설치 완료 후 명령 프롬프트를 재시작

### 가상환경 생성 실패
**증상**: "Failed to create virtual environment" 오류

**해결 방법**:
*   디스크 공간 확인 (최소 500MB 필요)
*   폴더 쓰기 권한 확인
*   바이러스 백신 소프트웨어 일시 비활성화 후 재시도

### PowerShell 실행 정책 오류
**증상**: 스크립트 실행 시 "이 시스템에서 스크립트를 실행할 수 없습니다" 오류

**해결 방법**:
```powershell
# PowerShell을 관리자 권한으로 실행한 후:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 패키지 설치 실패
**증상**: pip install 중 오류 발생

**해결 방법**:
1.  인터넷 연결 확인
2.  방화벽/프록시 설정 확인
3.  pip 업데이트 후 재시도:
    ```bash
    venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

### VTK 관련 오류
**증상**: 3D 미리보기 창이 표시되지 않거나 오류 발생

**해결 방법**:
*   그래픽 드라이버 업데이트
*   VTK 재설치:
    ```bash
    pip uninstall vtk
    pip install vtk
    ```

### 애플리케이션 실행 시 크래시
**해결 방법**:
1.  로그 확인 (콘솔 출력 확인)
2.  가상환경 재생성:
    ```bash
    rmdir /s /q venv
    setup.bat
    ```