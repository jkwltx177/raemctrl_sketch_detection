# RAEMCTRL: Sketch Detection (Archived)

## 🌟 프로젝트 개요

이 리포지토리는 인터랙티브 미디어 아트 '[RAEMCTRL](https://github.com/jkwltx177/raemctrl)' 프로젝트의 초기 컴퓨터 비전 모델 개발 과정을 담은 기록 보관소입니다.

주요 목표는 YOLO(You Only Look Once) 모델을 활용하여 HTP(집-나무-사람) 투사 검사에서 나타나는 관객의 그림을 분석하는 컴퓨터 비전 모델을 설계하고 제작하는 것이었습니다. 이 과정에서 수많은 시행착오를 겪었으며, 이 리포지토리는 그 실험과 탐색의 과정을 기록하기 위해 보존되고 있습니다.

**중요: 여기에 포함된 `sketch_detection` 관련 코드는 최종 `RAEMCTRL` 프로젝트에서는 다른 접근 방식으로 대체되었습니다. 따라서 이 리포지토리는 개발 과정의 기록물로서의 의미를 가집니다.**

## 💻 기술 스택

- **언어**: Python
- **핵심 라이브러리**: YOLO, PyTorch, OpenCV, Roboflow, Supervision

## 📝 시행착오 및 개발 과정

본 리포지토리의 코드들은 HTP 투사 검사 그림을 객체 탐지 모델로 분석하려는 초기 아이디어를 구현하며 발생한 다양한 시도들을 보여줍니다.

- **데이터 수집 및 전처리**: `ndjson` 및 `json` 형식의 초기 데이터를 YOLO 포맷으로 변환하고(`json2yolo.py`, `ndjson2yolo.py`), DeepFashion과 같은 외부 데이터셋을 전처리하는(`preprocess_deepfashion.py`) 등, 모델 학습에 적합한 데이터셋을 구축하기 위한 다양한 노력이 있었습니다.

- **모델 실험**: 초기 YOLOv8 모델부터 시작하여, ResNet 백본을 결합한 커스텀 모델(`yolo11_resnet50.yaml`), DETR(Detection Transformer) 계열의 `rf-detr.py` 등 다양한 아키텍처를 실험하며 성능 개선을 시도했습니다.

- **라벨링 및 데이터 정제**: 부정확한 바운딩 박스를 수정하고(`check_bbox.py`, `fix_label.py`), 여러 클래스를 병합하거나 변경하는(`merge_labels.py`, `change_class.py`) 등, 모델의 인식률을 높이기 위해 라벨 데이터를 지속적으로 정제하는 과정을 거쳤습니다.

- **분석 및 평가**: `analyze.py`와 같은 스크립트를 통해 모델의 성능을 분석하고, 탐지 결과를 시각화하며 반복적으로 모델을 개선했습니다.

이러한 과정들은 최종적으로는 `RAEMCTRL` 프로젝트의 방향성과는 다른 결론으로 이어졌지만, 초기 아이디어를 구체화하고 기술적 한계를 탐색하는 중요한 경험이었습니다.

## 🚀 최종 프로젝트

이 리포지토리의 결과물은 사용되지 않았으며, 최종 프로젝트는 아래 링크에서 확인하실 수 있습니다.

- **[https://github.com/jkwltx177/raemctrl](https://github.com/jkwltx177/raemctrl)**
