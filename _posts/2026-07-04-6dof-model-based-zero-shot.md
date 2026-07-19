---
title: "[공부] Model-based Zero-shot 6DoF 추정: FoundationPose & SAM-6D"
excerpt: "CAD·참조 이미지만으로 novel object의 6DoF를 추정하는 FoundationPose와 SAM-6D"
date: 2026-07-04
categories:
  - 공부
  - CV
tags:
  - 6DoF pose estimation
  - FoundationPose
  - SAM-6D
  - foundation model
  - zero-shot
toc: true
toc_sticky: true
---

> [시리즈 개요]({% post_url 2026-07-01-6dof-pose-estimation-overview %}) 의 세 번째 방법론. **물체마다 재학습하지 않고(zero-shot)** 새로운 물체를 다루는 최신 딥러닝 접근이다.

## 1. 왜 "Model-based Zero-shot"인가

과거 학습 기반(PoseCNN, PVNet, DenseFusion 등)은 **물체/데이터셋마다 별도로 학습**해야 했다. 실제 로봇 현장에서는 물체가 수시로 바뀌므로 매번 학습은 비현실적이다.

최근 파운데이션 접근은 다음을 만족한다.

- **Model-based**: 물체의 **CAD 모델**(또는 소수 참조 이미지)만 주면 됨
- **Zero-shot**: 그 물체로 **추가 학습 없이** 바로 포즈 추정
- 공통 패러다임: **render-and-compare**(가설 포즈로 렌더링한 모습과 실제 관측을 비교) 또는 **template/포인트 매칭**

## 2. FoundationPose (NVIDIA, CVPR 2024)

**하나의 모델로 6DoF 포즈 추정과 추적(tracking)을 통합**하고, **model-based**(CAD 있음)와 **model-free**(참조 영상 몇 장) 두 세팅을 모두 지원하는 것이 특징이다.

### 구조 요약
1. **Neural Object Field (model-free 대응)**: CAD가 없을 때, 소수 참조 이미지로 물체의 **암시적 3D 표현**을 학습해 임의 시점 렌더링(novel view synthesis)을 가능하게 함
2. **Pose Hypothesis + Refinement**: 여러 포즈 가설을 세우고, **render-and-compare** 방식으로 관측과 렌더링 차이를 줄이도록 반복 정밀화
3. **Hierarchical Pose Selection**: 정밀화된 가설들을 계층적으로 비교(대조 학습)해 최종 포즈 선택

- 대규모 **합성 데이터**(LLM 보조 데이터 생성 포함)로 학습해 일반화를 확보
- BOP benchmark에서 SOTA급 성능, **추적**까지 한 모델로 처리

> 입력은 보통 **RGB-D + 물체 segmentation 마스크 + (CAD 또는 참조영상)** 이다. 첫 프레임은 추정, 이후는 추적으로 이어갈 수 있어 매니퓰레이션 루프에 적합하다.

## 3. SAM-6D (CVPR 2024)

이름 그대로 **Segment Anything Model(SAM)** 을 6DoF에 접목한 zero-shot 방법이다. **두 단계**로 구성된다.

### ① Instance Segmentation Model (ISM)
- SAM(또는 FastSAM)으로 장면의 **모든 물체 마스크**를 생성
- 각 마스크를 대상 물체의 **template(렌더링)** 과 의미(semantic)·외형·기하 관점에서 매칭해, 찾는 물체 인스턴스를 골라냄

### ② Pose Estimation Model (PEM)
- 관측 point cloud와 모델 point cloud 사이의 **부분-대-부분(partial-to-partial) 점 매칭**을 학습
- **coarse → fine 2단계**로 대응을 정교화하고, 대응이 없는 점을 처리하는 **background token** 설계로 outlier/부분 관측에 대응
- 최종적으로 대응으로부터 6DoF 포즈 산출

- CAD 모델 기반으로 **novel object에 zero-shot** 동작하며, BOP 코어 데이터셋들에서 강력한 성능을 낸다

![FoundationPose는 render-and-compare, SAM-6D는 SAM segmentation + 점 매칭](/assets/images/6dof-foundationpose-vs-sam6d.svg)

## 4. 두 방법 비교

| 항목 | FoundationPose | SAM-6D |
| --- | --- | --- |
| 핵심 패러다임 | render-and-compare + 정밀화/선택 | SAM segmentation + 점 매칭 |
| 물체 사전정보 | CAD **또는** 참조영상(model-free 가능) | CAD 모델 |
| 강점 | 추정+**추적 통합**, model-free 지원 | SAM 기반 강력한 segmentation, 모듈 분리 |
| 공통 | RGB-D 입력, **zero-shot novel object**, segmentation 필요 | |

## 5. 매니퓰레이션 관점 정리

- **임의의 새로운 물체**를 학습 없이 다룰 수 있어 실제 파지에 매우 유용하다.
- 다만 **segmentation 마스크**와 (대개) **CAD 또는 참조영상**, 그리고 상당한 **연산 자원**이 필요하다.
- 초기 포즈는 이들 모델로 잡고, 이후 프레임은 추적/ICP로 정밀·가속하는 하이브리드가 실용적이다.

## 시리즈 마무리

| 상황 | 추천 접근 |
| --- | --- |
| calibration·고정 지그·마커 부착 가능 | [마커 & 기하 기반]({% post_url 2026-07-02-6dof-marker-geometry-based %}) |
| 무늬 없는 물체 + 깊이 센서 | [포인트 정합 (TEASER++)]({% post_url 2026-07-03-6dof-point-cloud-registration %}) |
| 임의의 novel object, 학습 없이 | **Model-based zero-shot (이 글)** |

→ 전체 지도: [시리즈 개요]({% post_url 2026-07-01-6dof-pose-estimation-overview %})
