---
title: "[공부] 3D Reconstruction ① 기초 기하학 — camera calibration과 다중 시점 기하학"
excerpt: "3D 복원의 출발점. 카메라 내부·외부 파라미터를 추정하는 calibration과, 두 시점을 잇는 epipolar geometry·특징점 매칭을 정리한다."
date: 2026-07-05 09:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - camera calibration
  - epipolar geometry
  - feature matching
toc: true
toc_sticky: true
---

> [학습 로드맵]({% post_url 2026-07-05-3d-reconstruction-roadmap %})의 첫 축. 3D를 복원하려면 먼저 **카메라가 세계를 2D로 어떻게 투영하는지**를 수식으로 알아야 한다.

## 1. 카메라 투영 모델

핀홀 카메라는 3D 점 $$X=(X,Y,Z)$$ 를 이미지 점 $$(u,v)$$ 로 투영한다.

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K\,[R \mid t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

- $$K$$: **내부 파라미터(intrinsics)** — 초점거리 $$f_x, f_y$$, 주점 $$(c_x, c_y)$$
- $$[R \mid t]$$: **외부 파라미터(extrinsics)** — 세계 좌표계에 대한 카메라의 회전·평행이동
- $$s$$: 스케일(깊이에 따른 동차 좌표 정규화)

## 2. camera calibration

calibration은 위 $$K$$ 와 $$[R \mid t]$$ 를 **추정**하는 과정이다.

- **내부 파라미터**: 렌즈·센서 고유값. 체커보드처럼 **알려진 패턴**을 여러 각도에서 찍어, 관측된 코너와 투영 모델을 맞춰 추정한다(Zhang's method)
- **왜곡 계수(distortion)**: 렌즈의 방사·접선 왜곡을 함께 보정한다
- **외부 파라미터**: 세계 좌표계 기준 카메라의 자세. 대상마다 달라진다

내부 파라미터를 한 번 구해 두면, 이후 **픽셀 ↔ 광선(ray)** 변환이 정확해져 모든 3D 복원의 기반이 된다.

## 3. 다중 시점 기하학 — epipolar geometry

한 장으로는 깊이가 모호하다($$s$$ 를 모른다). **두 시점**을 쓰면 삼각측량으로 깊이를 복원할 수 있다. 이때 두 이미지의 대응을 제약하는 것이 **epipolar geometry**다.

![두 카메라 중심과 3D 점이 이루는 평면이 각 이미지에 epipolar 선을 만들고, 대응점은 그 선 위에 놓인다](/assets/images/3drecon-epipolar.svg)

- 좌 이미지의 점 $$x$$ 에 대응하는 우 이미지의 점 $$x'$$ 는 **epipolar 선** 위에만 존재한다
- 이 제약은 **기본 행렬(Fundamental matrix)** $$F$$ 로 표현된다.

$$
x'^\top F x = 0
$$

- 내부 파라미터를 알면 $$F$$ 대신 **본질 행렬(Essential matrix)** $$E = K'^\top F K$$ 를 쓰고, $$E$$ 를 분해해 두 시점 사이의 **상대 자세** $$(R, t)$$ 를 얻는다

epipolar 제약은 대응 탐색을 2D 전체에서 **1D 선 위**로 줄여 준다 — 매칭이 빠르고 견고해진다.

## 4. 특징점 매칭

대응 $$x \leftrightarrow x'$$ 를 실제로 찾는 단계다.

1. **특징점 검출**: 코너처럼 구별되는 점 검출 (Harris, FAST 등)
2. **서술자(descriptor)**: 각 점 주변을 벡터로 요약 (SIFT, ORB 등)
3. **매칭**: 서술자 거리로 두 이미지의 점을 짝짓는다
4. **outlier 제거**: **RANSAC** + epipolar 제약으로 잘못된 매칭을 걸러낸다

이렇게 얻은 대응과 상대 자세로 **삼각측량**하면 3D 점을 복원한다 — 이것이 SfM(Structure from Motion)·스테레오의 핵심 골격이다.

## 정리 & 다음 글

| 단계 | 얻는 것 |
| --- | --- |
| calibration | 내부 $$K$$, 왜곡, 외부 $$[R \mid t]$$ |
| epipolar geometry | 대응 제약($$F$$/$$E$$), 상대 자세 |
| 특징점 매칭 | 시점 간 대응 → 삼각측량 |

- 기하학은 **픽셀을 3D 광선으로 바꾸고, 시점 간 대응을 제약**하는 도구다.
- 여기서 복원한 3D 점들이 다음 글의 **point cloud**가 된다.
- 다음 글: [3D Reconstruction ② point cloud — 노이즈 제거와 downsampling]({% post_url 2026-07-05-3d-reconstruction-point-cloud %})
