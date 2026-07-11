---
title: "[공부] 매니퓰레이터 조작을 위한 6DoF 포즈 추정 — 개요와 방법론 분류"
excerpt: "로봇 매니퓰레이션에서 6DoF 포즈 추정이 왜 필요한지, 그리고 마커·포인트 정합·모델 기반 세 갈래로 방법론을 정리한 시리즈 개요."
date: 2026-07-01
categories:
  - 공부
  - CV
tags:
  - 6DoF pose estimation
  - manipulation
  - robot vision
  - overview
toc: true
toc_sticky: true
---

> 매니퓰레이터(로봇 팔)로 물체를 집고 조작하려면 "그 물체가 **어디에, 어떤 자세로** 놓여 있는가"를 알아야 한다.
> 이 시리즈는 6DoF 포즈 추정 방법을 **세 갈래**로 나눠 정리한다. 이 글은 전체 지도다.

## 1. 6DoF 포즈란?

물체의 위치와 자세를 카메라(또는 로봇) 좌표계 기준으로 표현한 것이 **6 자유도(6 Degrees of Freedom)** 포즈다.

- **위치(Translation)** 3자유도: $$t = (x, y, z)$$
- **자세(Rotation)** 3자유도: 회전행렬 $$R \in SO(3)$$ (또는 quaternion/오일러각)

둘을 합치면 강체 변환(rigid transform) $$T \in SE(3)$$ 로 표현된다.

$$
T = \begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix}, \quad R \in SO(3),\; t \in \mathbb{R}^3
$$

즉 6DoF 포즈 추정은 카메라가 본 물체의 $$T_{\text{cam}}^{\text{obj}}$$ 를 구하는 문제다.

![6DoF 포즈 = 위치 3자유도 + 자세 3자유도](/assets/images/6dof-overview.svg)

## 2. 매니퓰레이션에서 왜 필요한가

로봇이 물체를 집는(grasp) 전체 파이프라인은 대략 이렇게 흐른다.

```
인식(6DoF pose) → 파지 계획(grasp planning) → 모션 계획(motion planning) → 실행
```

- 물체의 6DoF 포즈를 알아야 **어느 지점을 어떤 방향으로 잡을지**(grasp pose) 계산할 수 있다.
- 카메라 좌표계의 포즈를 **로봇 base 좌표계**로 옮기려면 **hand-eye calibration**(카메라↔end-effector 변환)이 필요하다. 이 calibration 자체에도 마커 기반 포즈 추정이 쓰인다.
- 조립·삽입(peg-in-hole)처럼 정밀도가 중요한 작업일수록 포즈 오차가 곧 실패로 이어진다.

## 3. 문제의 어려운 점

| 도전 과제 | 설명 |
| --- | --- |
| Occlusion / Clutter | 물체가 가려지거나 어질러진 환경 |
| Texture-less | 무늬 없는 금속·플라스틱 표면 → 특징점 매칭 어려움 |
| Novel objects | 학습에 없던 새로운 물체 |
| Symmetry | 대칭 물체는 자세가 모호(여러 정답) |
| 실시간성 | 매니퓰레이션 루프에 들어갈 속도 필요 |

## 4. 방법론 분류 (이 시리즈 구성)

이 시리즈는 6DoF 추정을 접근 방식에 따라 세 편으로 나눈다.

### ① 마커 & 기하 기반 (전통적)
AprilTag·ArUco 같은 **인공 마커**를 붙이거나, 물체의 알려진 기하/특징점을 이용해 **PnP**로 포즈를 푸는 고전적 방법.
→ [마커 & 기하 기반 6DoF 추정]({% post_url 2026-07-02-6dof-marker-geometry-based %})

### ② 기하학적 포인트 정합
관측한 **point cloud**를 물체 모델에 정합(registration)해 변환을 구하는 방법. ICP, FPFH+RANSAC, 그리고 outlier에 강인한 TEASER++.
→ [point cloud 정합 기반 6DoF 추정]({% post_url 2026-07-03-6dof-point-cloud-registration %})

### ③ Model-based Zero-shot (딥러닝/파운데이션)
물체 CAD 모델이나 소수 참조 이미지만 있으면 **학습 없이(zero-shot)** 새로운 물체의 포즈를 추정하는 최신 파운데이션 모델. FoundationPose, SAM-6D.
→ [Model-based Zero-shot 6DoF 추정]({% post_url 2026-07-04-6dof-model-based-zero-shot %})

## 5. 한눈에 비교

| 구분 | 대표 방법 | 주 입력 | 물체 사전정보 | 강점 | 약점 |
| --- | --- | --- | --- | --- | --- |
| 마커·기하 | AprilTag, PnP, LINEMOD | RGB | 마커 or CAD/template | 빠르고 정확, 구현 쉬움 | 마커 부착 필요/무늬 없으면 취약 |
| 포인트 정합 | ICP, TEASER++ | Depth(point cloud) | 물체 point cloud/CAD | 무늬 불필요, 기하만으로 | 초기값·대응점 품질 의존 |
| Model-based | FoundationPose, SAM-6D | RGB-D | CAD or 참조영상 | novel object zero-shot, 고정확 | 연산량 큼, segmentation 필요 |

## 정리

- 6DoF 포즈는 $$SE(3)$$ 강체 변환이고, 매니퓰레이션의 출발점이다.
- **환경을 통제할 수 있으면** 마커 기반이 가장 실용적이다(특히 calibration).
- **임의의 물체**를 다뤄야 하면 포인트 정합 또는 모델 기반으로 간다.
- 다음 글부터 방법별로 구체적으로 정리한다.
- 다음 글: [마커 & 기하 기반 6DoF 추정]({% post_url 2026-07-02-6dof-marker-geometry-based %})
