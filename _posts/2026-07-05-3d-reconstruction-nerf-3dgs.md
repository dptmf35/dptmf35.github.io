---
title: "[공부] 3D Reconstruction ④ 신경망 기반 표현 — NeRF vs 3DGS"
excerpt: "장면을 신경망·Gaussian으로 연속 표현하는 최신 접근. volume rendering의 NeRF와 rasterization의 3DGS를 인터랙티브 위젯으로 비교한다."
date: 2026-07-05 12:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - NeRF
  - 3D Gaussian Splatting
  - novel view synthesis
toc: true
toc_sticky: true
---

> [학습 로드맵]({% post_url 2026-07-05-3d-reconstruction-roadmap %})의 마지막 표현. [voxel·mesh]({% post_url 2026-07-05-3d-reconstruction-voxel-tsdf %})가 이산 격자였다면, NeRF·3DGS는 장면을 **연속 함수**로 담아 사실적인 신규 시점을 합성한다.

## 1. 신경망 기반 표현의 등장

앞선 표현들은 장면을 **이산(점·voxel·삼각형)**으로 담았다. 신경망 기반 표현은 다른 길을 택한다.

- **연속 표현**: 공간의 임의 위치·방향에서 색·밀도를 **함수로 질의**한다
- **novel view synthesis**: 학습에 없던 카메라 위치의 이미지를 사실적으로 렌더한다
- 입력은 여전히 **여러 장의 이미지 + 카메라 자세**([calibration·SfM]({% post_url 2026-07-05-3d-reconstruction-geometry %}))다

대표 주자가 **NeRF**와 **3DGS**다. 둘의 차이는 아래 위젯에서 렌더링 과정을 직접 보며 이해하는 편이 빠르다.

{% include nerf-3dgs-compare.html %}

*위젯의 **렌더링 모드**를 NeRF ↔ 3DGS로 바꿔 보면, 샘플링 방식·속도·연산 부하가 어떻게 달라지는지 보인다.*

## 2. NeRF — volume rendering

**NeRF(Neural Radiance Fields)**는 장면을 하나의 **MLP**로 표현한다. 위치 $$(x,y,z)$$ 와 시선 방향 $$(\theta, \phi)$$ 를 넣으면 색 $$c$$ 와 밀도 $$\sigma$$ 를 출력한다.

$$
F_\Theta : (x, y, z, \theta, \phi) \;\mapsto\; (c, \sigma)
$$

렌더링은 **volume rendering(ray marching)**으로 한다.

1. 카메라에서 각 픽셀로 **광선**을 쏜다
2. 광선을 따라 여러 점을 **샘플링**해 MLP로 $$(c, \sigma)$$ 를 질의한다
3. 밀도로 가중 적분해 픽셀 색을 만든다

$$
C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(t)\,c(t)\,dt, \quad T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(s)\,ds\right)
$$

- **암시적(implicit)** 표현: 장면이 MLP 가중치 안에 들어 있다
- 픽셀마다 수십~수백 점을 MLP로 질의하므로 **렌더링이 느리다**(연산 부하 큼)

## 3. 3DGS — 명시적 Gaussian + rasterization

**3DGS(3D Gaussian Splatting)**는 장면을 수백만 개의 **3D Gaussian(타원체)** 집합으로 표현한다. 각 Gaussian은 위치·공분산(모양)·색·불투명도를 가진다.

$$
G(x) = \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
$$

렌더링은 광선 적분 대신 **rasterization**로 한다.

1. 각 Gaussian을 이미지 평면에 **투영(splat)**해 2D 타원으로 만든다
2. 깊이 순으로 정렬해 **알파 블렌딩**으로 합성한다

- **명시적(explicit)** 표현: Gaussian point cloud이 그대로 기하 구조다
- GPU rasterization라 **실시간 렌더링**이 가능하다(속도 빠름, 연산 부하 낮음)

## 4. 비교

| 항목 | NeRF | 3DGS |
| --- | --- | --- |
| 데이터 구조 | 암시적 MLP | 명시적 Gaussian point cloud |
| 렌더링 원리 | volume rendering(ray marching) | 타원체 투사 + rasterization |
| 렌더링 속도 | 느림 | 빠름(실시간) |
| 연산 부하 | 큼 | 낮음 |
| Sim2Real 연동 | 어려움(부피 데이터) | 용이(명시적 기하) |

**로봇 관점**에서 3DGS의 **명시적 기하**는 큰 장점이다. Gaussian point cloud을 mesh·충돌 형상으로 옮기기 쉬워, 복원한 장면을 시뮬레이터에 넣는 **Sim2Real**에 유리하다.

## 정리 & 시리즈 마무리

- NeRF·3DGS는 장면을 **연속 표현**으로 담아 사실적 신규 시점을 합성한다.
- **NeRF = 암시적·volume rendering(느리지만 사실적)**, **3DGS = 명시적·rasterization(빠르고 Sim2Real 친화적)**.
- 이로써 [기초 기하학]({% post_url 2026-07-05-3d-reconstruction-geometry %}) → [point cloud]({% post_url 2026-07-05-3d-reconstruction-point-cloud %}) → [voxel·mesh]({% post_url 2026-07-05-3d-reconstruction-voxel-tsdf %}) → 신경망 표현까지 3D Reconstruction의 큰 지도를 한 바퀴 돌았다.
- 처음으로: [3D Reconstruction 학습 로드맵]({% post_url 2026-07-05-3d-reconstruction-roadmap %})
