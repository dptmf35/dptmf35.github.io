---
title: "[공부] 로봇 비전을 위한 3D Reconstruction — 학습 로드맵"
excerpt: "2D 이미지에서 3D 세계를 복원하는 3D Reconstruction. 기초 기하학부터 point cloud·voxel·mesh 표현, NeRF·3DGS까지 이어지는 학습 트리를 한눈에 정리한다."
date: 2026-07-05 08:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - robot vision
  - roadmap
toc: true
toc_sticky: true
---

> 로봇이 물체를 잡고 공간을 이동하려면 먼저 **세계의 3D 형상**을 알아야 한다.
> 이 시리즈는 2D 이미지·깊이 데이터에서 3D를 복원하는 **3D Reconstruction**을 학습 트리 순서로 따라간다. 이 글은 전체 지도를 그린다.

## 1. 3D Reconstruction이란

**3D Reconstruction**은 카메라·깊이 센서로 얻은 관측에서 장면이나 물체의 **3차원 형상**을 복원하는 문제다. 로봇 비전에서 중요한 이유는 분명하다.

- **파지(grasping)**: 물체의 표면·형상을 알아야 어디를 어떻게 잡을지 정한다
- **내비게이션·충돌 회피**: 주변 공간의 기하를 알아야 안전하게 이동한다
- **Sim2Real**: 복원한 3D를 시뮬레이터에 넣어 정책을 학습하고 실제로 옮긴다

## 2. 학습 트리

이 시리즈가 따라갈 전체 구조다. 크게 **① 기초 기하학**과 **② 3D 데이터 표현**의 두 축으로 나뉜다.

![기초 기하학에서 대응·카메라를 풀고 SfM·MVS로 복원한 뒤, point cloud·voxel·mesh로 표현하고 신경망 표현으로 확장하는 3D Reconstruction 학습 트리](/assets/images/3drecon-roadmap.svg)

## 3. 두 축의 역할

### ① 기초 기하학 및 수학

3D를 복원하려면 먼저 **카메라가 세계를 어떻게 2D로 투영하는지**를 알아야 한다.

- **camera calibration**: 내부(초점거리·주점)·외부(위치·자세) 파라미터를 추정한다
- **다중 시점 기하학**: 두 시점 사이의 **epipolar geometry**로 대응을 제약하고, **특징점 매칭**으로 같은 3D 점을 찾는다
- **SfM·MVS**: two-view 블록을 수백 장으로 확장해 **모든 카메라 자세 + sparse 구조**를 풀고(SfM·bundle adjustment), 픽셀 단위 깊이로 **dense 복원**까지 만든다(MVS) — COLMAP으로 대표되는 전통 파이프라인의 본체

여기서 얻는 것은 **카메라 자세**와 **sparse~dense 3D 구조**다.

### ② 3D 데이터 표현 및 처리

복원한 3D를 **어떤 자료구조로 담느냐**가 이후 처리를 결정한다.

- **point cloud**: 가장 날것의 표현. 노이즈 제거·downsampling으로 다듬는다
- **voxel·volume 표현**: 공간을 격자로 나눠 **TSDF**로 표면을 담고, **Marching Cubes**로 mesh를 뽑는다
- **신경망 기반 표현**: **NeRF·3DGS** — 장면을 신경망/Gaussian으로 연속 표현한다

## 4. 표현 방식 비교

| 표현 | 저장 형태 | 강점 | 약점 |
| --- | --- | --- | --- |
| point cloud | 3D 점들의 집합 | 센서 출력에 가장 가까움 | 표면·연결 정보 없음 |
| voxel/TSDF | 격자 + 부호 거리 | 표면 융합·mesh화 쉬움 | 해상도↑ 시 메모리 급증 |
| mesh | 정점 + 삼각형 | 렌더링·물리 시뮬 친화 | 위상 변화에 약함 |
| NeRF | 암시적 MLP | 사실적 novel view synthesis | 렌더링 느림 |
| 3DGS | 명시적 Gaussian point cloud | 빠른 렌더링·명시적 기하 | 큰 저장 용량 |

## 정리 & 다음 글

- 3D Reconstruction은 **기하학으로 대응·자세를 풀고 → 표현으로 담는** 두 축으로 이해하면 깔끔하다.
- 표현마다 강·약점이 달라, 용도(파지·내비·Sim2Real)에 맞게 고른다.
- 다음 글: [3D Reconstruction ① 기초 기하학 — camera calibration과 다중 시점 기하학]({% post_url 2026-07-05-3d-reconstruction-geometry %})
