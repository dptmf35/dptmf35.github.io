---
title: "[공부] 3D Reconstruction ③ point cloud — 노이즈 제거와 downsampling"
excerpt: "가장 날것의 3D 표현인 point cloud. outlier를 걷어내는 통계·반경 기반 필터와, 균일·경량화를 위한 voxel downsampling을 정리한다."
date: 2026-07-05 10:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - point cloud
  - outlier removal
  - downsampling
toc: true
toc_sticky: true
---

> [학습 로드맵]({% post_url 2026-07-05-3d-reconstruction-roadmap %})의 두 번째 축, 3D 데이터 표현의 시작. 센서가 뱉는 가장 원초적인 형태인 **point cloud**를 다듬는 기법을 본다.

## 1. point cloud란

**point cloud**는 3D 점 $$\{p_i = (x_i, y_i, z_i)\}$$ 의 집합이다. 깊이 카메라·LiDAR·스테레오·[triangulation]({% post_url 2026-07-05-3d-reconstruction-geometry %})·[SfM·MVS]({% post_url 2026-07-05-3d-reconstruction-sfm-mvs %})의 직접 출력이라 **센서에 가장 가깝다**.

- 장점: 표현이 단순하고 센서 출력 그대로
- 단점: **표면·연결 정보가 없다**. 점 사이가 비어 있고, 노이즈·밀도 편차가 크다

그래서 이후 처리(정합·mesh화·인식) 전에 **정제(filtering)**가 필요하다.

## 2. 노이즈 제거 (outlier 제거)

센서 오차·반사로 생긴 **동떨어진 점(outlier)**을 걷어낸다. 두 가지 대표 방식이 있다.

- **Statistical Outlier Removal**: 각 점에서 **k개 최근접 이웃까지의 평균 거리**를 구하고, 전체 분포에서 평균 + $$\alpha\sigma$$ 를 넘는 점을 제거한다. 주변보다 유독 먼 점이 걸린다
- **Radius Outlier Removal**: 반경 $$r$$ 안의 이웃 개수가 임계값보다 적으면 제거한다. 고립된 점을 직접 걸러낸다

![원본의 밀집 point cloud과 outlier를 제거하고, voxel 격자로 downsampling해 균일·경량화하는 과정](/assets/images/3drecon-pointcloud-filter.svg)

## 3. downsampling

점이 너무 많으면 느리고 밀도가 불균일하다. **downsampling**으로 점 수를 줄이면서 형상을 보존한다.

- **Voxel Grid downsampling**: 공간을 한 변 $$v$$ 의 격자로 나누고, **각 voxel 안의 점들을 대표점 하나(무게중심)**로 대체한다. 밀도가 균일해지고 점 수가 크게 준다
- **Uniform downsampling**: 일정 간격으로 점을 골라낸다(단순·빠름)
- **파라미터 트레이드오프**: voxel 크기 $$v$$ 가 크면 가볍지만 디테일이 뭉개지고, 작으면 디테일은 살지만 무겁다

## 4. 전형적 전처리 파이프라인

```
원본 point cloud
  → outlier 제거 (통계적 / 반경)
  → voxel downsample (균일·경량화)
  → (필요 시) 법선 추정
  → 정합 · mesh화 · 인식 단계로
```

법선(normal) 추정을 함께 해 두면 이후 **surface reconstruction**과 point-to-plane 정합에 쓰인다.

## 정리 & 다음 글

| 단계 | 목적 | 대표 기법 |
| --- | --- | --- |
| outlier 제거 | 노이즈 점 제거 | 통계적 / 반경 기반 |
| downsampling | 균일화·경량화 | voxel 그리드 |
| 법선 추정 | 표면 정보 확보 | 이웃 평면 적합 |

- point cloud는 **날것이라 강력하지만, 다듬어야 쓸 수 있다.**
- 다음 단계는 점을 **격자(voxel)로 융합**해 표면을 담는 것이다.
- 다음 글: [3D Reconstruction ④ voxel·volume 표현 — TSDF와 Marching Cubes]({% post_url 2026-07-05-3d-reconstruction-voxel-tsdf %})
