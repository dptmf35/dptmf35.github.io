---
title: "[공부] 3D Reconstruction ④ voxel·volume 표현 — TSDF와 Marching Cubes"
excerpt: "부호 거리로 표면을 담는 TSDF와 삼각형 mesh를 뽑아내는 Marching Cubes"
date: 2026-07-05 11:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - TSDF
  - voxel
  - marching cubes
  - mesh
toc: true
toc_sticky: true
---

> [학습 로드맵]({% post_url 2026-07-05-3d-reconstruction-roadmap %})의 두 번째 데이터 표현. 여러 프레임의 [point cloud]({% post_url 2026-07-05-3d-reconstruction-point-cloud %})를 하나의 **격자(volume)**로 융합해 매끈한 표면을 만드는 방법이다.

## 1. voxel·volume 표현

**voxel**은 2D 픽셀의 3D 버전으로, 공간을 정육면체 격자로 나눈 단위다. 각 voxel에 값을 저장해 **공간 전체를 표현**한다.

- point cloud와 달리 **공간을 빈틈없이 채워** 표면 안/밖을 구분할 수 있다
- 여러 시점의 관측을 같은 격자에 **누적·평균**해 노이즈를 줄인다(센서 융합)
- 한계: 해상도를 높이면 메모리가 $$O(n^3)$$ 으로 급증한다

## 2. TSDF 원리

**TSDF(Truncated Signed Distance Function)**는 각 voxel에 **가장 가까운 표면까지의 부호 거리**를 저장한다.

$$
\text{TSDF}(v) = \max\!\left(-1,\; \min\!\left(1,\; \frac{d(v)}{\tau}\right)\right)
$$

- $$d(v)$$: voxel 중심에서 표면까지의 부호 거리 — **음수면 표면 안쪽, 양수면 바깥쪽**
- **표면은 값이 0을 지나는 곳**(zero-crossing)에 있다
- $$\tau$$: 절단(truncation) 폭. 표면 근처 좁은 띠만 의미 있게 다루고, 먼 값은 $$\pm 1$$ 로 잘라 계산·저장을 아낀다(그래서 "Truncated")

![voxel마다 표면까지의 부호 거리를 저장하고(− 내부 / + 외부), 값이 0을 지나는 곳이 표면이 되며 Marching Cubes로 삼각형 mesh를 추출한다](/assets/images/3drecon-tsdf-marching.svg)

깊이 프레임이 들어올 때마다 각 voxel의 TSDF 값을 **가중 평균**으로 갱신하면(KinectFusion 방식), 여러 관측이 융합되어 표면이 점점 매끈해진다.

## 3. Marching Cubes — mesh 추출

TSDF는 값의 격자일 뿐이다. 이걸 **눈에 보이는 표면(mesh)**으로 바꾸는 것이 **Marching Cubes**다.

1. 격자를 인접한 8개 voxel이 이루는 **정육면체(큐브)** 단위로 훑는다(march)
2. 각 꼭짓점의 부호(안 $$-$$ / 밖 $$+$$)를 본다 → 8개 꼭짓점이면 $$2^8 = 256$$ 가지 패턴
3. 부호가 바뀌는 **모서리(edge)에서 표면이 지나가므로**, 미리 정의된 삼각형 조합표(lookup table)로 그 큐브 안의 삼각형을 만든다
4. 모서리 위 정확한 위치는 두 꼭짓점 값의 **선형 보간**으로 구한다 → 매끄러운 표면

모든 큐브의 삼각형을 모으면 **닫힌 삼각형 mesh**가 된다.

## 4. TSDF → mesh 파이프라인

```
깊이 프레임들
  → voxel 격자에 TSDF 누적 (가중 평균 융합)
  → Marching Cubes (zero-crossing에서 삼각형 추출)
  → 삼각형 mesh
  → 렌더링 · 충돌 검사 · 시뮬레이션
```

## 정리 & 다음 글

| 개념 | 역할 |
| --- | --- |
| voxel 격자 | 공간을 격자로 표현 |
| TSDF | 부호 거리로 표면 저장·융합 |
| Marching Cubes | zero-crossing → 삼각형 mesh |

- volume 표현은 **여러 관측을 융합해 매끈한 표면**을 얻는 고전적이고 강력한 방법이다.
- 다만 해상도-메모리 트레이드오프가 있어, 최근에는 장면을 **연속 함수/Gaussian**으로 담는 신경망 표현이 등장했다.
- 다음 글: [3D Reconstruction ⑤ 신경망 기반 표현 — NeRF vs 3DGS]({% post_url 2026-07-05-3d-reconstruction-nerf-3dgs %})
