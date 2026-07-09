---
title: "[공부] 3D Reconstruction ② SfM·MVS — 이미지 뭉치에서 dense 3D까지, 전통 파이프라인"
excerpt: "3D reconstruction의 전통 본체. 순서 없는 사진들에서 카메라 자세와 sparse 구조를 복원하는 SfM(incremental·bundle adjustment)과, 이를 dense하게 만드는 MVS를 COLMAP 파이프라인 순서로 정리한다."
date: 2026-07-05 09:30:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - SfM
  - MVS
  - bundle adjustment
  - COLMAP
toc: true
toc_sticky: true
---

> [①편]({% post_url 2026-07-05-3d-reconstruction-geometry %})에서 **두 시점**의 기하학을 풀었다. 이 글은 그것을 **수백 장의 순서 없는 사진**으로 확장한다 — 전통 3D reconstruction의 본체인 **SfM(Structure from Motion)** 과 **MVS(Multi-View Stereo)** 다.

## 1. SfM이란 — two-view에서 N-view로

**SfM**은 순서 없는 이미지 집합에서 **모든 카메라의 자세** $$\{R_i, t_i\}$$ 와 장면의 **sparse 3D 구조**(특징점들의 3D 위치)를 **동시에** 복원하는 문제다. 이름 그대로 "움직임(motion, 카메라 자세)에서 구조(structure)를" 얻는다.

- 빌딩 블록은 전부 ①편에서 나왔다: 특징점 매칭, $$E$$ 분해, triangulation, RANSAC
- two-view와 결정적으로 다른 점은 **누적**이다 — 시점이 늘수록 오차가 쌓이고(drift), 이를 잡는 전역 최적화(**bundle adjustment**)가 파이프라인의 심장이 된다
- 출력은 **자세 + sparse point cloud**. 여전히 스케일은 모른다(①편 3.2의 scale ambiguity) — 전체 지도가 "몇 배"인지는 외부 정보 없이는 정해지지 않는다

전체 흐름은 다음과 같다.

![대응 탐색, 초기 쌍 복원, PnP·triangulation·BA를 반복하는 점진적 확장, 그리고 MVS 조밀화로 이어지는 SfM·MVS 파이프라인](/assets/images/3drecon-sfm-pipeline.svg)

## 2. 대응 탐색 — scene graph 만들기

첫 단계는 ①편 4~5장의 매칭을 **모든 이미지 쌍**으로 확장하는 것이다.

1. 이미지마다 특징점 + descriptor 추출
2. 이미지 쌍마다 매칭 — $$n$$ 장이면 쌍이 $$\binom{n}{2}$$ 개라 전수 매칭은 $$O(n^2)$$. 대규모에서는 **vocabulary tree** 같은 이미지 검색으로 "겹칠 가능성이 있는 쌍"만 골라 매칭한다
3. 쌍마다 RANSAC + epipolar 제약으로 **기하 검증** — 검증을 통과한 매칭 수가 충분한 쌍만 남긴다

결과는 **scene graph**다: 노드가 이미지, 간선이 "검증된 대응을 공유하는 관계"인 그래프. 이후 모든 단계가 이 그래프 위에서 진행된다. 여러 이미지에 걸쳐 이어진 같은 점의 대응 사슬을 **track**이라 한다.

## 3. incremental SfM — 점진적 확장

현재 표준(COLMAP)의 방식이다. 두 시점으로 씨앗을 만들고, 이미지를 한 장씩 붙여 나간다.

### 3.1 초기 쌍 선택

첫 두 시점이 전체 복원의 뼈대가 되므로 신중히 고른다.

- **검증된 매칭이 많고**, 동시에 **baseline이 충분히 넓은** 쌍 (①편 6.4 — baseline이 좁으면 triangulation이 부정확하다)
- 선택한 쌍에서 $$E$$ 분해 → $$(R, t)$$, triangulation → 첫 3D 점들. 이것이 **씨앗 지도**다

### 3.2 새 이미지 등록 — PnP

다음 이미지를 붙일 차례다. 새 이미지의 특징점 중 일부는 이미 3D 위치를 아는 track에 연결되어 있다. 즉 **3D 점 ↔ 2D 관측** 대응이 생긴다. 이때 카메라 자세를 구하는 문제가 **PnP**(Perspective-n-Point)다.

- 알려진 3D 점 $$X_j$$ 와 그 2D 관측 $$x_j$$ 의 쌍 $$n$$ 개에서 $$[R \mid t]$$ 를 추정한다
- 최소 해는 **3쌍**(P3P — 자세 6 DoF, 쌍당 제약 2개). 매칭에 outlier가 섞이므로 항상 **RANSAC과 함께** 쓴다
- ①편의 6DoF pose estimation과 같은 문제다 — [마커·기하 기반 pose]({% post_url 2026-07-02-6dof-marker-geometry-based %})에서도 같은 PnP가 나온다

### 3.3 신규 triangulation과 반복

새 카메라가 등록되면, 그 이미지에서 처음 보이거나 이제야 두 시점 이상을 확보한 track들을 **triangulation**해 3D 점을 늘린다. 그리고 다음 이미지로 — **등록 → triangulation → (주기적) bundle adjustment**를 모든 이미지가 붙을 때까지 반복한다.

### 3.4 global SfM과의 비교

모든 쌍의 상대 자세를 먼저 구해 놓고 **한 번에** 전역으로 푸는 접근도 있다.

| | incremental | global |
| --- | --- | --- |
| 방식 | 한 장씩 등록 + 반복 BA | rotation averaging → translation → 한 번의 BA |
| 정확도·강건성 | 높다 (매 단계 outlier 걸러냄) | outlier·잘못된 쌍에 민감 |
| 속도 | 느리다 (BA를 반복) | 빠르다 |
| 대표 구현 | COLMAP | OpenMVG(global), GLOMAP |

## 4. bundle adjustment — 파이프라인의 심장

등록을 반복하면 오차가 누적된다(drift). **bundle adjustment(BA)** 는 지금까지의 **모든 카메라 자세와 모든 3D 점을 한꺼번에** 움직여 전체 재투영 오차를 최소화한다.

$$
\min_{\{R_i, t_i\},\ \{X_j\}} \;
\sum_{(i,j) \in \mathcal{V}}
\rho\!\left( \left\lVert \, x_{ij} - \pi\!\left(K_i, R_i, t_i, X_j\right) \right\rVert^2 \right)
$$

- $$\{R_i, t_i\}$$: $$i$$ 번째 카메라의 자세, $$\{X_j\}$$: $$j$$ 번째 3D 점 — **둘 다 최적화 변수**다
- $$\mathcal{V}$$: 가시성 집합 — "점 $$j$$ 가 이미지 $$i$$ 에 실제로 관측된" $$(i,j)$$ 쌍만 항에 들어간다
- $$x_{ij}$$: 이미지 $$i$$ 에서 관측된 점 $$j$$ 의 픽셀 위치
- $$\pi(\cdot)$$: 투영 함수 — ①편의 $$P = K[R \mid t]$$ 투영(왜곡 포함). 즉 각 항은 "지금 추정으로 다시 투영하면 관측과 몇 픽셀 어긋나는가"
- $$\rho(\cdot)$$: robust loss(Huber 등) — 잘못된 track 하나가 전체를 끌고 가지 못하게 큰 오차의 영향을 누른다

변수가 수만~수백만 개지만 풀 수 있는 이유는 **희소성(sparsity)** 이다. 각 3D 점은 소수의 카메라에만 보이므로 Jacobian이 희소하고, 카메라 블록과 점 블록을 분리하는 **Schur complement** 트릭으로 Levenberg–Marquardt를 대규모에서도 돌린다(Ceres 등).

- calibration(①편 2.3)의 정련과 같은 목적함수다 — 차이는 **3D 점의 위치까지 미지수**라는 것
- incremental SfM에서 BA를 "주기적으로" 도는 이유: 매 등록마다 풀면 너무 느리고, 안 풀면 drift로 다음 등록이 실패한다

## 5. MVS — sparse를 dense로

SfM의 출력은 **특징점이 있던 자리만** 복원한 sparse 구조다. 표면을 만들기엔 턱없이 모자라다. **MVS**는 자세를 아는 상태에서 **픽셀 단위 깊이**를 추정해 dense 복원을 만든다.

### 5.1 photometric consistency

핵심 가정: 같은 3D 표면은 여러 시점에서 **비슷하게 보인다**. 기준 이미지의 픽셀에 깊이 $$d$$ 를 가정하면 3D 점이 정해지고, 이를 이웃 이미지에 투영해 **패치 유사도**(NCC 등)를 잰다. 유사도가 최대가 되는 $$d$$ 가 그 픽셀의 깊이다.

- 자세를 이미 알므로 후보 위치는 ①편의 **epipolar line 위 1D 탐색**으로 줄어든다
- 픽셀마다 깊이와 함께 **normal**도 추정하면 기울어진 표면에서 패치 비교가 정확해진다

### 5.2 PatchMatch와 depth fusion

픽셀 × 깊이 후보를 전수 탐색하면 너무 비싸다. **PatchMatch**는 이를 확률적으로 푼다.

1. **랜덤 초기화**: 픽셀마다 무작위 깊이·normal
2. **전파(propagation)**: 이웃 픽셀의 (더 좋은) 가설을 물려받는다 — 표면은 대개 매끈해서 이웃의 답이 내 답과 비슷하다
3. **무작위 정련**: 현재 가설 주변을 흔들어 개선
4. 2~3을 몇 회 반복하면 시점별 **depth map**이 나온다

마지막으로 모든 시점의 depth map을 3D로 되쏘아 합치는 **depth fusion**을 한다 — 여러 시점에서 일관된 깊이만 남기고(가림·오추정 제거) **dense point cloud**를 만든다. 학습 기반 MVS(MVSNet 계열)는 이 비용 계산·정합을 신경망으로 대체한 것이다.

## 6. COLMAP, 그리고 전통 파이프라인의 한계

**COLMAP**은 incremental SfM + PatchMatch MVS를 묶은 표준 구현이다. 사진 폴더를 넣으면 자세·sparse·dense·mesh까지 뽑아 준다. NeRF·3DGS 논문들이 "카메라 자세는 COLMAP으로 구했다"라고 쓰는 그 도구다.

전통 파이프라인이 약한 곳은 가정이 깨지는 곳이다.

- **textureless 영역**(흰 벽, 매끈한 금속): 특징점도 photometric consistency도 잡을 것이 없다
- **반사·투명 표면**: 시점마다 다르게 보여 같은 표면이라는 가정이 무너진다
- **반복 패턴**: 잘못된 매칭이 기하 검증을 통과하기도 한다
- **시점이 적거나 baseline이 나쁠 때**: triangulation 각이 안 나온다

이 빈틈이 [NeRF·3DGS]({% post_url 2026-07-05-3d-reconstruction-nerf-3dgs %}) 같은 신경망 표현의 동기가 된다 — 다만 그들도 **입력 자세는 SfM에 의존**하므로, SfM은 여전히 모든 3D reconstruction의 출발점이다.

## 정리 & 다음 글

| 단계 | 입력 | 출력 |
| --- | --- | --- |
| 대응 탐색 | 이미지 집합 | scene graph, track |
| incremental SfM | scene graph | 카메라 자세 + sparse point cloud |
| bundle adjustment | 자세·점 전체 | drift 잡힌 전역 최적해 |
| MVS | 자세 + 이미지 | 시점별 depth map → dense point cloud |

- SfM은 ①편의 two-view 블록(매칭·$$E$$·PnP·triangulation)을 **BA로 묶어 N장으로 확장**한 것이다.
- MVS가 sparse를 dense로 채워, 다음 단계인 **point cloud 처리**의 입력을 만든다.
- 다음 글: [3D Reconstruction ③ point cloud — 노이즈 제거와 downsampling]({% post_url 2026-07-05-3d-reconstruction-point-cloud %})
