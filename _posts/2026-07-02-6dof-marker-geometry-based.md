---
title: "[공부] 마커 & 기하 기반 6DoF 추정 — fiducial marker, PnP, LINEMOD"
excerpt: "fiducial marker와 PnP 풀이법(P3P·EPnP·IPPE), 평면 pose ambiguity, template 기반 LINEMOD까지 전통 기하 기반 6DoF 추정"
date: 2026-07-02
categories:
  - 공부
  - CV
tags:
  - 6DoF pose estimation
  - AprilTag
  - ArUco
  - PnP
  - LINEMOD
toc: true
toc_sticky: true
---

> [시리즈 개요]({% post_url 2026-07-01-6dof-pose-estimation-overview %}) 의 첫 번째 방법론. 딥러닝 이전부터 로봇에서 널리 쓰인 **기하 기반** 접근을 정리한다.

## 1. 핵심 아이디어: 2D–3D 대응 → PnP

기하 기반 방법의 뼈대는 **알려진 3D 점**과 **이미지에서 관측한 2D 점**의 대응(correspondence)을 만들고, 이를 만족하는 카메라-물체 변환을 푸는 것이다. 이 문제를 **PnP(Perspective-n-Point)** 라고 한다.

카메라 투영 모델은 다음과 같다.

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \,[R \mid t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

- $$K$$: 카메라 내부 파라미터(intrinsics, calibration으로 획득)
- $$(X,Y,Z)$$: 물체 좌표계의 3D 점 / $$(u,v)$$: 이미지 픽셀
- 미지수: $$R, t$$ (즉 6DoF 포즈)

3D–2D 대응이 최소 3~4쌍 이상 있으면 $$R, t$$ 를 풀 수 있다. 실제로 어떻게 푸는지 대표 풀이법을 하나씩 본다.

![마커 검출 → 코너 2D-3D 대응 → PnP → 6DoF 포즈 파이프라인](/assets/images/6dof-marker-pnp.svg)

### P3P — 최소 3점, 최대 4개의 해

카메라 중심 $$C$$ 에서 두 3D 점 $$A, B$$ 를 보는 **ray 사잇각** $$\theta_{AB}$$ 는 픽셀 좌표만으로 계산된다 — 정규화 ray $$\hat{r} = K^{-1}[u, v, 1]^\top$$ 두 개의 cosine이다. 한편 $$A, B$$ 는 물체 모델의 점이므로 둘 사이 거리 $$d_{AB}$$ 도 안다. 그러면 미지의 카메라-점 거리 $$a = \lVert CA \rVert,\ b = \lVert CB \rVert$$ 에 대해 코사인 법칙이 성립한다.

$$
d_{AB}^2 = a^2 + b^2 - 2ab\cos\theta_{AB}
$$

- 3쌍의 대응이면 이런 방정식이 3개, 미지수도 거리 3개($$a, b, c$$) — 정리하면 **4차 다항식**이 되어 **실해가 최대 4개** 나온다
- 그래서 P3P는 항상 **4번째 점으로 검증**한다: 각 후보 pose로 4번째 점을 재투영해 가장 잘 맞는 해를 고른다
- 거리 $$a, b, c$$ 가 정해지면 3D-3D 정렬([Kabsch]({% post_url 2026-07-08-kabsch-algorithm-robot-calibration %}))로 $$R, t$$ 를 복원한다. OpenCV의 `SOLVEPNP_P3P`/`SOLVEPNP_AP3P`

RANSAC의 최소 표본으로 쓰이는 것이 바로 이 P3P다 — 표본이 작을수록 반복 수가 기하급수로 줄기 때문이다([기하학 글의 RANSAC]({% post_url 2026-07-05-3d-reconstruction-geometry %}) 참고).

### DLT와 EPnP — 대응이 많을 때

- **DLT**: 투영행렬 $$P = K[R \mid t]$$ 의 12개 원소를 통째로 선형으로 푸는 방식(6쌍 이상). 그런데 $$K$$ 를 이미 아는 상황에서도 12개를 다 풀므로 낭비고, 대수적 오차 최소화라 정확도가 낮다
- **EPnP** (Lepetit et al., 2009): 모든 3D 점을 **4개의 제어점(control point)에 대한 무게중심 좌표**로 표현한다. 그러면 미지수가 제어점들의 카메라 좌표 12개로 고정되어, 대응이 $$n$$ 쌍이어도 **$$O(n)$$** 에 풀린다 — 대응이 많은 실전에서 표준 초기해

### 반복 정밀화 — 재투영 오차 최소화

닫힌형 해(P3P/EPnP)는 초기값이고, 최종적으로는 **재투영 오차**를 비선형 최소화(Levenberg–Marquardt)로 다듬는다.

$$
\min_{R, t} \sum_i \left\lVert\, \begin{bmatrix} u_i \\ v_i \end{bmatrix} - \pi\!\left(K, R, t, X_i\right) \right\rVert^2
$$

- $$\pi(\cdot)$$: 3D 점 $$X_i$$ 를 현재 pose로 투영하는 함수, $$(u_i, v_i)$$: 관측 픽셀 — "지금 pose로 다시 투영하면 관측과 몇 픽셀 어긋나는가"([camera calibration의 목적함수]({% post_url 2026-07-05-3d-reconstruction-geometry %})와 같은 형태)
- 잘못된 대응(outlier)에 대비해 전체를 **RANSAC**으로 감싼다 — OpenCV `solvePnPRansac`이 P3P 표본 + EPnP/LM 정밀화의 표준 조합이다

## 2. 인공 마커(Fiducial Marker)

물체나 환경에 **알려진 패턴의 마커**를 붙이면 대응 문제를 아주 쉽고 견고하게 만들 수 있다. 마커의 네 꼭짓점은 3D 좌표가 정확히 알려져 있으므로, 이미지에서 꼭짓점만 검출하면 곧바로 PnP를 풀 수 있다.

### AprilTag
- 미시간대 APRIL 연구실에서 제안(Olson, 2011). QR코드를 닮은 2D 바코드형 마커
- **서브픽셀 정확도**의 코너 검출 + 해밍 거리로 안전한 ID 디코딩 → 조명/블러/부분 가림에 강인
- `tag36h11` 등 여러 태그 패밀리 제공. AprilTag 3에서는 더 유연한 레이아웃 지원
- 로봇에서 **hand-eye calibration**, 물체 위치 기준 마커, AR 등에 광범위하게 사용

### ArUco / ChArUco
- OpenCV에 내장된 마커 시스템(Garrido-Jurado, 2014). 사용이 간편
- **ChArUco**는 체스보드 + ArUco를 결합해 코너 정확도를 높여 calibration에 유리

### 검출 파이프라인 — 코너는 어떻게 나오나 (AprilTag 기준)

마커 검출기가 내부에서 하는 일은 대략 네 단계다.

1. **이진화·세그멘테이션**: 밝기 gradient 기반으로 어두운/밝은 영역 경계를 찾는다
2. **quad 검출**: 경계 픽셀들을 클러스터링(union-find)하고 선분을 fitting해 **사각형 후보**를 만든다
3. **코너 서브픽셀 정밀화**: 경계의 gradient에 직선을 다시 맞추고, 네 직선의 **교점**으로 코너를 계산한다 — 픽셀 격자보다 정밀한 코너가 나오는 이유이고, **PnP 정확도의 원천**이다
4. **디코딩**: homography로 태그 내부 격자를 정면으로 펴서 비트를 샘플링하고, 태그 사전과의 **해밍 거리**로 ID를 확정한다 — 몇 비트가 틀려도 오류 정정이 되므로 오검출이 드물다

### 다중 마커 / marker board

마커 하나는 대응이 **코너 4개**뿐이다 — 노이즈에 민감하고, 일부가 가려지면 통째로 실패한다. 실전 처방은 **board**다.

- 여러 마커의 **상대 배치를 미리 알고 있는** 판(board)을 쓰면, 검출된 모든 마커의 코너를 **한 번의 PnP에 몰아넣을 수 있다** — 대응이 4개에서 $$4N$$ 개가 되어 정확도·안정성이 크게 오른다
- 일부 마커가 가려져도 나머지로 동작한다
- **ChArUco가 calibration 표준인 이유**가 이것이다: 체스보드 코너(검출이 가장 정밀) + ArUco ID(어느 코너인지 전역 식별)의 결합
- **크기·거리 감각**: pose 오차는 대략 (코너 픽셀 노이즈) × (거리/마커 크기)에 비례해 커진다. 특히 **회전(마커 법선 방향)이 가장 먼저 무너진다** — 다음 절의 ambiguity와 직결되는 문제다

> **매니퓰레이션 팁**: 마커 기반은 정확·고속이라 calibration, 고정 지그(fixture), 알려진 부품 정렬에 특히 강력하다. 다만 실제 대상 물체마다 마커를 붙이기 어려운 경우가 많다는 게 한계다.

## 3. 평면 마커의 함정 — pose ambiguity와 IPPE

마커의 코너 4개는 **모두 한 평면 위**에 있다. 평면 대상의 PnP에는 고유한 함정이 있다 — **원근 효과가 약하면 서로 다른 두 pose가 거의 같은 재투영을 만든다.**

- 마커가 작거나 멀면 투영이 **약한 원근(weak perspective)**에 가까워지고, 카메라 ray에 대해 **거울상으로 기울어진 두 자세(+θ / −θ)** 의 재투영 차이가 코너 노이즈보다 작아진다
- 프레임마다 노이즈에 따라 두 해가 번갈아 선택되면 pose가 **덜덜 떨리며 뒤집히는(flip)** 현상이 생긴다 — 마커 pose의 법선 방향이 튀는 전형적인 증상
- **IPPE** (Collins & Bartoli, 2014): 평면 전용 풀이법으로, homography의 1차 원근 전개에서 **두 후보 해를 해석적으로 모두 계산**하고 각각의 재투영 오차와 함께 반환한다. 정사각 마커 전용인 OpenCV `SOLVEPNP_IPPE_SQUARE`가 마커 pose의 사실상 표준이다
- 실무 처방: **두 해의 오차 비율을 확인**하고, 비슷하면(모호하면) 그 프레임의 회전을 신뢰하지 않는다 — 이전 프레임과의 연속성으로 해를 고르거나, 시간 필터링, 또는 **board로 대응을 늘려** 모호성 자체를 줄인다

아래 위젯에서 직접 확인할 수 있다 — 거리를 멀리 하거나 노이즈를 키우면 두 해의 오차가 비슷해지고 flip이 시작된다.

{% include marker-flip-ambiguity.html %}

## 4. 마커 없이: 특징점 & template

마커를 붙일 수 없을 때는 물체 자체의 기하/외형 정보를 사용한다.

### 특징점 기반
- SIFT/ORB 같은 로컬 특징으로 이미지와 물체 모델(텍스처가 있는 CAD/스캔)을 매칭 → 2D–3D 대응 → PnP + RANSAC
- **무늬(texture)가 있는 물체**에 잘 동작. 금속처럼 무늬 없는 물체엔 취약

### template 기반 — LINEMOD
- Hinterstoisser 등(2012)이 제안한 **무늬 없는 물체**용 고전 방법. 물체를 수천 개 시점에서 렌더링해 template을 만들고, 입력 영상과 매칭해 대략적 포즈를 찾은 뒤 depth로 ICP 정밀화한다
- **무엇을 매칭하나**: 밝기 자체가 아니라 **gradient의 방향(orientation)** — 물체 윤곽에서의 gradient 방향과 (depth가 있으면) 표면 normal 방향. 밝기 크기와 달리 방향은 **조명 변화에 불변**이라 texture-less 물체의 윤곽 정보를 안정적으로 쓸 수 있다
- **왜 빠른가**: 방향을 몇 개 구간으로 **양자화해 비트로 표현**하고, 이웃 픽셀로 방향 비트를 퍼뜨리는 **spreading**으로 작은 위치 어긋남을 흡수한다. 유사도 계산이 비트 연산 + 사전계산된 response map 조회가 되어, **수천 개 template을 실시간으로** 매칭한다
- 딥러닝 이전 texture-less 6DoF의 사실상 표준이었고, 이후 학습 기반 방법의 비교 기준(LINEMOD 데이터셋)이 됨

## 5. 장단점 정리

| 항목 | 마커 기반 | 특징점/template 기반 |
| --- | --- | --- |
| 정확도 | 매우 높음 | 물체·환경에 따라 편차 |
| 속도 | 빠름(실시간) | 중간 |
| 사전 준비 | 마커 부착 필요 | 물체 CAD/template 필요 |
| 약점 | 마커를 못 붙이는 대상 | 무늬 없음/가림에 취약 |
| 대표 용도 | calibration, 지그, AR | 알려진 부품 인식 |

## 정리 & 다음 글

- 뼈대는 **2D–3D 대응 + PnP(+RANSAC)** 이다. 마커는 이 대응을 "공짜로" 만들어 준다. P3P(최소해)·EPnP(다수 대응)·재투영 오차 정밀화가 표준 조합이다.
- **평면 마커에는 pose ambiguity가 있다** — 멀거나 작은 마커의 회전은 IPPE의 두 해 오차를 확인하고, board로 대응을 늘려 방어한다.
- 통제된 환경·calibration엔 여전히 최고의 실용성을 보인다.
- 하지만 **임의의 물체·무늬 없는 표면**엔 한계가 있어 깊이(point cloud) 기반 정합으로 확장한다.
- 다음 글: [point cloud 정합 기반 6DoF 추정]({% post_url 2026-07-03-6dof-point-cloud-registration %})
