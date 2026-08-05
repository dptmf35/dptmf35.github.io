---
title: "[공부] PnP 풀이법 총정리: P3P, DLT, EPnP, IPPE, SQPnP"
excerpt: "P3P부터 DLT, EPnP, IPPE, SQPnP까지 자주 쓰이는 PnP 풀이법의 원리와 정확도·복잡도 비교"
date: 2026-08-05 12:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - PnP
  - P3P
  - EPnP
  - SQPnP
  - camera pose estimation
toc: true
toc_sticky: true
---

> [마커 & 기하 기반 6DoF 글]({% post_url 2026-07-02-6dof-marker-geometry-based %})에서 PnP를 pipeline의 한 단계로 짧게 다뤘다. 이 글은 PnP 자체를 파고든다. 자주 쓰이는 풀이법(P3P, DLT, EPnP, IPPE, SQPnP)이 각각 문제를 어떤 형태로 바꿔 푸는지, 언제 무엇을 골라야 하는지 정리한다.

## 1. PnP 문제 정의

**PnP(Perspective-n-Point)** 는 3D 점 $$n$$ 개와 그 점들이 찍힌 2D 픽셀의 대응이 주어졌을 때, 카메라의 pose(회전 $$R$$, 이동 $$t$$)를 구하는 문제다. 카메라 투영 모델에서 출발한다.

$$
s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = K \, [R \mid t] \begin{bmatrix} X_i \\ Y_i \\ Z_i \\ 1 \end{bmatrix}
$$

- $$(X_i, Y_i, Z_i)$$: 물체(또는 세계) 좌표계에서 알고 있는 $$i$$ 번째 3D 점, $$(u_i, v_i)$$: 그 점이 관측된 픽셀 좌표
- $$K$$: 카메라 intrinsics ($$3 \times 3$$, calibration으로 미리 획득)
- $$s_i$$: 점마다 다른 depth 비례 상수(스케일)
- 미지수: $$R \in SO(3)$$ (3자유도)와 $$t \in \mathbb{R}^3$$ (3자유도), 합쳐서 6DoF

![PnP 문제의 기하: 알려진 3D 점과 2D 관측, 미지의 R, t](/assets/images/pnp-problem-geometry.svg)

대응 하나가 방정식 2개(픽셀의 $$u, v$$)를 주므로, 미지수 6개를 결정하려면 **최소 3쌍**이 필요하다. 다만 3쌍이면 해가 유일하지 않고(최대 4개), 4쌍부터 일반적으로 유일해진다. 이 문제가 어려운 이유는 $$R$$ 이 그냥 9개 숫자가 아니라 **회전행렬이라는 비선형 제약**($$R^\top R = I$$, $$\det R = 1$$)을 갖기 때문이다. 각 풀이법의 차이는 결국 이 제약을 어떻게 다루느냐의 차이다.

## 2. 풀이법 지도

PnP 풀이법은 크게 세 부류로 나뉜다.

![PnP 풀이법 분류: minimal solver, closed-form, iterative](/assets/images/pnp-solver-map.svg)

- **minimal solver**: 정확히 3점만 쓰는 P3P. RANSAC의 최소 표본용
- **closed-form n점 해법**: 대응 전부를 한 번에 넣고 닫힌형으로 푸는 DLT, EPnP, IPPE, SQPnP
- **iterative 정밀화**: 초기해를 받아 재투영 오차를 비선형 최소화(Gauss-Newton, Levenberg-Marquardt)

실전 pipeline은 보통 "P3P + RANSAC으로 outlier 제거 → n점 해법으로 초기해 → LM 정밀화"의 조합이다.

## 3. P3P: 최소 3점의 기하학

P3P는 1841년 Grunert까지 거슬러 올라가는 가장 오래된 풀이법이다. 핵심 관찰은 두 가지다.

1. 카메라 중심 $$C$$ 에서 두 3D 점 $$A, B$$ 를 보는 **ray 사잇각** $$\theta_{AB}$$ 는 픽셀만으로 계산된다. 정규화 ray $$\hat{r}_i = K^{-1}[u_i, v_i, 1]^\top$$ 두 개 사이의 각이기 때문이다
2. $$A, B$$ 는 모델의 점이므로 둘 사이 거리 $$d_{AB}$$ 를 안다

그러면 미지의 카메라-점 거리 $$a = \lVert CA \rVert, b = \lVert CB \rVert, c = \lVert CC' \rVert$$ ($$C'$$ 은 세 번째 점)에 대해 점 쌍마다 코사인 법칙이 성립한다.

$$
d_{AB}^2 = a^2 + b^2 - 2ab \cos\theta_{AB}
$$

점이 3개면 이런 식이 3개, 미지수도 거리 3개다. 변수를 소거하면 **4차 다항식** 하나로 정리되고, 4차식이므로 실해가 **최대 4개** 나온다. 그래서 P3P는 항상 4번째 점으로 검증한다. 각 후보 pose로 4번째 점을 재투영해 오차가 가장 작은 해를 고른다. 거리 $$a, b, c$$ 가 정해지면 카메라 좌표의 세 점이 복원되므로, 3D-3D 정렬([Kabsch]({% post_url 2026-07-08-kabsch-algorithm-robot-calibration %}))로 $$R, t$$ 를 얻는다.

고전 구현은 "거리 풀기 → 3D-3D 정렬"의 2단계였는데, 현대 solver들은 이를 개선했다.

- **Kneip et al. (2011)**: 중간에 점 정렬을 거치지 않고 카메라 pose를 **직접** 4차식으로 푼다. 수치 오차 누적이 줄어든다
- **AP3P** (Ke & Roumeliotis, 2017): 대수적 소거를 정교화해 더 빠르고 수치적으로 안정적. OpenCV `SOLVEPNP_AP3P`
- **Lambda Twist** (Persson & Nordberg, 2018): 4차식의 근 찾기를 고유값 문제로 바꿔 속도·안정성을 더 높인 현재의 대표 구현

P3P의 존재 이유는 정확도가 아니라 **표본 크기**다. RANSAC의 반복 수는 표본 크기에 지수적으로 늘어나므로([RANSAC 정리]({% post_url 2026-07-05-3d-reconstruction-geometry %})), 최소 표본인 3점 solver가 outlier가 섞인 실전에서 필수가 된다.

## 4. DLT: 선형으로 통째로 풀기

**DLT(Direct Linear Transform)** 는 투영행렬 $$P = K[R \mid t]$$ 의 원소 12개를 하나의 벡터로 보고 선형으로 푼다. 투영식에서 스케일 $$s_i$$ 를 소거하면(양변을 외적 형태로 정리하면) 대응 하나당 $$P$$ 의 원소에 대한 선형 방정식 2개가 나온다.

$$
A \, \mathrm{vec}(P) = 0
$$

여기서 $$A$$ 는 대응들로 채워지는 $$2n \times 12$$ 행렬, $$\mathrm{vec}(P) \in \mathbb{R}^{12}$$ 는 $$P$$ 를 펼친 벡터다. 미지수 12개(스케일 제외 11)이므로 **6쌍 이상**이 필요하고, SVD의 최소 특이벡터가 해가 된다. 얻은 $$P$$ 에서 $$K$$ 를 분리해 $$R, t$$ 를 복원한다.

DLT의 문제는 두 가지다.

- $$K$$ 를 이미 아는데도 12개를 전부 푸는 **낭비**. 그만큼 노이즈에 취약하다
- 최소화하는 것이 기하적 의미가 없는 **대수적 오차**라서, 같은 데이터에서 정확도가 떨어진다. 또한 풀린 $$R$$ 이 정확한 회전행렬이라는 보장이 없어 사후에 직교화해야 한다

그래서 DLT는 단독 PnP solver보다는 calibration 초기화 등 $$K$$ 까지 미지수일 때의 도구로 쓰인다.

## 5. EPnP: 제어점 4개로 O(n)

**EPnP** (Lepetit, Moreno-Noguer & Fua, 2009)는 "미지수를 점 개수와 무관하게 고정하는" 아이디어로 PnP를 사실상 표준화했다. 핵심은 좌표 표현의 변경이다.

![EPnP의 제어점 아이디어: 모든 점을 4개 제어점의 가중합으로 표현](/assets/images/pnp-epnp-control-points.svg)

**1단계: 무게중심 좌표.** 가상의 **제어점(control point)** 4개 $$c_1, \ldots, c_4$$ 를 잡고(보통 점들의 무게중심 + 주성분 방향 3개), 모든 3D 점을 제어점의 가중합으로 표현한다.

$$
p_i^{w} = \sum_{j=1}^{4} \alpha_{ij} \, c_j^{w}, \qquad \sum_{j=1}^{4} \alpha_{ij} = 1
$$

여기서 $$p_i^w$$ 는 물체 좌표계의 $$i$$ 번째 점, $$\alpha_{ij}$$ 는 무게중심 좌표(barycentric coordinates)다. 이 표현의 좋은 점은 **rigid 변환에 불변**이라는 것이다. 카메라 좌표계에서도 같은 $$\alpha_{ij}$$ 로 성립한다.

$$
p_i^{c} = \sum_{j=1}^{4} \alpha_{ij} \, c_j^{c}
$$

**2단계: 선형계 구성.** 투영식에 대입하고 $$s_i$$ 를 소거하면, 미지수는 오직 **제어점 4개의 카메라 좌표 12개**($$x = [c_1^{c}; \ldots; c_4^{c}] \in \mathbb{R}^{12}$$)가 되고, 대응 하나당 선형 방정식 2개가 나온다.

$$
M x = 0
$$

여기서 $$M$$ 은 $$2n \times 12$$ 행렬이다. 점이 아무리 많아도 미지수가 12개로 고정이므로 전체 계산이 **$$O(n)$$** 이다.

**3단계: null space와 스케일 복원.** 해는 $$M$$ 의 null space에 있다.

$$
x = \sum_{k=1}^{N} \beta_k v_k
$$

여기서 $$v_k$$ 는 $$M^\top M$$ 의 가장 작은 고유값들에 대응하는 고유벡터, $$N$$ 은 null space 차원(노이즈 때문에 확정이 어려워 $$N = 1, \ldots, 4$$ 를 모두 시도한다), $$\beta_k$$ 는 미지 계수다. $$\beta_k$$ 는 "제어점 사이 거리는 좌표계와 무관하게 같다"는 제약으로 푼다.

$$
\lVert c_j^{c} - c_k^{c} \rVert^2 = \lVert c_j^{w} - c_k^{w} \rVert^2
$$

$$\beta$$ 에 대한 이 소규모 비선형계는 닫힌형 + 짧은 Gauss-Newton으로 다듬는다. 제어점의 카메라 좌표가 정해지면 물체 좌표의 제어점과의 3D-3D 정렬(Kabsch)로 $$R, t$$ 를 복원한다.

**특성.** 빠르고($$O(n)$$), 대응이 많을수록 안정적이라 초기해의 표준이다. 반면 최소화 대상이 재투영 오차가 아닌 대수적 오차라 최종 정확도는 LM 정밀화에 못 미치고, 점이 적을 때($$n \le 5$$)나 노이즈가 클 때 정확도가 떨어진다. 평면 배치에서는 제어점을 3개로 줄여 대응한다.

## 6. IPPE: 평면 전용 해석해

점들이 모두 한 평면 위에 있으면(마커, 체스보드) 3D-2D 관계가 **homography** $$H$$ 로 축약되고, PnP는 "H에서 $$R, t$$ 분해" 문제가 된다. **IPPE** (Collins & Bartoli, 2014)는 이 경우의 해석적 풀이법으로, 원근이 약할 때 재투영이 거의 같은 **두 후보 해(+θ / −θ 기울기)** 를 모두 계산해 각각의 오차와 함께 반환한다. 평면 PnP 특유의 pose ambiguity와 flip 현상, 실전 대처법은 [마커 글의 3절]({% post_url 2026-07-02-6dof-marker-geometry-based %})에서 자세히 다뤘다. OpenCV `SOLVEPNP_IPPE`(일반 평면), `SOLVEPNP_IPPE_SQUARE`(정사각 마커 전용).

## 7. Iterative: 재투영 오차 최소화

닫힌형 해법들이 최소화하는 대수적 오차는 계산이 편할 뿐 우리가 정말 줄이고 싶은 양이 아니다. 최종 기준은 **재투영 오차**다.

$$
\min_{R, t} \sum_{i=1}^{n} \left\lVert \begin{bmatrix} u_i \\ v_i \end{bmatrix} - \pi(K, R, t, p_i^{w}) \right\rVert^2
$$

여기서 $$\pi(\cdot)$$ 는 3D 점 $$p_i^w$$ 를 현재 pose로 투영해 픽셀을 내는 함수다. 픽셀 노이즈가 가우시안이라는 가정 아래 이 최소화가 **maximum likelihood 추정**이므로, 정확도의 이론적 상한에 해당한다.

문제는 목적함수가 $$R$$ 에 대해 비선형·비볼록이라는 것이다. 그래서 Gauss-Newton이나 **Levenberg-Marquardt(LM)** 로 반복 최적화하며, 회전은 axis-angle 같은 최소 파라미터로 두고 매 반복 선형화한다. 수렴은 지역적이므로 **좋은 초기해가 필수**고, 초기해가 나쁘면 지역 최소값에 빠진다. OpenCV `SOLVEPNP_ITERATIVE`가 이 방식으로, 내부적으로 DLT(비평면) 또는 homography 분해(평면)로 초기화한 뒤 LM을 돈다. 어떤 solver를 쓰든 마지막에 LM 한 번을 얹는 것(`solvePnPRefineLM`)이 사실상 공짜 정확도 향상이다.

## 8. SQPnP: 전역 최적을 보장하는 현대적 해법

**SQPnP** (Terzakis & Lourakis, ECCV 2020)는 "PnP를 처음부터 **비볼록 최적화 문제로 정면 돌파**하되, 전역 최적해를 보장하자"는 접근이다. EPnP처럼 선형화 근사를 하지도, LM처럼 초기해에 의존하지도 않는다.

**1단계: 문제를 이차형식으로.** 정규화 이미지 점 $$m_i = K^{-1}[u_i, v_i, 1]^\top$$ 에 대해 투영식은 $$s_i m_i = R p_i^{w} + t$$ 다. 스케일 $$s_i$$ 는 세 번째 성분에서 $$s_i = e_3^\top (R p_i^w + t)$$ 로 정해지므로($$e_3 = [0,0,1]^\top$$), 이를 대입한 잔차

$$
\epsilon_i = m_i \, e_3^\top (R p_i^{w} + t) - (R p_i^{w} + t)
$$

는 $$R$$ 의 원소와 $$t$$ 에 대해 **선형**이다. $$r = \mathrm{vec}(R) \in \mathbb{R}^9$$ 로 두고 $$\sum_i \lVert \epsilon_i \rVert^2$$ 을 $$t$$ 에 대해 먼저 최소화하면 $$t$$ 는 $$r$$ 의 선형 함수로 닫힌형으로 소거되고, 목적함수는 순수한 이차형식이 된다.

$$
\min_{r} \; r^\top \Omega \, r \quad \text{s.t.} \quad r = \mathrm{vec}(R), \; R \in SO(3)
$$

여기서 $$\Omega$$ 는 데이터로부터 계산되는 $$9 \times 9$$ 준정부호(positive semidefinite) 행렬이다. 즉 PnP 전체가 "**회전행렬 제약이 걸린 이차계획(QCQP)**"으로 정리된다.

**2단계: SQP로 제약면 위 최적화.** 이 문제를 **sequential quadratic programming(SQP)** 으로 푼다. 현재 추정 회전 주변에서 $$SO(3)$$ 제약(직교성 6개 + 방향)을 선형화하고, 등식 제약 이차계획을 풀어 다음 추정으로 이동하는 것을 수렴까지 반복한다. 한 번의 SQP는 지역 수렴이지만, SQPnP는 시작점을 체계적으로 고른다. $$\Omega$$ 의 **작은 고유값에 대응하는 고유벡터들을 가장 가까운 회전행렬로 투영**(orthogonal Procrustes)한 후보들에서 SQP를 각각 시작하고, 그중 최소 비용 해를 고른다. 저자들은 이 시작점 집합이 전역 최소를 놓치지 않음을 보여, 결과적으로 **위 목적함수의 전역 최적해가 보장**된다.

**특성.**

- $$n \ge 3$$ 부터 동작하고, 평면·비평면·near-planar 배치를 **특수 처리 없이 동일하게** 다룬다. EPnP의 평면 분기, IPPE의 평면 전용 같은 케이스 나누기가 없다
- $$\Omega$$ 구성이 $$O(n)$$ 이고 SQP는 9차원 소규모 문제라 점이 많아도 빠르다
- 최소화 대상이 재투영 오차 자체는 아니고 위의 대수적 잔차이지만, 재투영 오차와 매우 가까워 실측 정확도가 기존 방법들과 같거나 낫다. 필요하면 LM 한 번으로 마무리한다
- OpenCV `SOLVEPNP_SQPNP` (4.5부터). `solvePnPGeneric`으로 모든 해 후보를 받을 수도 있다

## 9. 비교와 선택 가이드

| 방법 | 연도 | 최소 점 수 | 복잡도 | 푸는 형태 | 평면 배치 | 초기해 필요 | OpenCV flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P3P (Kneip/AP3P/Lambda Twist) | 1841/2011~ | 3 (+1 검증) | $$O(1)$$ | 4차 다항식 | 가능 | 불필요 | `SOLVEPNP_P3P`, `SOLVEPNP_AP3P` |
| DLT | 고전 | 6 | $$O(n)$$ | 선형(SVD) | 퇴화 | 불필요 | (단독 flag 없음) |
| EPnP | 2009 | 4 | $$O(n)$$ | 선형화 + null space | 제어점 3개로 대응 | 불필요 | `SOLVEPNP_EPNP` |
| IPPE | 2014 | 4 (평면) | $$O(n)$$ | homography 해석해 2개 | 평면 전용 | 불필요 | `SOLVEPNP_IPPE`, `SOLVEPNP_IPPE_SQUARE` |
| Iterative (LM) | 고전 | 6 (평면 4) | $$O(n \cdot \text{iter})$$ | 재투영 오차 비선형 최소화 | 가능 | **필요** | `SOLVEPNP_ITERATIVE` |
| SQPnP | 2020 | 3 | $$O(n)$$ | QCQP + SQP, **전역 최적** | 특수 처리 불필요 | 불필요 | `SOLVEPNP_SQPNP` |

상황별 선택은 다음과 같이 정리할 수 있다.

- **outlier가 섞인 대응(특징점 매칭)**: `solvePnPRansac` (내부: P3P 표본 + inlier로 EPnP/LM 정밀화)
- **깨끗한 대응 다수(보드, 정제된 매칭)**: EPnP 또는 SQPnP로 풀고 LM 정밀화
- **정사각 마커 1개**: `SOLVEPNP_IPPE_SQUARE` + 두 해의 오차 비율로 ambiguity 검사
- **점이 3~5개뿐이거나 near-planar 등 애매한 배치**: SQPnP. 최소 점 수가 3이고 배치를 가리지 않으며 전역 최적이 보장된다
- **비디오 tracking처럼 직전 pose가 있는 경우**: `SOLVEPNP_ITERATIVE` + `useExtrinsicGuess=true` 로 직전 해에서 LM만 돌리는 것이 가장 싸고 정확하다

## 정리

- PnP는 3D-2D 대응 $$n$$ 쌍에서 $$R, t$$ (6DoF)를 구하는 문제고, 어려움의 근원은 $$SO(3)$$ 제약이다
- **P3P**는 코사인 법칙 → 4차식으로 최소 3점을 풀며, RANSAC 최소 표본으로 산다
- **DLT**는 선형이지만 $$K$$ 를 낭비하고 대수적 오차라 정확도가 낮다
- **EPnP**는 제어점 4개의 무게중심 좌표로 미지수를 12개로 고정해 $$O(n)$$ 을 달성한 표준 초기해다
- **IPPE**는 평면 전용 해석해로, 두 후보 해를 모두 반환해 pose ambiguity를 다룰 수 있게 한다
- **Iterative(LM)** 는 재투영 오차의 maximum likelihood 추정으로 정확도의 기준이지만 초기해가 필요하다
- **SQPnP**는 PnP를 회전 제약 이차계획으로 정리하고 SQP로 풀어, $$n \ge 3$$ 어떤 배치에서든 전역 최적을 보장하는 현대적 기본 선택지다

관련 글: [마커 & 기하 기반 6DoF 추정]({% post_url 2026-07-02-6dof-marker-geometry-based %}), [Kabsch 알고리즘]({% post_url 2026-07-08-kabsch-algorithm-robot-calibration %}), [3D 비전 기하학 기초]({% post_url 2026-07-05-3d-reconstruction-geometry %})
