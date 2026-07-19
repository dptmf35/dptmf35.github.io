---
title: "[공부] 로봇 calibration을 위한 Kabsch 알고리즘: 기준점·측정점 정렬의 수학"
excerpt: "기준점·측정점을 잇는 최적 rigid transform (R, t)을 SVD 한 번으로 구하는 Kabsch 알고리즘"
date: 2026-07-08 12:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - Kabsch
  - calibration
  - point set registration
  - SVD
  - robot vision
toc: true
toc_sticky: true
---

> 로봇 base와 tool을 calibration할 때, 우리는 **알고 있는 기준점**과 **센서로 잰 측정점**이라는 두 벌의 대응하는 점을 손에 쥔다.
> 이 둘을 잇는 **최적 rigid transform (R, t)** 을 SVD 한 번으로 닫힌형으로 푸는 것이 **Kabsch 알고리즘**이다. 문제 정의부터 유도·구현·잔차 해석까지 따라간다.

## 1. 문제 정의: 두 point set을 잇는 rigid transform

로봇 calibration의 뼈대는 같다. 좌표계 A에서 알고 있는 점들과, 좌표계 B에서 그 점들을 잰 측정값이 **쌍(pair)** 으로 주어진다.

- **base calibration**: 외부 tracker/camera가 잰 점 ↔ 로봇 base 좌표계에서 아는 점
- **tool(TCP) calibration**: 여러 자세에서 잰 tool 끝점 ↔ 명령한 목표점
- 공통점: **대응이 이미 알려진** 두 point set의 정렬

기준점 $$Q = \{q_i\}$$ 와 측정점 $$P = \{p_i\}$$ ($$i = 1, \dots, N$$, $$p_i, q_i \in \mathbb{R}^3$$)가 주어질 때, 측정점을 기준점에 겹치는 회전 $$R \in SO(3)$$ 과 평행이동 $$t \in \mathbb{R}^3$$ 을 찾는다. 목적함수는 대응점 거리의 제곱합이다.

$$
E(R, t) = \sum_{i=1}^{N} w_i \, \lVert R p_i + t - q_i \rVert^2
$$

$$w_i$$ 는 각 점의 신뢰도 가중치(없으면 $$w_i = 1$$)다. 이 문제는 **orthogonal Procrustes 문제**의 rigid 버전으로, Kabsch(1976)가 닫힌형 해를 제시했다.

- 회전만 자유롭고 스케일·shear가 없다 → **rigid**(강체) 정렬
- 대응 $$p_i \leftrightarrow q_i$$ 가 **이미 알려져 있다** → 대응을 함께 추정하는 [ICP]({% post_url 2026-07-03-6dof-point-cloud-registration %})와 다르다. Kabsch는 ICP 각 iteration의 **정렬 단계 그 자체**다.

## 2. 알고리즘 한눈에

전체 흐름은 다섯 단계다. 반복 없이 한 번에 끝난다.

![측정점·기준점 입력에서 중심 정렬, 교차공분산 H, SVD, 그리고 R·t 해까지 이어지는 Kabsch 알고리즘의 5단계 파이프라인](/assets/images/kabsch-pipeline.svg)

1. **무게중심(centroid)** $$\bar{p}, \bar{q}$$ 를 구한다
2. 두 point set을 무게중심 기준으로 **중심 정렬**한다: $$x_i = p_i - \bar{p}$$, $$y_i = q_i - \bar{q}$$
3. **교차공분산 행렬** $$H = \sum_i w_i \, x_i y_i^\top$$ ($$3 \times 3$$)을 만든다
4. $$H$$ 를 **SVD** 한다: $$H = U \Sigma V^\top$$
5. **reflection 처리** 후 해를 얻는다: $$R = V D U^\top$$, $$t = \bar{q} - R \bar{p}$$

## 3. 왜 이렇게 푸는가

### 3.1 무게중심으로 translation을 떼어낸다

핵심은 $$R$$ 과 $$t$$ 를 동시에 풀지 않고 **분리**하는 것이다. $$R$$ 을 고정한 채 $$E$$ 를 $$t$$ 로 미분해 0으로 두면

$$
\frac{\partial E}{\partial t} = 2 \sum_i w_i (R p_i + t - q_i) = 0
\quad\Rightarrow\quad
t^\ast = \bar{q} - R \bar{p}
$$

즉 **최적 $$t$$ 는 두 무게중심을 겹치는 값**으로, $$R$$ 만 알면 자동으로 정해진다. 이 $$t^\ast$$ 를 다시 대입하면 목적함수에서 translation이 사라지고, 중심 정렬된 좌표 $$x_i, y_i$$ 만 남는다.

$$
E(R) = \sum_i w_i \, \lVert R x_i - y_i \rVert^2
$$

### 3.2 trace 최대화 → SVD

위 식을 전개하면

$$
E(R) = \sum_i w_i \left( \lVert x_i \rVert^2 + \lVert y_i \rVert^2 - 2\, y_i^\top R x_i \right)
$$

앞 두 항은 $$R$$ 과 무관한 상수다. 따라서 $$E$$ 를 **최소화**하는 것은 마지막 항, 즉

$$
\sum_i w_i \, y_i^\top R x_i = \operatorname{tr}\!\left( R \sum_i w_i \, x_i y_i^\top \right) = \operatorname{tr}(R H)
$$

를 **최대화**하는 것과 같다. 여기서 교차공분산 $$H = \sum_i w_i \, x_i y_i^\top$$ 이 등장한다. 이제 $$H = U \Sigma V^\top$$ 로 SVD 하면

$$
\operatorname{tr}(R H) = \operatorname{tr}(R U \Sigma V^\top) = \operatorname{tr}(\Sigma \, V^\top R U)
$$

$$M = V^\top R U$$ 는 직교행렬이라 대각 원소가 $$\lvert M_{ii} \rvert \le 1$$ 이다. $$\Sigma$$ 의 특이값 $$\sigma_i \ge 0$$ 이므로

$$
\operatorname{tr}(\Sigma M) = \sum_i \sigma_i M_{ii} \le \sum_i \sigma_i
$$

등호는 $$M = I$$, 곧 $$V^\top R U = I$$ 일 때다. 정리하면 최적 회전은

$$
R = V U^\top
$$

## 4. Reflection 처리 (det correction)

$$R = V U^\top$$ 은 직교행렬이지만 $$\det R = -1$$ 인 **반사(reflection)** 일 수 있다. 점들이 거의 한 평면에 몰려 있거나 노이즈가 크면 실제로 이런 해가 나온다. 로봇 좌표계에는 반사가 없으므로 $$SO(3)$$ (proper rotation, $$\det = +1$$)으로 강제해야 한다.

해법은 마지막 특이값에 부호를 넣는 것이다.

$$
d = \operatorname{sign}\!\left( \det(V U^\top) \right), \qquad
D = \begin{bmatrix} 1 & & \\ & 1 & \\ & & d \end{bmatrix}, \qquad
R = V D U^\top
$$

$$d = +1$$ 이면 그대로, $$d = -1$$ 이면 가장 작은 특이값 방향만 뒤집어 반사를 회전으로 바로잡는다. 이때 잃는 정확도는 **가장 작은 $$\sigma$$ 에 해당하는 축**에 국한되므로 손해가 가장 적다. 이 한 줄을 빠뜨리면 드물게 좌우가 뒤집힌 calibration이 나온다.

## 5. 인터랙티브: 정렬 과정 보기

어긋난 측정점(파랑)에 Kabsch가 구한 $$(R, t)$$ 를 점진적으로 적용해 기준점(초록)에 겹치는 과정이다. **재생**하면 진행률 $$f$$ 를 0→1로 올리며, 그때의 잔차 RMSD가 노이즈 바닥까지 줄어드는 걸 볼 수 있다. **새 데이터**로 초기 자세·노이즈를 바꿔가며 확인한다.

{% include kabsch-align.html %}

무게중심(테두리 원)이 먼저 겹쳐지고($$t$$), 이어 그 점을 축으로 회전($$R$$)이 걸리는 순서가 3.1의 "translation 분리"와 정확히 대응한다.

## 6. 구현

numpy로는 유도 그대로 열 줄이면 된다.

```python
import numpy as np

def kabsch(P, Q, w=None):
    """측정점 P (N,3)를 기준점 Q (N,3)에 맞추는 R, t. Q ≈ R @ P.T + t"""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()

    # 1) 가중 무게중심, 2) 중심 정렬
    p_bar = (w[:, None] * P).sum(0)
    q_bar = (w[:, None] * Q).sum(0)
    X, Y = P - p_bar, Q - q_bar

    # 3) 교차공분산 4) SVD
    H = (X * w[:, None]).T @ Y
    U, S, Vt = np.linalg.svd(H)

    # 5) reflection 처리
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_bar - R @ p_bar
    return R, t

def rmsd(P, Q, R, t):
    err = (R @ P.T).T + t - Q
    return np.sqrt((err ** 2).sum(1).mean())
```

- `svd`가 돌려주는 것은 $$U, \Sigma, V^\top$$ 이라 코드의 `Vt`가 $$V^\top$$, 따라서 $$V = $$ `Vt.T` 다
- $$N \ge 3$$ 이고 점들이 **한 직선/평면에 몰려 있지 않아야** $$R$$ 이 유일하게 정해진다

## 7. 잔차와 outlier: calibration 품질 판단

정렬 후 남는 **RMSD가 calibration 오차의 지표**다.

$$
\text{RMSD} = \sqrt{\frac{1}{N} \sum_i \lVert R p_i + t - q_i \rVert^2}
$$

- 값이 센서 노이즈 수준까지 내려가면 좋은 정렬이다
- 특정 점만 크게 튀면 그 대응이 **outlier**(잘못된 측정·잘못된 대응)일 가능성이 높다

Kabsch는 **최소제곱**이라 outlier에 민감하다. 한 점이 크게 어긋나면 전체 $$R, t$$ 가 그쪽으로 끌려간다. 실무에서는 이렇게 방어한다.

| 방법 | 아이디어 |
| --- | --- |
| RANSAC | 최소 대응(3쌍)으로 후보 $$(R,t)$$ 를 여러 번 뽑고, inlier가 가장 많은 해를 채택 |
| robust weighting | 잔차가 큰 점의 $$w_i$$ 를 낮춰(Huber 등) 재정렬을 반복 |
| trimmed | 잔차 상위 몇 %를 버리고 Kabsch를 다시 |

가중치 $$w_i$$ 를 그대로 받는 위 구현이 robust weighting·RANSAC의 정렬 단계로 재사용된다.

## 정리

- **Kabsch 알고리즘**은 대응이 알려진 두 point set을 잇는 최적 rigid transform $$(R, t)$$ 를 **SVD 한 번**으로 닫힌형으로 푼다
- **무게중심으로 translation을 분리**($$t = \bar{q} - R\bar{p}$$)하고, 남은 회전은 **$$\operatorname{tr}(RH)$$ 최대화 = $$H$$ 의 SVD**로 $$R = V D U^\top$$
- **det correction $$D$$** 로 반사를 막아 $$SO(3)$$ 을 보장한다
- 정렬 후 **RMSD가 calibration 오차**이며, outlier에는 RANSAC·robust weighting을 얹는다

**관련 글**

- [ICP와 point cloud registration]({% post_url 2026-07-03-6dof-point-cloud-registration %}): 대응을 모를 때 Kabsch를 반복해 정렬을 찾는 방법
- [point cloud 다루기]({% post_url 2026-07-05-3d-reconstruction-point-cloud %}): 정렬 전 노이즈·outlier 전처리
