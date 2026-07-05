---
title: "[논문리뷰] PPO: Proximal Policy Optimization Algorithms (2017)"
excerpt: "TRPO의 안정성을 1차 최적화만으로 달성한 PPO. clipped surrogate objective의 설계 의도부터 실험, 결론까지 논문의 구성을 따라 정리한다."
date: 2026-07-05
categories:
  - 논문리뷰
  - RL
tags:
  - PPO
  - reinforcement learning
  - policy gradient
  - TRPO
toc: true
toc_sticky: true
---

> **TRPO가 주는 안정성을, 1차 최적화(Adam)만으로 달성할 수 있는가?** 이 한 문장이 PPO의 출발점이다.
> 답은 "확률 비율을 클리핑하면 된다"이고, 이 단순함이 논문 전체를 관통하는 아이디어다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347) |
| 저자 | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov |
| 소속 / 연도 | OpenAI, 2017 (arXiv:1707.06347) |
| 분야 | 강화학습 / 정책 경사 최적화 |
| 코드 | [OpenAI Baselines (ppo2)](https://github.com/openai/baselines) |

## 1. 배경: PG → TRPO → PPO

2017년 당시 딥RL의 주류 세 접근은 각기 다른 벽에 막혀 있었다.

- **Deep Q-learning**: 이산 행동 공간에 특화된 구조라, 로봇 제어처럼 연속·고차원 행동 공간에 적용하기 어렵다
- **Vanilla policy gradient(A2C/A3C)**: 구현은 단순하지만 수집한 데이터로 **딱 한 번만** gradient 업데이트를 하므로 샘플 효율이 낮다. 같은 데이터로 여러 번 업데이트하면 정책이 붕괴(policy collapse)한다
- **TRPO**: policy update 시 KL divergence를 명시적 제약으로 두어 **monotonic improvement를 보장**한다. 그러나 2차 최적화(conjugate gradient + Fisher-vector product + line search)가 필요해 구현이 복잡하고, dropout·파라미터 공유 아키텍처와 호환되지 않는다

PPO는 TRPO의 안정성을 유지하되 복잡도를 걷어내는 것을 목표로 한다.

![Vanilla PG는 안정성 문제, TRPO는 복잡도 문제, PPO는 clipping으로 둘을 동시에 해결](/assets/images/ppo-evolution.svg)

## 2. 배경 수식: Policy Gradient와 TRPO

### Policy Gradient

기대 누적 보상 $$J(\pi_\theta) = \mathbb{E}_\tau[\sum_t r_t]$$ 를 최대화하며, gradient는 다음과 같다.

$$
\hat{g} = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\right]
$$

여기서 $$\hat{A}_t$$ 는 어드밴티지 추정값으로, 행동 $$a_t$$ 가 기준선(value function)보다 얼마나 좋은지를 나타낸다. Vanilla PG의 핵심 제약은 **같은 샘플로 한 번만 업데이트**해야 한다는 것이다. 여러 번 쓰면 $$\pi_\theta$$ 와 샘플을 만든 $$\pi_{\theta_\text{old}}$$ 의 괴리가 커져 정책이 붕괴한다.

### TRPO

확률 비율 $$r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$ 를 도입하면, TRPO가 최대화하는 CPI(Conservative Policy Iteration) objective는:

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right]
$$

TRPO는 이를 KL 제약 하에 최적화한다.

$$
\underset{\theta}{\text{maximize}}\; L^{CPI}(\theta) \quad \text{s.t.} \quad \hat{\mathbb{E}}_t\left[\text{KL}[\pi_{\theta_\text{old}}(\cdot \mid s_t),\, \pi_\theta(\cdot \mid s_t)]\right] \leq \delta
$$

Kakade & Langford(2002)의 이론으로 단조 향상이 보장되지만, 제약을 만족시키는 2차 최적화가 구현을 무겁게 만든다.

## 3. 핵심: Clipped Surrogate Objective

PPO는 KL 제약 대신 **클리핑**으로 정책 변화를 억제한다.

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \operatorname{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t\right)\right]
$$

핵심 설계는 $$\min$$ 이다. 클리핑한 값과 안 한 값 중 **더 비관적인(pessimistic) 값**을 택한다. 경우의 수로 분해하면 의도가 드러난다.

| 조건 | $$r_t$$ 상태 | $$\min$$ 선택 | 효과 |
| --- | --- | --- | --- |
| $$\hat{A}_t>0$$, $$r_t \le 1+\epsilon$$ | 정상 | $$r_t\hat{A}_t$$ (unclipped) | 정상 학습 |
| $$\hat{A}_t>0$$, $$r_t > 1+\epsilon$$ | 과도 증가 | $$(1+\epsilon)\hat{A}_t$$ (clipped) | gradient 차단 |
| $$\hat{A}_t<0$$, $$r_t \ge 1-\epsilon$$ | 정상 | $$r_t\hat{A}_t$$ (unclipped) | 정상 학습 |
| $$\hat{A}_t<0$$, $$r_t < 1-\epsilon$$ | 과도 감소 | $$(1-\epsilon)\hat{A}_t$$ (clipped) | gradient 차단 |

결과적으로 objective가 **개선되는** 방향으로 $$r_t$$ 가 범위를 벗어나면 이득을 무시하고, **악화되는** 방향으로는 끝까지 페널티를 주는 **단방향 보수주의(one-sided conservatism)** 가 구현된다.

![A>0이면 상승 후 평탄, A<0이면 평탄 후 하강 — 클리핑 밖에서 gradient가 0이 된다](/assets/images/ppo-clip-objective.svg)

### 왜 단순 클리핑이 아니라 min인가

단순히 $$\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t$$ 만 쓰면, $$r_t > 1+\epsilon$$ 이면서 $$\hat{A}_t < 0$$ 인 상황 — 즉 **나쁜 행동의 확률이 과도하게 높아지는** 경우 — 에도 gradient를 차단해 나쁜 방향으로의 이탈을 방치한다. $$\min$$ 을 쓰면 이 경우 unclipped 값이 더 작으므로 페널티가 온전히 전달된다.

머신러닝 관점에서 $$L^{CLIP}$$ 은 **확률 비율 공간에서의 암묵적 근위(proximal) 정규화**로 볼 수 있다. $$\ell_2$$ 정규화가 파라미터 변화를 제한하듯, clipping은 정책 자체의 변화를 제한한다 — TRPO의 명시적 KL 제약과 기능적으로 유사하되 훨씬 단순하다.

### 왜 중요한가

- **다중 에폭 미니배치 업데이트가 안전해진다.** clipping이 매 step마다 $$r_t$$ 범위를 제한하므로, $$N$$ 개 병렬 액터가 $$T$$ 스텝 rollout으로 모은 $$NT$$ 개 샘플을 $$K$$ 에폭 동안 재사용할 수 있다. 이것이 Vanilla PG 대비 핵심적인 샘플 효율 개선이다
- **구현이 한 줄로 요약된다.** conjugate gradient·Fisher-vector product·line search를 모두 제거하고 `min(r*A, clip(r,1-ε,1+ε)*A)` 를 loss로 쓰면 끝이다. Adam·dropout·파라미터 공유를 그대로 쓸 수 있다

## 4. Adaptive KL Penalty (두 번째 변형)

PPO의 또 다른 변형은 KL을 penalty로 쓰되 계수 $$\beta$$ 를 자동 조정한다.

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}(\cdot \mid s_t),\, \pi_\theta(\cdot \mid s_t)]\right]
$$

업데이트 후 실제 KL $$d$$ 를 측정해 $$\beta$$ 를 갱신한다.

$$
\beta \leftarrow \begin{cases} \beta/2 & d < d_\text{targ}/1.5 \\ \beta \times 2 & d > d_\text{targ}\times 1.5 \\ \beta & \text{otherwise} \end{cases}
$$

지수적 조정을 택한 이유는 초기값과 무관하게 빠르게 수렴하기 위해서다(PID 제어기의 적분항과 유사). 다만 **한 이터레이션 내에서는 $$\beta$$ 가 고정된 채 미니배치 SGD가 진행되어**, KL이 목표를 넘어도 즉각 반응하지 못한다. 반면 clipping은 매 step 자동 보호된다 — 실험에서 Adaptive KL이 Clipping에 뒤지는 주 원인으로 추정된다.

## 5. 전체 알고리즘

최종 학습 objective는 세 항의 결합이다.

$$
L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\left[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)\right]
$$

- $$L^{VF}_t = (V_\theta(s_t) - V_t^{targ})^2$$ : value function의 squared-error loss
- $$S[\pi_\theta](s_t)$$ : entropy bonus, 탐험 장려
- $$c_1, c_2$$ : 각 항의 가중치

어드밴티지는 GAE로 계산한다.

$$
\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$\lambda$$ 는 bias-variance tradeoff를 제어한다($$\lambda=0.95$$ 채택). 이 $$\hat{A}_t$$ 는 $$\theta_\text{old}$$ 로 rollout할 때 한 번 계산되고 $$K$$ 에폭 동안 재사용된다.

```text
PPO (Actor-Critic Style):
for each iteration:
  for actor = 1..N (병렬 환경):
    π_θ_old로 T 스텝 상호작용 → (s, a, r) 수집
  GAE로 advantage Â_t 추정
  K 에폭 반복:
    미니배치 M 랜덤 샘플링
    Adam으로 L^{CLIP+VF+S} 최대화
  θ_old ← θ
```

한 가지 미묘한 점: $$K$$ 에폭 동안 후반 미니배치는 이미 달라진 $$\theta$$ 에서 gradient를 계산하지만, $$\hat{A}_t$$ 는 $$\theta_\text{old}$$ 의 rollout에서 얻은 값이다. $$K$$ 가 클수록 이 불일치에서 오는 편향이 커지고, clipping이 그 크기를 제한하는 이중 역할을 한다.

## 6. 실험

### 6.1 Surrogate objective 비교 (MuJoCo)

7개 연속 제어 환경, 3시드, 1M 스텝. 랜덤 정책을 0, 최고 점수를 1로 정규화.

| 알고리즘 | 정규화 점수 |
| --- | --- |
| No clipping or penalty | −0.39 |
| Clipping $$\epsilon=0.1$$ | 0.76 |
| **Clipping $$\epsilon=0.2$$** | **0.82** |
| Clipping $$\epsilon=0.3$$ | 0.70 |
| Adaptive KL ($$d_\text{targ}=0.01$$) | 0.74 |
| Fixed KL ($$\beta=1$$) | 0.71 |

No clipping이 −0.39(랜덤보다 나쁨)를 기록한 건 HalfCheetah에서 정책이 붕괴했기 때문이다. 같은 데이터로 unconstrained 다중 업데이트가 얼마나 위험한지를 실증한다. $$\epsilon=0.2$$ 가 최적이며, $$\epsilon \in \{0.1, 0.2, 0.3\}$$ 모두 0.70\~0.82로 하이퍼파라미터에 비교적 둔감하다.

### 6.2 연속 제어에서 타 알고리즘 비교

같은 MuJoCo 7환경에서 A2C, A2C+TR, CEM, PPO, Vanilla PG, TRPO를 비교한다. PPO가 거의 모든 환경에서 1위 또는 그에 준하는 성능을 보인다. TRPO와 비교하면 유사하거나 PPO가 앞서면서, 구현은 훨씬 단순하다.

### 6.3 Roboschool Showcase

세 Humanoid 과제(50\~100M 스텝)에서 PPO 단독 학습 결과를 제시하며, 안정적으로 수렴한다. 타 알고리즘과의 비교 없이 PPO의 학습 곡선만 보여주는 쇼케이스다.

### 6.4 Atari 49게임

40M 프레임, 3시드. A2C·ACER·PPO의 "승리 횟수":

| 측정 기준 | A2C | ACER | PPO | Tie |
| --- | --- | --- | --- | --- |
| 전체 학습 평균(빠른 학습) | 1 | 18 | **30** | 0 |
| 마지막 100 에피소드(최종) | 1 | **28** | 19 | 1 |

두 기준의 결과가 뒤바뀐다. PPO는 **초기 학습 속도**(빠른 학습)에서 30승으로 앞서지만, **최종 성능**에서는 ACER가 앞선다. ACER는 off-policy로 experience replay를 사용해 데이터를 재사용하고, PPO는 on-policy라 $$NT$$ 샘플을 $$K$$ 에폭 뒤 버린다는 차이가 있다.

## 7. 결론

논문은 PPO를 **매 정책 업데이트마다 여러 에폭의 확률적 경사 상승(SGD)을 수행하는 정책 최적화 기법군**으로 정리한다. 저자가 밝힌 핵심은 다음과 같다.

- **Trust-region 기법(TRPO)의 안정성과 신뢰성**을 갖는다.
- 그러면서 **구현은 훨씬 단순하다** — vanilla policy gradient 구현에서 몇 줄만 바꾸면 된다.
- **더 일반적인 상황에 적용 가능하다** — 예컨대 정책과 가치 함수가 파라미터를 공유하는 아키텍처에서도 쓸 수 있다.
- 실험 전반에서 **더 나은 종합 성능**을 보인다.

## 참고: 도메인별 하이퍼파라미터

| 파라미터 | MuJoCo | Roboschool | Atari |
| --- | --- | --- | --- |
| Horizon $$T$$ | 2048 | 512 | 128 |
| 에폭 $$K$$ | 10 | 15 | 3 |
| 미니배치 | 64 | 4096 | 256 |
| $$\gamma$$ / $$\lambda$$ | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 |
| Adam stepsize | $$3\times10^{-4}$$ | adaptive | $$2.5\times10^{-4}\times\alpha$$ |
| Clipping $$\epsilon$$ | 0.2 | — | $$0.1\times\alpha$$ |
| 액터 $$N$$ | — | 32\~128 | 8 |
