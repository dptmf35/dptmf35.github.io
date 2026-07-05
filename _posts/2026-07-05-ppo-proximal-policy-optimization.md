---
title: "[논문리뷰] PPO: Proximal Policy Optimization Algorithms (2017)"
excerpt: "TRPO의 안정성을 1차 최적화만으로 달성한 PPO. clipped surrogate objective의 설계 의도부터 실험, 강·약점까지 정리한다."
date: 2026-07-05
categories:
  - 논문리뷰
  - RL
tags:
  - PPO
  - reinforcement learning
  - policy gradient
  - TRPO
  - RLHF
toc: true
toc_sticky: true
---

> **TRPO가 주는 안정성을, 1차 최적화(Adam)만으로 달성할 수 있는가?** 이 한 문장이 PPO의 출발점이다.
> 답은 "확률 비율을 클리핑하면 된다"이고, 이 단순함이 PPO를 RL의 사실상 표준으로 만들었다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347) |
| 저자 | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov |
| 소속 / 연도 | OpenAI, 2017 (arXiv:1707.06347) |
| 분야 | 강화학습 / 정책 경사 최적화 |

## 1. 배경: PG → TRPO → PPO

2017년 당시 딥RL의 주류 세 접근은 각기 다른 벽에 막혀 있었다.

- **Deep Q-learning**: 이산 행동 공간에 특화된 구조라, 로봇 제어처럼 연속·고차원 행동 공간에 적용하기 어렵다
- **Vanilla policy gradient(A2C/A3C)**: 구현은 단순하지만 수집한 데이터로 **딱 한 번만** gradient 업데이트를 하므로 샘플 효율이 낮다. 같은 데이터로 여러 번 업데이트하면 정책이 붕괴(policy collapse)한다
- **TRPO**: policy update 시 KL divergence를 명시적 제약으로 두어 **monotonic improvement를 보장**한다. 그러나 2차 최적화(conjugate gradient + Fisher-vector product + line search)가 필요해 구현이 복잡하고, dropout·파라미터 공유 아키텍처와 호환되지 않는다

PPO는 TRPO의 안정성을 유지하되 복잡도를 걷어내는 것을 목표로 한다.

![Vanilla PG는 안정성 문제, TRPO는 복잡도 문제, PPO는 clipping으로 둘을 동시에 해결](/assets/images/ppo-evolution.svg)

## 2. 배경 수식: Policy Gradient와 TRPO

### Policy Gradient

기대 누적 보상 $J(\pi_\theta) = \mathbb{E}_\tau[\sum_t r_t]$ 를 최대화하며, gradient는 다음과 같다.

$$
\hat{g} = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\,\hat{A}_t\right]
$$

여기서 $\hat{A}_t$ 는 어드밴티지 추정값으로, 행동 $a_t$ 가 기준선(value function)보다 얼마나 좋은지를 나타낸다. Vanilla PG의 핵심 제약은 **같은 샘플로 한 번만 업데이트**해야 한다는 것이다. 여러 번 쓰면 $\pi_\theta$ 와 샘플을 만든 $\pi_{\theta_\text{old}}$ 의 괴리가 커져 정책이 붕괴한다.

### TRPO

확률 비율 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 를 도입하면, TRPO가 최대화하는 CPI(Conservative Policy Iteration) objective는:

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right]
$$

TRPO는 이를 KL 제약 하에 최적화한다.

$$
\underset{\theta}{\text{maximize}}\; L^{CPI}(\theta) \quad \text{s.t.} \quad \hat{\mathbb{E}}_t\left[\text{KL}[\pi_{\theta_\text{old}}(\cdot|s_t),\, \pi_\theta(\cdot|s_t)]\right] \leq \delta
$$

Kakade & Langford(2002)의 이론으로 단조 향상이 보장되지만, 제약을 만족시키는 2차 최적화가 구현을 무겁게 만든다.

## 3. 핵심: Clipped Surrogate Objective

PPO는 KL 제약 대신 **클리핑**으로 정책 변화를 억제한다.

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \operatorname{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t\right)\right]
$$

핵심 설계는 $\min$ 이다. 클리핑한 값과 안 한 값 중 **더 비관적인(pessimistic) 값**을 택한다. 경우의 수로 분해하면 의도가 드러난다.

| 조건 | $r_t$ 상태 | $\min$ 선택 | 효과 |
| --- | --- | --- | --- |
| $\hat{A}_t>0$, $r_t \le 1+\epsilon$ | 정상 | $r_t\hat{A}_t$ (unclipped) | 정상 학습 |
| $\hat{A}_t>0$, $r_t > 1+\epsilon$ | 과도 증가 | $(1+\epsilon)\hat{A}_t$ (clipped) | gradient 차단 |
| $\hat{A}_t<0$, $r_t \ge 1-\epsilon$ | 정상 | $r_t\hat{A}_t$ (unclipped) | 정상 학습 |
| $\hat{A}_t<0$, $r_t < 1-\epsilon$ | 과도 감소 | $(1-\epsilon)\hat{A}_t$ (clipped) | gradient 차단 |

결과적으로 objective가 **개선되는** 방향으로 $r_t$ 가 범위를 벗어나면 이득을 무시하고, **악화되는** 방향으로는 끝까지 페널티를 주는 **단방향 보수주의(one-sided conservatism)** 가 구현된다.

![A>0이면 상승 후 평탄, A<0이면 평탄 후 하강 — 클리핑 밖에서 gradient가 0이 된다](/assets/images/ppo-clip-objective.svg)

### 왜 단순 클리핑이 아니라 min인가

단순히 $\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t$ 만 쓰면, $r_t > 1+\epsilon$ 이면서 $\hat{A}_t < 0$ 인 상황 — 즉 **나쁜 행동의 확률이 과도하게 높아지는** 경우 — 에도 gradient를 차단해 나쁜 방향으로의 이탈을 방치한다. $\min$ 을 쓰면 이 경우 unclipped 값이 더 작으므로 페널티가 온전히 전달된다.

머신러닝 관점에서 $L^{CLIP}$ 은 **확률 비율 공간에서의 암묵적 근위(proximal) 정규화**로 볼 수 있다. $\ell_2$ 정규화가 파라미터 변화를 제한하듯, clipping은 정책 자체의 변화를 제한한다 — TRPO의 명시적 KL 제약과 기능적으로 유사하되 훨씬 단순하다.

### 왜 중요한가

- **다중 에폭 미니배치 업데이트가 안전해진다.** clipping이 매 step마다 $r_t$ 범위를 제한하므로, $N$ 개 병렬 액터가 $T$ 스텝 rollout으로 모은 $NT$ 개 샘플을 $K$ 에폭 동안 재사용할 수 있다. 이것이 Vanilla PG 대비 핵심적인 샘플 효율 개선이다
- **구현이 한 줄로 요약된다.** conjugate gradient·Fisher-vector product·line search를 모두 제거하고 `min(r*A, clip(r,1-ε,1+ε)*A)` 를 loss로 쓰면 끝이다. Adam·dropout·파라미터 공유를 그대로 쓸 수 있다

이 단순성이 PPO를 이후 수년간 RL의 표준으로 만들었고, 특히 InstructGPT(2022) 이후 **RLHF를 통한 LLM 훈련**의 핵심 알고리즘으로 자리잡는 결정적 이유가 되었다.

## 4. Adaptive KL Penalty (두 번째 변형)

PPO의 또 다른 변형은 KL을 penalty로 쓰되 계수 $\beta$ 를 자동 조정한다.

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}(\cdot|s_t),\, \pi_\theta(\cdot|s_t)]\right]
$$

업데이트 후 실제 KL $d$ 를 측정해 $\beta$ 를 갱신한다.

$$
\beta \leftarrow \begin{cases} \beta/2 & d < d_\text{targ}/1.5 \\ \beta \times 2 & d > d_\text{targ}\times 1.5 \\ \beta & \text{otherwise} \end{cases}
$$

지수적 조정을 택한 이유는 초기값과 무관하게 빠르게 수렴하기 위해서다(PID 제어기의 적분항과 유사). 다만 **한 이터레이션 내에서는 $\beta$ 가 고정된 채 미니배치 SGD가 진행되어**, KL이 목표를 넘어도 즉각 반응하지 못한다. 반면 clipping은 매 step 자동 보호된다 — 실험에서 Adaptive KL이 Clipping에 뒤지는 주 원인으로 추정된다.

## 5. 전체 알고리즘

최종 학습 objective는 세 항의 결합이다.

$$
L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\left[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t)\right]
$$

- $L^{VF}_t = (V_\theta(s_t) - V_t^{targ})^2$ : value function의 squared-error loss
- $S[\pi_\theta](s_t)$ : entropy bonus, 탐험 장려
- $c_1, c_2$ : 각 항의 가중치

어드밴티지는 GAE로 계산한다.

$$
\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$\lambda$ 는 bias-variance tradeoff를 제어한다($\lambda=0.95$ 채택). 이 $\hat{A}_t$ 는 $\theta_\text{old}$ 로 rollout할 때 한 번 계산되고 $K$ 에폭 동안 재사용된다.

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

한 가지 미묘한 점: $K$ 에폭 동안 후반 미니배치는 이미 달라진 $\theta$ 에서 gradient를 계산하지만, $\hat{A}_t$ 는 $\theta_\text{old}$ 의 rollout에서 얻은 값이다. $K$ 가 클수록 이 불일치에서 오는 편향이 커지고, clipping이 그 크기를 제한하는 이중 역할을 한다.

## 6. 실험

### 6.1 Surrogate objective 비교 (MuJoCo)

7개 연속 제어 환경, 3시드, 1M 스텝. 랜덤 정책을 0, 최고 점수를 1로 정규화.

| 알고리즘 | 정규화 점수 |
| --- | --- |
| No clipping or penalty | −0.39 |
| Clipping $\epsilon=0.1$ | 0.76 |
| **Clipping $\epsilon=0.2$** | **0.82** |
| Clipping $\epsilon=0.3$ | 0.70 |
| Adaptive KL ($d_\text{targ}=0.01$) | 0.74 |
| Fixed KL ($\beta=1$) | 0.71 |

No clipping이 −0.39(랜덤보다 나쁨)를 기록한 건 HalfCheetah에서 정책이 붕괴했기 때문이다. 같은 데이터로 unconstrained 다중 업데이트가 얼마나 위험한지를 실증한다. $\epsilon=0.2$ 가 최적이며, $\epsilon \in \{0.1, 0.2, 0.3\}$ 모두 0.70\~0.82로 하이퍼파라미터에 비교적 둔감하다.

### 6.2 연속 제어에서 타 알고리즘 비교

같은 MuJoCo 7환경에서 A2C, A2C+TR, CEM, PPO, Vanilla PG, TRPO를 비교한다. PPO가 거의 모든 환경에서 1위 또는 그에 준한다. 가장 의미 있는 비교는 **PPO vs TRPO** 로, 유사하거나 PPO가 우세하면서 구현 복잡도는 훨씬 낮다 — PPO의 실용적 가치를 가장 잘 보여준다.

### 6.3 Roboschool Showcase

세 Humanoid 과제(50\~100M 스텝)에서 PPO 단독 학습으로 안정적 수렴을 보인다. 다만 타 알고리즘과 비교 없이 단독 결과만 제시해, 제목대로 정량 증명보다 **질적 데모** 성격이 강하다.

### 6.4 Atari 49게임

40M 프레임, 3시드. A2C·ACER·PPO의 "승리 횟수":

| 측정 기준 | A2C | ACER | PPO | Tie |
| --- | --- | --- | --- | --- |
| 전체 학습 평균(빠른 학습) | 1 | 18 | **30** | 0 |
| 마지막 100 에피소드(최종) | 1 | **28** | 19 | 1 |

두 기준이 역전되는 게 흥미롭다. PPO는 **초기 학습 속도**에서 압도적(30승)이지만 **최종 성능**은 ACER에 뒤진다. 원인은 ACER의 experience replay다 — off-policy로 과거 데이터를 재사용해 같은 프레임 수에서 더 많은 학습이 누적된다. PPO는 on-policy라 $NT$ 샘플을 $K$ 에폭 후 버리므로 후반 누적 효율에서 밀린다.

## 7. 강점

1. **구현 단순성과 범용성** — 단 하나의 수식 변경으로 TRPO의 무거운 2차 최적화를 대체한다. dropout·recurrent·policy-value 공유 아키텍처와 자연스럽게 결합되어, 연구자가 알고리즘 대신 **아키텍처 혁신**에 집중하게 했다
2. **샘플 재사용의 실증적 정당화** — no-clipping의 −0.39가 보여주듯, clipping이 $K$ 에폭 반복을 안전하게 만든다
3. **단방향 보수주의** — $\min$ 설계로 개선 방향의 과도 이득만 무시하고 악화 방향은 끝까지 페널티(3절 참고)
4. **하이퍼파라미터 견고성** — $\epsilon$ 세 값 모두 유사 성능, 새 환경에 세밀한 튜닝 없이 작동
5. **광범위한 실험** — MuJoCo 7 + Atari 49 + 3D Humanoid에 걸쳐 일관된 경쟁력

## 8. 약점

1. **이론적 보장의 부재** — TRPO의 monotonic improvement 같은 증명이 없다. $L^{CLIP}$ 이 실제 성능의 lower bound라는 증명 없이 "경험적으로" 작동함을 인정하는 수준이다
2. **다중 에폭 편향 분석 부재** — $K$ 에폭에서 $\hat{A}_t$(old)와 gradient($\theta$)의 불일치로 생기는 편향을 정량 분석하지 않는다($K$ 도 환경마다 다름: MuJoCo 10, Atari 3, Roboschool 15)
3. **Roboschool 비교 기준 부재** — PPO 단독 곡선만 제시
4. **on-policy의 구조적 샘플 비효율** — 데이터를 $K$ 에폭 후 버려, Atari 최종 성능에서 ACER에 뒤진다
5. **Annealing의 미분석** — Atari에서 $\alpha$ 를 1→0으로 줄여 lr·$\epsilon$ 을 동시 감소시키는데, ablation이 없어 PPO 자체 기여와 분리 불가
6. **파라미터 공유 이점 미검증** — 강점으로 주장하지만 MuJoCo 실험은 공유하지 않는 설정을 사용

## 9. 총평 & Takeaway

읽을 가치는 이론적 깊이가 아니라 **실질적 영향력과 아이디어의 명확성**에 있다. PPO는 잘 작동하고, 이해하기 쉽고, 구현하기 쉽다. 2022년 InstructGPT·ChatGPT의 RLHF에 채택된 것은 우연이 아니다 — 언어 모델의 보상 기반 fine-tuning이라는 전혀 다른 문제에서도 구현 단순성·파라미터 공유·안정적 학습이 그대로 유효했다.

**핵심 Takeaway 3가지**

1. **제약보다, 페널티보다, 클리핑이 실용적이다** — 이론적 우아함보다 실용적 구현 가능성이 더 큰 영향력을 가질 수 있다
2. **샘플 재사용의 핵심은 "얼마나 많이"가 아니라 "얼마나 안전하게"다** — 정규화가 과적합 방지를 넘어 **최적화 경로 자체를 안정화**한다는 원리의 사례
3. **단순한 알고리즘 + 광범위한 실험이, 복잡한 알고리즘 + 좁은 증명보다 영향력이 클 수 있다**

## 참고: 도메인별 하이퍼파라미터

| 파라미터 | MuJoCo | Roboschool | Atari |
| --- | --- | --- | --- |
| Horizon $T$ | 2048 | 512 | 128 |
| 에폭 $K$ | 10 | 15 | 3 |
| 미니배치 | 64 | 4096 | 256 |
| $\gamma$ / $\lambda$ | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 |
| Adam stepsize | $3\times10^{-4}$ | adaptive | $2.5\times10^{-4}\times\alpha$ |
| Clipping $\epsilon$ | 0.2 | — | $0.1\times\alpha$ |
| 액터 $N$ | — | 32\~128 | 8 |
