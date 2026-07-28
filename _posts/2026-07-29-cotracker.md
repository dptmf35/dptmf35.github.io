---
title: "[논문리뷰] CoTracker: It is Better to Track Together (ECCV 2024)"
excerpt: "track 간 attention과 support point로 여러 점을 jointly 추적해 occlusion에 강한 transformer 기반 point tracker CoTracker 논문 리뷰"
categories:
  - 논문리뷰
  - CV
tags:
  - CoTracker
  - point tracking
  - optical flow
  - motion estimation
  - transformer
toc: true
toc_sticky: true
---

> **점을 하나씩 추적하지 말고, 같이 추적하라.**
> CoTracker는 track 사이의 attention으로 여러 점을 jointly 추적하는 transformer 기반 point tracker다. 점들 사이의 통계적 의존성을 활용해 occlusion 중인 점, 심지어 화면 밖으로 나간 점까지 추적하고, proxy token으로 단일 GPU에서 70k개 점을 동시에 추적한다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [CoTracker: It is Better to Track Together](https://arxiv.org/abs/2307.07635) |
| 저자 | Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht |
| 소속 / 연도 | Meta AI, Visual Geometry Group(University of Oxford), ECCV 2024 (arXiv:2307.07635) |
| 분야 | CV / point tracking |
| 코드 | [GitHub](https://github.com/facebookresearch/co-tracker) |

## 1. Introduction

동적 물체와 움직이는 카메라가 있는 비디오에서 점 대응(point correspondence)을 추정하는 문제에는 두 가지 변형이 있다. **optical flow**는 프레임 안 모든 점의 속도를 추정한다. 모든 점에 대해 jointly 계산하지만, 움직임은 무한소 거리에서만 예측된다. **point tracking**은 점의 움직임을 긴 시간에 걸쳐 추정한다. 효율을 위해 보통 sparse한 점 집합에 집중하고, 점들을 **통계적으로 독립**으로 취급한다. TAPIR·PIPs++ 같은 최신 기법도 transformer 기반 architecture로 occlusion 속에서도 점을 추적하지만, 점들을 여전히 독립적으로 다룬다. 그러나 점들은 (예를 들어 같은 물체에 속해 있어서) 강한 통계적 의존성을 갖는 경우가 많고, 여기에 개선의 여지가 있다.

논문은 추적되는 점들 사이의 의존성을 고려하면 tracking 성능이 크게 좋아진다고 가정하고, 이를 확인하기 위해 **CoTracker**를 제안한다. transformer 기반 architecture와 track 사이의 attention으로 매우 많은 수의 track을 동시에 jointly 추정하는 tracker다. ablation에서 보이듯 joint tracking은 특히 점이 가려졌을 때 정확도를 크게 높인다. 나아가 사용자가 요청한 것보다 **더 많은 점을 추가로 추적하면 품질이 더 좋아진다**는 사실도 발견한다. 이 추가 **support point**들은 tracker의 context를 넓혀 주는데, 이는 visual object tracking에서 흔히 쓰는 context 개념을 point tracking으로 가져온 것이다.

CoTracker는 joint tracking 외에도 몇 가지 설계 혁신을 담는다. 네트워크는 시간과 track 집합을 두 축으로 하는 2차원 token 표현 위에서 sliding window 방식으로 동작하는 transformer다. 적절한 self-attention 연산으로 window 길이 동안 각 track 전체를 고려하고, track 사이에 정보를 교환한다. 다만 track이 많으면 attention 비용이 비싸지므로, register와 비슷하게 동작하면서 메모리 복잡도를 줄이는 **proxy token** 개념을 point tracking에 도입한다. proxy token은 소수의 추가 track처럼 처리되고, track 사이의 비싼 self-attention을 track과 proxy 사이의 효율적인 cross-attention으로 바꿔 준다. 그 결과 CoTracker는 단일 GPU에서 거의 dense한 track 집합을 jointly 추적할 수 있다.

CoTracker는 비교적 짧은 프레임 window 위에서 동작하는 online tracker로 설계됐다. window 안에서 track은 query된 점으로 초기화되고, 네트워크는 transformer를 반복 적용하며 초기 추정을 점진적으로 정제한다. window들은 부분적으로 겹치며 서로 통신하는데, 이는 recurrent network와 비슷하다. 이 recurrent한 적용을 **unrolled training**으로 최적화한다. 이 개념 역시 recurrent network와 visual object tracking에서 point tracking으로 가져온 것이다. 이렇게 joint tracking과 recurrent tracking을 결합하면 한 window보다 긴 occlusion을 통과해서도 점을 추적할 수 있다.

모델은 synthetic 데이터셋 TAP-Vid-Kubric으로 학습하고 TAP-Vid-DAVIS, TAP-Vid-RGB-Stacking, PointOdyssey, Dynamic Replica로 평가한다. 단일 점 추적에서도 잘 동작하고 점 그룹 추적에서 특히 뛰어나, 여러 벤치마크에서 기존 tracker 대비 state-of-the-art를 달성한다. 요약하면 기여는 다음과 같다.

- attention 메커니즘으로 추적 점들 사이에 정보를 공유하는 **joint point tracking** 개념 도입
- point tracking에 추가 context를 제공하는 **support point** 제안
- **proxy token**으로 모델의 메모리 복잡도 절감
- tracking·occlusion 정확도를 더 높이는 **unrolled training** 전략 제안

![배경 점은 cyan, 전경 점은 magenta로 칠한 비교로, 독립 추적은 자전거 하강 후 발 위의 점을 잃지만 joint 추적은 유지하는 모습](/assets/images/cotracker-fig2.jpg)
*[Figure 2] 독립(non-joint) tracking(윗줄)과 joint tracking(아랫줄)의 비교*

## 2. Related Work

**Optical flow.** optical flow는 dense한 순간 움직임을 근사한다. 원래 brightness constancy 방정식으로 접근했지만 FlowNet 이후 딥러닝으로 다뤄졌다. DCFlow가 4D cost volume 계산을 도입해 이후 대부분의 연구에서 쓰였고, 그중 RAFT는 점진적 flow update를 도입해 여러 후속 연구에 영감을 줬다. CoTracker도 4D cost volume과 반복 update를 쓰지만 tracking에 적용한다. transformer도 optical flow에 적용됐다. Flowformer는 RAFT에서 영감을 받아 4D cost volume을 tokenize했고, GMFlow는 update 네트워크를 self-attention 기반 softmax 정제로 대체했다. optical flow를 시간에 걸쳐 적분하면 tracking에 쓸 수 있지만 drift 형태로 오차가 누적되므로, 더 긴 시간 범위에서 점을 추적하는 CoTracker 같은 architecture가 필요하다.

**Multi-frame optical flow.** optical flow를 여러 프레임으로 확장한 연구들이 있다. 초기에는 시간적 일관성을 위해 Kalman filtering을 썼고, 현대 방법들은 dense flow를 만든다. RAFT는 warm-start 방식으로 multi-frame 추정에 적용될 수 있고, VideoFlow는 3~5 연속 프레임으로 확장했으며, MFT는 먼 프레임 사이의 optical flow를 추정해 가장 신뢰할 만한 flow chain을 고른다. 그러나 이들은 장기 tracking용으로 설계되지 않았고 오래 가려진 점을 고려하지 않는다.

**Visual object tracking.** 딥러닝 이전에 hand-crafted joint tracker들이 제안됐지만, CoTracker처럼 딥네트워크로 joint tracking을 다룬 연구는 드물다. occlusion·외형 변화·시간적 prior를 다뤄 온 multiple object tracking과 약하게 관련되고, source·target 이미지 영역에서 feature를 추출·결합하는 최근 point tracker들은 visual object tracking에서 영감을 받았다. 다만 이 논문의 초점은 물체가 아니라 배경 점을 포함한 **점**의 추적이다.

**Tracking any point.** Particle Video가 비디오에서 임의 점을 추적하는 문제(TAP)를 도입했고, PIPs는 sliding window로 track을 예측하고 점이 마지막으로 보인 프레임에서 재시작해 occlusion을 더 잘 통과하는 모델을 제안했다. 그러나 PIPs는 window 길이를 넘어서는 점을 추적하지 못한다. TAP-Vid는 새 벤치마크와 단순한 baseline을 제안했다. 원조 Particle Video는 점을 jointly 추적하지만, PIPs와 TAP-Vid는 점을 독립적으로(병렬로) 추적한다. TAPIR는 TAP-Vid식 matching과 PIPs식 정제를 결합한 2단계 feed-forward tracker로 성능을 더 끌어올렸다. PointOdyssey는 PIPs를 단순화한 PIPs++와 장기 tracking 벤치마크를 제안했지만 PIPs++ 역시 점을 독립적으로 추적한다. OmniMotion은 비디오마다 volumetric 표현을 최적화하는데, test-time 최적화가 필요해 비용상 online 응용에 적합하지 않다. tracker와 optical flow 모델은 실측 annotation이 어려워 synthetic 데이터셋으로 학습하는 경우가 많고, 이 방식이 실세계 데이터로 일반화됨이 확인돼 왔다.

## 3. CoTracker

목표는 $$T$$ 장의 RGB 프레임 $$I_t \in \mathbb{R}^{3 \times H \times W}$$ 로 이뤄진 비디오 $$V = (I_t)_{t=1}^{T}$$ 전체에서 2D 점들을 추적하는 것이다. tracker는 $$N$$ 개의 point track

$$
P^i_t = (x^i_t, y^i_t) \in \mathbb{R}^2, \quad t = t^i, \dots, T, \quad i = 1, \dots, N
$$

을 예측한다. 여기서 $$t^i \in \{1, \dots, T\}$$ 는 track $$i$$ 가 시작되는 시각이다. tracker는 각 프레임에서 점이 보이는지 가려졌는지를 나타내는 visibility flag $$v^i_t \in \{0, 1\}$$ 도 예측한다. 과제 정의를 명확히 하기 위해 각 점은 track 시작 시점에 보인다고($$v^i_{t^i} = 1$$) 가정한다. 즉 tracker는 비디오 $$V$$ 와 $$N$$ 개 track의 시작 위치·시각 $$(P^i_{t^i}, t^i)_{i=1}^N$$ 을 입력받아, 유효한 모든 시각 $$t \geq t^i$$ 에 대해 track 위치와 visibility의 추정치 $$(\hat{P}^i_t = (\hat{x}^i_t, \hat{y}^i_t), \hat{v}^i_t)$$ 를 출력한다.

### 3.1 Transformer formulation

tracker는 transformer 네트워크 $$\Psi: G \to O$$ 로 구현한다. 이 transformer의 목표는 track의 초기 추정을 개선하는 것이다. track은 track $$i = 1, \dots, N$$ 과 시각 $$t = 1, \dots, T$$ 마다 하나씩 있는 입력 token $$G^i_t$$ 의 grid $$G$$ 로 인코딩되고, 갱신된 track은 대응하는 출력 token $$O^i_t$$ 의 grid $$O$$ 로 표현된다.

![CNN feature, 4D correlation, token grid를 거쳐 시간·track 축 attention을 반복 적용하는 CoTracker architecture](/assets/images/cotracker-fig3.png)
*[Figure 3] convolutional feature와 correlation feature로 token을 구성해 transformer를 반복 적용하는 CoTracker architecture*

**Image features.** 각 프레임 $$I_t$$ 에서 end-to-end로 학습되는 CNN으로 dense한 $$d$$ 차원 appearance feature $$\phi(I_t) \in \mathbb{R}^{d \times \frac{H}{k} \times \frac{W}{k}}$$ 를 추출한다. 효율을 위해 해상도를 $$k = 4$$ 배 줄인다. 또 stride $$s = 1, \dots, S$$ 로 축소한 feature $$\phi_s(I_t)$$ 를 $$S = 4$$ 개 scale로 두는데, base feature에 average pooling을 적용해 얻는다.

**Track features.** track의 appearance는 feature 벡터 $$Q^i_t \in \mathbb{R}^d$$ 로 표현한다(track의 외형 변화를 수용하도록 시간에 의존한다). 시작 위치에서 sampling한 image feature를 broadcasting해 초기화하고, 이후 네트워크가 갱신한다.

**Spatial correlation features.** track과 이미지의 matching을 돕기 위해 RAFT와 비슷한 correlation feature $$C^i_t \in \mathbb{R}^S$$ 를 쓴다. $$C^i_t$$ 는 track feature $$Q^i_t$$ 를 현재 track 위치 추정 $$\hat{P}^i_t$$ 주변의 image feature $$\phi_s(I_t)$$ 와 비교해 얻는다. 구체적으로 내적들을 쌓아서 만든다.

$$
[C^i_t]_{s\delta} = \langle Q^i_t, \ \phi_s(I_t)[\hat{P}^i_t / ks + \delta] \rangle
$$

여기서 $$s = 1, \dots, S$$ 는 feature scale이고, $$\delta \in \mathbb{Z}^2$$, $$\|\delta\|_\infty \leq \Delta$$ 는 주변을 훑는 offset이다. 즉 각 scale에서 현재 추정 위치를 중심으로 한 $$(2\Delta+1) \times (2\Delta+1)$$ 격자 위의 image feature와 track feature의 유사도를 재는 것이다. image feature $$\phi_s(I_t)$$ 는 bilinear interpolation과 border padding으로 정수가 아닌 위치에서도 sampling한다. $$S = 4$$, $$\Delta = 3$$ 선택에서 $$C^i_t$$ 의 차원은 $$(2\Delta+1)^2 S = 196$$ 이다.

**Tokens.** 입력 token $$G(\hat{P}, \hat{v}, Q)$$ 는 track의 위치·visibility·appearance·correlation을 인코딩한다. positional encoding을 더한 feature들의 concatenation으로 주어진다.

$$
G^i_t = (\hat{P}^i_t - \hat{P}^i_1, \ \hat{v}^i_t, \ Q^i_t, \ C^i_t, \ \eta(\hat{P}^i_t - \hat{P}^i_1)) + \eta'(\hat{P}^i_1) + \eta'(t)
$$

마지막 성분 $$\eta(\hat{P}^i_t - \hat{P}^i_1)$$ 는 시각 $$t = 1$$ 의 초기 위치에 대한 track 위치의 sinusoidal positional encoding이다. 여기에 시작 위치 $$\hat{P}^i_1$$ 의 encoding $$\eta'(\hat{P}^i_1)$$ 과 시각 $$t$$ 의 encoding $$\eta'(t)$$ 를 적절한 차원으로 더한다. 첫 프레임에서의 점 위치와 그 프레임에 대한 상대 변위를 따로 인코딩하는 것이 유리했다고 한다. 출력 token $$O(\Delta\hat{P}, \Delta Q)$$ 는 위치와 appearance의 update를 담는다. 즉 $$O^i_t = (\Delta\hat{P}^i_t, \Delta Q^i_t)$$ 이다.

**Iterated transformer application.** track 추정을 점진적으로 개선하기 위해 transformer를 $$M$$ 번 적용한다. $$m = 0, 1, \dots, M$$ 을 추정의 index라 하고 $$m = 0$$ 은 초기화를 뜻한다. 각 update는

$$
O(\Delta\hat{P}, \Delta Q) = \Psi(G(\hat{P}^{(m)}, \hat{v}^{(0)}, Q^{(m)}))
$$

를 계산하고 $$\hat{P}^{(m+1)} = \hat{P}^{(m)} + \Delta\hat{P}$$, $$Q^{(m+1)} = Q^{(m)} + \Delta Q$$ 로 갱신한다. visibility mask $$\hat{v}$$ 는 반복 갱신하지 않고 마지막 transformer 적용 후 한 번만 $$\hat{v}^{(M)} = \sigma(W Q^{(M)})$$ 로 계산한다. 여기서 $$\sigma$$ 는 sigmoid 활성 함수, $$W$$ 는 학습된 가중치 행렬이다. visibility를 반복 갱신해도 성능이 좋아지지 않았는데, visibility 예측에는 먼저 정확한 위치 예측이 필요하기 때문으로 보인다. $$m = 0$$ 에서 위치·visibility·appearance 추정 $$\hat{P}^{(0)}$$, $$v^{(0)}$$, $$Q^{(0)}$$ 는 query 점의 초기 위치 $$P^i_{t^i}$$, visibility $$v^i_{t^i} = 1$$(보임), appearance $$\phi(I_{t^i})[P^i_{t^i}/k]$$ 를 모든 시각 $$t = 1, \dots, T$$ 로 broadcasting해 초기화한다.

### 3.2 Transformer architecture and proxy tokens

transformer $$\Psi$$ 는 시간 축과 track 축에 각각 작용하는 attention layer를 번갈아 쌓는다. attention을 시간과 track으로 factorize하면 복잡도가 $$O(N^2 T^2)$$ 에서 $$O(N^2 + T^2)$$ 로 줄어 계산이 감당 가능해진다. 그러나 $$N$$ 이 아주 크면 이 비용도 여전히 부담스럽다. 그래서 $$K$$ 개의 **proxy track**을 도입한다($$K \ll N$$ 은 hyper-parameter). proxy track은 학습된 고정 token으로, transformer 입력에서 '일반' track 목록에 이어 붙이고 출력에서 버린다.

시간 attention에서는 일반 track과 proxy track을 동일하게 처리한다. 반면 track attention에서는 일반 track이 서로를 attend하지 않고 **proxy만 cross-attend**해, 비용이 $$O(NK + K^2 + T^2)$$ 로 줄어든다. proxy는 그래프 신경망 가속 등에 자주 쓰이고, DETR는 transformer에서 학습 가능한 query를 쓰며 Perceiver IO는 dense 출력의 decoding에 쓴다. 이 논문의 proxy token은 다르다. register와 비슷하게 일반 track의 계산을 그대로 따라가며(shadow) 보조한다.

### 3.3 Windowed inference and unrolled training

tracking을 track window의 점진적 정제로 정식화하면 임의로 긴 비디오를 처리할 수 있다는 장점이 있다. 이전 window와 부분적으로 겹치는 track으로 다음 window를 초기화하면 된다.

architecture가 지원하는 최대 window 길이 $$T$$ 보다 긴 길이 $$T' > T$$ 의 비디오 $$V$$ 를 생각하자. 비디오 전체에서 점을 추적하기 위해 비디오를 길이 $$T$$, 겹침 $$T/2$$ 프레임의 window $$J = \lceil 2T'/T - 1 \rceil$$ 개로 나눈다($$T$$ 는 짝수로 가정한다). 위 첨자 $$(m, j)$$ 로 $$j$$ 번째 window에 대한 transformer의 $$m$$ 번째 적용을 나타내면, transformer 반복과 window에 걸친 $$M \times J$$ 개 추정량 $$(\hat{P}^{(m,j)}, \hat{v}^{(m,j)}, Q^{(m,j)})$$ 의 grid가 생긴다. $$m = 0$$, $$j = 1$$ 은 단일 window 경우처럼 초기화하고, transformer를 $$M$$ 번 적용해 추정 $$(M, 1)$$ 을 얻는다. 이것으로 추정 $$(0, 2)$$ 를 broadcasting해 초기화한다. 구체적으로 $$\hat{P}^{(0,2)}$$ 의 앞 $$T/2$$ 성분은 $$\hat{P}^{(M,1)}$$ 의 뒤 $$T/2$$ 성분의 복사본이고, $$\hat{P}^{(0,2)}$$ 의 뒤 $$T/2$$ 성분은 $$\hat{P}^{(M,1)}$$ 의 마지막 시각 성분의 복사본이다. $$\hat{v}^{(0,2)}$$ 도 같은 규칙으로 갱신하고, $$Q^{(0,j)}$$ 는 항상 track 시작 위치의 feature $$Q$$ 로 초기화한다. 이 과정을 추정 $$(M, J)$$ 를 얻을 때까지 반복한다.

![하나의 sliding window 안에서 M번의 반복 update로 track과 correlation feature가 갱신되는 과정](/assets/images/cotracker-fig4.png)
*[Figure 4] 한 sliding window 안에서 $$M$$ 번의 update로 point track $$\hat{P}^{(m)}$$ 을 정제하는 과정*

windowed transformer는 본질적으로 recurrent network처럼 동작하므로 **unrolled 방식으로 학습**한다. 구체적으로 transformer 반복 적용과 window에 걸쳐 합한 track 예측 오차를 최적화한다.

$$
\mathcal{L}_1(\hat{P}, P) = \sum_{j=1}^{J} \sum_{m=1}^{M} \gamma^{M-m} \| \hat{P}^{(m,j)} - P^{(j)} \|
$$

여기서 $$\gamma = 0.8$$ 은 이른 transformer update의 비중을 낮추는 할인 계수이고, $$P^{(j)}$$ 는 window $$j$$ 로 제한한 ground-truth 궤적이다(window 중간에서 시작하는 궤적은 뒤로 padding한다). 두 번째 loss는 visibility flag의 cross entropy

$$
\mathcal{L}_2(\hat{v}, v) = \sum_{j=1}^{J} \operatorname{CE}(\hat{v}^{(M,j)}, v^{(j)})
$$

다. 학습 때는 계산 비용 때문에 적당한 수의 window만 loss에 쓰지만, test 때는 windowed transformer 적용을 임의로 unroll할 수 있어 원칙적으로 어떤 길이의 비디오도 다룬다. joint tracking과 결합하면, unrolling 덕분에 한 sliding window보다 긴 occlusion을 통과해서도, 심지어 카메라 시야 밖의 점도 추적할 수 있다.

### 3.4 Support points

CoTracker는 여러 점을 jointly 추적하는 데서 이득을 얻는다. 응용에서는 보통 추적할 점이 여러 개지만, inference에서 몇 개 또는 단 하나의 점만 추적하고 싶은 경우도 있다. 이런 경우 사용자가 명시적으로 요청하지 않은 **추가 support point들을 함께 추적하는 것이 유리**함을 발견했다. support point의 구성(configuration)에 따라 성능이 조금씩 달라진다. "**global**" 전략에서는 support point가 이미지 전체에 걸친 regular grid를 이룬다. "**local**" 전략에서는 추적하려는 점 주위에 grid를 중심 배치해 모델이 그 이웃에 집중하게 한다. 이 패턴들은 inference에서만 쓰이고, context를 넣어 target 점의 추적 정확도를 높이는 용도다.

![단일 target 점, global grid, local grid, 그리고 둘을 결합한 support point 구성별 track 시각화](/assets/images/cotracker-fig5.jpg)
*[Figure 5] 추적 대상 점과 support point의 여러 구성*

## 4. Experiments

여러 표준 실사·synthetic tracking 벤치마크에서 CoTracker를 평가한다.

**Datasets and benchmarks.** 학습은 선행 연구와 같이 **TAP-Vid-Kubric**으로 한다. Kubric 엔진으로 생성한 24프레임 시퀀스로, 3D 강체들이 중력에 의해 떨어지고 튕기는 장면이다. point track은 주로 물체 위에서, 일부는 배경에서 무작위로 고른다. 평가는 실사 시퀀스 30개(약 100프레임)의 **TAP-Vid-DAVIS**와, 로봇이 물체를 옮기는 synthetic 시퀀스 50개(200~300프레임)의 **TAP-Vid-RGB-Stacking**으로 한다. 평가 protocol은 두 가지다. "**first**"는 각 점을 처음 보이는 프레임에서 한 번만 query하고, tracker는 인과적으로(미래 프레임에 대해서만) 동작해야 한다. "**strided**"는 5프레임마다 점을 query하고 양방향으로 추적한다. 대부분의 tracker(CoTracker, PIPs, PIPs++)는 인과적이므로 비디오와 그 역방향에 두 번 돌린다. 지표는 TAP-Vid 지표를 쓴다. **OA**(Occlusion Accuracy)는 이진 분류로 본 occlusion 예측 정확도, $$\delta_{\text{avg}}^{\text{vis}}$$ 는 보이는 점 중 1, 2, 4, 8, 16픽셀 안에서 추적된 비율을 threshold별로 평균한 값, **AJ**(Average Jaccard)는 위치와 occlusion 예측 정확도를 함께 재는 지표다. 이미지를 $$256 \times 256$$ 으로 가상 resize한 뒤 threshold를 적용하고, TAP-Vid 벤치마크는 보이는 점만 평가한다.

장기 tracking용 synthetic 벤치마크 **PointOdyssey**로도 평가한다. 수천 프레임 길이의 시퀀스 100개로, Kubric보다 훨씬 사실적인 장면에서 물체와 캐릭터가 움직인다. 여기서는 PointOdyssey로 학습·평가하고 $$\delta_{\text{avg}}^{\text{vis}}$$, $$\delta_{\text{avg}}^{\text{occ}}$$, $$\delta_{\text{avg}}$$ 를 보고한다. 뒤의 둘은 각각 가려진 점과 전체 점에 대한 같은 지표다(synthetic이라 보이지 않는 점의 실측 위치를 알 수 있다). 추적 오차가 50픽셀을 넘으면 실패로 보고, 실패까지의 평균 프레임 비율인 **Survival rate**도 보고한다. **Dynamic Replica**는 장기 tracking annotation이 있는 3D reconstruction용 synthetic 데이터셋으로, 사람·동물 articulated 모델의 300프레임 시퀀스 500개로 이뤄진다. Kubric으로 학습한 모델을 "valid" split에서 $$\delta_{\text{avg}}^{\text{vis}}$$ 와 $$\delta_{\text{avg}}^{\text{occ}}$$ 로 평가한다.

**Implementation details.** 6,000개의 TAP-Vid-Kubric 시퀀스($$T' = 24$$ 프레임)에서 sliding window 크기 $$T = 8$$ 로 50,000 iteration 학습하며 40시간이 걸린다. NVIDIA A100 80GB GPU 32개, GPU당 batch size 1을 쓴다. 학습 track은 물체 위에서 우선적으로 sampling한다. 학습 중에는 시퀀스의 첫 프레임 또는 중간 프레임에서 보이는 점들로 $$N = 768$$ 개 track의 batch를 구성해, 두 경우 모두 다루도록 학습한다. 같은 방식으로 PointOdyssey의 train split에서도 두 번째 버전의 CoTracker를 학습한다(길이 $$T' = 56$$ 시퀀스에서 무작위 sampling한 128개 track). 기술적 문제 하나는 새 점이 언제든 추가될 수 있어 window마다 track 수 $$N$$ 이 달라진다는 점이다. token 수를 바꾸면 학습 batching이 어려워지므로, 전체 추적 점 수만큼 token을 미리 할당하고 아직 쓰이지 않는 token은 masking으로 무시한다. 해상도도 성능에 영향을 준다. TAP-Vid 평가에서는 protocol대로 비디오를 $$256 \times 256$$ 으로 downsampling한 뒤, 공정성을 위해 각 tracker의 native 해상도(PIPs·CoTracker는 $$384 \times 512$$, PIPs++는 $$512 \times 896$$)로 resize해 돌린다.

### 4.1 Comparisons to the State of the Art

TAP-Vid-DAVIS와 TAP-Vid-RGB-Stacking에서 state-of-the-art tracker들과 비교한다(Table 1). 공정성을 위해 모든 벤치마크에서 ablation으로 찾은 동일한 support point 구성을 쓴다. PointOdyssey(Table 2)와 Dynamic Replica(Table 3)에서는 모든 target 점을 jointly 추적해 평가한다.

| Method | Train | DAVIS First AJ ↑ | $$\delta_{\text{avg}}^{\text{vis}}$$ ↑ | OA ↑ | DAVIS Strided AJ ↑ | $$\delta_{\text{avg}}^{\text{vis}}$$ ↑ | OA ↑ | RGB-S First AJ ↑ | $$\delta_{\text{avg}}^{\text{vis}}$$ ↑ | OA ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAP-Net | K | 33.0 | 48.6 | 78.8 | 38.4 | 53.1 | 82.3 | - | - | - |
| OmniMotion | - | 52.8 | 66.9 | 87.1 | 51.7 | 67.5 | 85.3 | 69.5 | 82.4 | **90.3** |
| PIPs | FT++ | 42.2 | 64.8 | 77.7 | 52.4 | 70.0 | 83.6 | - | 59.1 | - |
| MFT | K++ | 47.3 | 66.8 | 77.8 | 56.1 | 70.8 | 86.9 | - | - | - |
| PIPs++ | PO | - | 69.1 | - | - | 73.7 | - | - | 77.8 | - |
| TAPIR | K | 56.2 | 70.0 | 86.5 | 61.3 | 73.6 | 88.8 | 60.3 | 74.1 | 85.5 |
| CoTracker (Ours) | K | **62.2** | **75.7** | **89.3** | **65.9** | **79.4** | **89.9** | **71.6** | **83.3** | 89.6 |

*[Table 1] TAP-Vid 벤치마크의 "first"·"strided" protocol 비교(굵게 최고). Train 열의 K는 Kubric, K++는 Kubric+추가 데이터, FT++는 FlyingThings++, PO는 PointOdyssey. TAPIR·OmniMotion은 평가 때 전체 프레임에 한 번에 접근하지만 CoTracker는 causal sliding window로 online 처리하며, 움직임이 느린 RGB-S에서는 5프레임마다 하나씩 남겨 frame rate를 낮춰 평가*

| Method | $$\delta_{\text{avg}}$$ ↑ | $$\delta_{\text{avg}}^{\text{vis}}$$ ↑ | $$\delta_{\text{avg}}^{\text{occ}}$$ ↑ | Survival ↑ |
| --- | --- | --- | --- | --- |
| TAP-Net | 28.4 | - | - | 18.3 |
| PIPs | 27.3 | - | - | 42.3 |
| PIPs++† | 29.0 | 32.4 | 18.8 | 47.0 |
| CoTracker (Ours) | **30.2** | **32.7** | **24.2** | **55.2** |

*[Table 2] PointOdyssey 평가(모든 방법을 PointOdyssey로 학습, 굵게 최고). †는 공개 코드·모델로 얻은 결과*

| Method | $$\delta_{\text{avg}}$$ ↑ | $$\delta_{\text{avg}}^{\text{vis}}$$ ↑ | $$\delta_{\text{avg}}^{\text{occ}}$$ ↑ |
| --- | --- | --- | --- |
| TAP-Net | 45.5 | 53.3 | 20.0 |
| PIPs | 41.0 | 47.1 | 21.0 |
| PIPs++ | 55.5 | 64.0 | 28.5 |
| TAPIR | 56.8 | 66.1 | 27.2 |
| CoTracker (Ours) | **61.6** | **68.9** | **37.6** |

*[Table 3] Dynamic Replica 평가(굵게 최고). 가려진 점의 정확도 $$\delta_{\text{avg}}^{\text{occ}}$$ 에서 격차가 더 큼*

CoTracker는 TAP-Vid 벤치마크 대부분의 지표를 큰 폭으로 개선하며, synthetic Kubric에서 실사 DAVIS로 잘 일반화된다. tracking 정확도 $$\delta_{\text{avg}}^{\text{vis}}$$ 는 모든 경우에 개선된다. PointOdyssey에서는 window 크기가 8프레임뿐인데도 128프레임 sliding window를 쓰는 PIPs++보다 Survival rate가 높다. 이는 긴 sliding window 연쇄에 걸쳐 정보를 전파하도록 학습시키는 unrolled training의 힘을 보여준다. Table 2·3에서 $$\delta_{\text{avg}}^{\text{occ}}$$ 의 격차가 $$\delta_{\text{avg}}^{\text{vis}}$$ 보다 큰 것은, 점 그룹의 joint tracking 덕분에 CoTracker가 가려진 점 추적에 특히 뛰어남을 보여준다.

![자동차 장면에서 PIPs++는 점들이 차 앞부분에 붙어 버리고 TAPIR는 가려진 뒤 복구하지 못하지만 CoTracker는 정확히 추적하는 비교](/assets/images/cotracker-fig6.jpg)
*[Figure 6] PIPs++(윗줄)·TAPIR(가운데)·CoTracker(아랫줄)의 정성 비교*

### 4.2 Ablations

**Tracking together is better.** CoTracker의 핵심 동기인 joint tracking의 중요성을 확인한다(Table 4). tracker에서 track 사이의 cross-track attention을 제거해 track 간 의존성을 완전히 무시하게 한다. 공정성을 위해 attention layer 6개를 단순히 빼는 대신 time-attention layer 12개로 대체해 모델 크기를 비슷하게 유지한다. Dynamic Replica에서 joint tracking은 가려진 점의 정확도 $$\delta_{\text{avg}}^{\text{occ}}$$ 를 28.8에서 37.6으로(+30.6%), 보이는 점의 정확도 $$\delta_{\text{avg}}^{\text{vis}}$$ 를 62.4에서 68.9로(+10.4%) 올린다. 가려진 점의 개선 폭이 더 크다는 것은 joint tracking이 현재 보이지 않는 점을 포함한 장면 전체의 움직임을 이해하는 데 효과적임을 보여준다.

| Mode | DAVIS First AJ ↑ | $$\delta_{\text{avg}}$$ ↑ | OA ↑ | Dynamic Replica $$\delta_{\text{avg}}$$ ↑ | $$\delta_{\text{avg}}^{\text{vis}}$$ ↑ | $$\delta_{\text{avg}}^{\text{occ}}$$ ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| no joint | 55.6 | 70.1 | 83.0 | 54.4 | 62.4 | 28.8 |
| joint | **62.2** | **75.7** | **89.3** | **61.6** | **68.9** | **37.6** |

*[Table 4] joint tracking의 중요성(굵게 최고). DAVIS에서는 단일 target 점을 support point 유무로, Dynamic Replica에서는 모든 target 점을 개별·joint로 추적*

**Importance of unrolled training.** sliding window 방식으로 설계된 CoTracker에서 학습 중 window unrolling의 효과를 평가한다(Table 5). unrolled training을 끄면 성능이 AJ 기준 18점 떨어진다. unrolled training은 학습에 쓴 시퀀스보다 10배 이상 긴 기간의 tracking을 돕는다.

| Unrolled Training | AJ ↑ | $$\delta_{\text{avg}}$$ ↑ | OA ↑ |
| --- | --- | --- | --- |
| ✗ | 44.6 | 60.5 | 75.3 |
| ✓ | **62.2** | **75.7** | **89.3** |

*[Table 5] DAVIS First에서 unrolled training의 효과(굵게 최고)*

**Effect of proxy tokens on scalability.** inference에서 proxy token의 이득을 평가한다(Table 6). 고정 메모리 예산(80GB)에서 full self-attention 대신 proxy token을 쓰면 7.4배 많은 점을 추적할 수 있고, 실제로 $$263 \times 263$$ grid 전체(입력 비디오 해상도 기준 quasi-dense)를 추적할 수 있다. proxy token은 시간 복잡도도 줄여 최대 track 수 기준 inference가 7배 빨라진다. proxy token 수는 확장성에는 영향이 없지만 성능에는 영향을 주며 64개일 때 가장 좋다. 요컨대 proxy token은 naive self-attention보다 거의 한 자릿수 많은 track을 더 높은 정확도로 추적하게 해 준다.

| Num. proxy tokens | AJ ↑ | $$\delta_{\text{avg}}$$ ↑ | OA ↑ | Max. num. tracks | Time [s] |
| --- | --- | --- | --- | --- | --- |
| 0 | 61.6 | 75.6 | 88.3 | 9.4k | 207.3 |
| 32 | 60.2 | 74.4 | 88.5 | 69.2k | 26.8 |
| 64 | **62.2** | **75.7** | **89.3** | 69.2k | 27.9 |
| 128 | 60.9 | 74.8 | 88.4 | 69.2k | 30.1 |

*[Table 6] DAVIS First에서 proxy token 수에 따른 성능과 확장성(굵게 최고). 최대 track 수는 80GB GPU 기준*

**Optimal support point configurations.** Figure 5의 support point 구성들을 비교한다(Table 7). 벤치마크 점 하나씩만 target으로 삼고 support point를 더해 joint tracking을 수행한다. 이 protocol은 state-of-the-art와의 공정한 비교에도 중요하다. 벤치마크 점들은 물체 쪽으로 편향돼 있을 수 있어(TAP-Vid-DAVIS에서 특히 뚜렷하다), 그런 점 여러 개를 동시에 넣으면 물체의 윤곽이 tracker에 드러나 ground-truth 정보가 새어 들어갈 수 있기 때문이다. 벤치마크 점을 한 번에 하나만 고려하면 이런 누출이 없고, ground-truth 점을 한 번에 둘 이상 추적하는 것을 명시적으로 금지하는 TAP-Vid 벤치마크의 규칙도 지킨다. 어떤 구성이든 support point를 더하면 도움이 되는데, local 구성이 global보다 훨씬 크게 돕는다. 모델이 같은 물체 위의 다른 점들과의 의존성을 활용하기 때문으로 보인다. global과 local의 결합이 가장 좋은데, 이 경우 모델이 카메라 움직임과 물체 움직임을 모두 추적할 수 있기 때문으로 보인다.

| global | local | AJ ↑ | $$\delta_{\text{avg}}$$ ↑ | OA ↑ |
| --- | --- | --- | --- | --- |
| ✗ | ✗ | 55.6 | 70.1 | 83.0 |
| ✓ | ✗ | 56.8 | 71.2 | 85.8 |
| ✗ | ✓ | 60.4 | 75.4 | 87.3 |
| ✓ | ✓ | **62.2** | **75.7** | **89.3** |

*[Table 7] DAVIS First에서 support point 구성별 성능(굵게 최고)*

### 4.3 Limitations

높은 성능에도 불구하고 CoTracker는 사람이라면 하지 않을 tracking 실수를 가끔 한다. 순수 synthetic 데이터로만 학습되기 때문에 반사와 그림자가 있는 복잡한 시각 장면에 잘 일반화되지 않을 때가 있다. 예를 들어 그림자를 그것을 드리운 물체와 함께 추적하는 경향이 있다. 이는 응용에 따라 바람직한 성질일 수도(예: 비디오 편집), 아닐 수도(예: 움직임 분석) 있다.

## 5. Conclusions

track 사이의 의존성을 고려해 여러 점을 jointly 추적하는 transformer 기반 point tracker인 CoTracker를 제안했다. CoTracker는 표준 tracking 벤치마크에서 종종 큰 격차로 state-of-the-art이고, occlusion을 통과하거나 점이 시야를 벗어나도 수백 프레임까지 추적할 수 있으며, 매우 많은 수의 점을 동시에 추적할 수 있다. transformer architecture는 유연하고 메모리 효율적이어서, 앞으로 3D reconstruction 같은 기능의 통합도 가능하다.
