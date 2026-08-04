---
title: "[논문리뷰] FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (CVPR 2024)"
excerpt: "CAD model 유무와 무관하게 novel object의 6D pose estimation과 tracking을 하나의 framework로 통합한 FoundationPose 논문 리뷰"
categories:
  - 논문리뷰
  - CV
tags:
  - 6D pose estimation
  - object pose tracking
  - neural implicit representation
  - synthetic data generation
  - contrastive learning
toc: true
toc_sticky: true
---

> NVIDIA의 FoundationPose(CVPR 2024 Highlight) 논문을 리뷰한다. 학습 때 본 적 없는 novel object에 대해, CAD model이 있든(model-based) 참조 이미지 몇 장만 있든(model-free) 같은 모델로 6D pose estimation과 tracking을 모두 수행하는 통합 foundation model이다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://arxiv.org/abs/2312.08344) |
| 저자 | Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield |
| 소속 / 연도 | NVIDIA, 2024 (arXiv:2312.08344, CVPR 2024 Highlight) |
| 분야 | 6D Object Pose Estimation & Tracking |
| 코드 | [GitHub](https://github.com/NVlabs/FoundationPose) |

## 1. Introduction

물체에서 카메라로의 rigid 6D 변환을 계산하는 object pose estimation은 로봇 manipulation과 mixed reality 등에 필수적인 기술이다. 고전적인 방법은 instance-level로, 학습 때 정해진 특정 물체 instance에서만 동작한다. 이런 방법은 학습 데이터 생성을 위해 textured CAD model이 필요하고, 테스트 시점에 처음 보는 물체에는 적용할 수 없다. category-level 방법은 이 가정(instance별 학습, CAD model)을 제거하지만, 학습한 카테고리 안의 물체로 범위가 제한된다. 게다가 category-level 학습 데이터는 pose 정규화·검수 단계가 추가로 필요해 확보하기가 매우 어렵다.

이런 한계를 해결하려고 최근에는 임의의 novel object에 대한 즉시(pose 추정 전 fine-tuning 없는) pose estimation이 연구되고 있다. 테스트 시점에 주어지는 정보에 따라 두 가지 setup이 있다. **model-based**는 물체의 textured 3D CAD model이 주어지고, **model-free**는 물체를 찍은 참조 이미지 여러 장이 주어진다. 각 setup에서 개별적으로는 많은 진전이 있었지만, 실제 응용마다 주어지는 정보가 다르므로 두 setup을 통합해서 다루는 단일 방법이 필요하다.

single-frame pose estimation과 별개로, pose tracking 방법은 시간축 단서를 활용해 비디오에서 더 효율적이고 부드럽고 정확한 pose 추정을 한다. 이들 역시 물체에 대한 가정에 따라 pose estimation과 같은 문제를 공유한다.

![FoundationPose 통합 framework와 4개 태스크 성능](/assets/images/foundationpose-fig1.jpg)
*[Figure 1] model-based와 model-free를 모두 지원하는 통합 framework와 4개 태스크에서의 기존 방법 대비 성능*

이 논문은 RGBD 이미지를 사용해 model-based와 model-free 두 setup 모두에서 novel object의 pose estimation과 tracking을 모두 수행하는 통합 framework **FoundationPose**를 제안한다. Figure 1처럼 네 태스크 각각에 특화된 기존 state-of-the-art 방법을 모두 능가한다. 강한 일반화 능력은 LLM의 도움을 받은 대규모 합성 데이터 학습, 새로운 transformer 기반 구조, contrastive learning으로 얻는다. 적은 수($$\sim 16$$장)의 참조 이미지로 novel view synthesis가 가능한 neural implicit representation으로 두 setup 사이의 간극을 메우며, 기존 render-and-compare 방법보다 훨씬 빠른 렌더링 속도를 달성한다.

기여는 다음과 같이 요약된다.

- novel object의 pose estimation과 tracking을 모두 지원하는 통합 framework. object 중심 neural implicit representation이 model-based와 model-free setup의 간극을 메운다
- 다양한 texture 증강으로 3D 학습 asset의 다양성을 키우는 LLM 보조 합성 데이터 생성 pipeline
- transformer 기반 네트워크 구조와 contrastive learning 정식화의 새로운 설계로, 합성 데이터만으로 학습해도 강한 일반화 달성
- 여러 공개 데이터셋에서 각 태스크 특화 방법을 큰 차이로 능가. 가정을 줄였는데도 instance-level 방법에 필적하는 결과

## 2. Related Work

**CAD Model 기반 Object Pose Estimation.** instance-level 방법은 물체의 textured CAD model이 주어진다고 가정하고, 학습과 테스트가 완전히 같은 instance에서 이뤄진다. pose는 직접 regression하거나, 2D-3D 대응을 만들어 PnP로 풀거나, 3D-3D 대응을 만들어 least squares fitting으로 푼다. 물체 지식에 대한 가정을 완화한 category-level 방법은 같은 카테고리의 novel instance에 적용되지만, 카테고리 밖의 임의 물체로 일반화하지 못한다. 최근에는 테스트 시점에 CAD model만 주어지면 임의의 novel object를 즉시 다루는 연구가 진행되고 있다.

**Few-shot Model-free Object Pose Estimation.** model-free 방법은 명시적 textured model 대신 대상 물체를 찍은 참조 이미지 여러 장을 이용한다. RLLG와 NeRF-Pose는 CAD model 없이 instance별 학습을 제안했고, 특히 NeRF-Pose는 neural radiance field를 만들어 object coordinate map과 mask에 준지도를 준다. 이와 달리 이 논문은 SDF representation 위에 neural object field를 만들어 효율적인 RGB·depth 렌더링을 수행하고, 이것으로 model-based와 model-free 시나리오의 간극을 메운다. 또한 이 논문은 특정 instance가 아니라 일반화 가능한 novel object pose estimation을 다룬다. novel object를 다루기 위해 Gen6D는 detection, retrieval, refinement pipeline을 설계했지만 out-of-distribution 테스트를 피하려면 fine-tuning이 필요하다. OnePose와 OnePose++는 SfM으로 물체를 modeling하고 2D-3D matching 네트워크를 사전학습해 대응 관계로 pose를 푼다. FS6D도 비슷한 방식으로 RGBD modality에 집중한다. 그러나 대응 관계에 의존하는 방식은 textureless 물체나 심한 occlusion에서 취약해진다.

**Object Pose Tracking.** 6D object pose tracking은 시간축 단서를 활용해 비디오에서 효율적이고 부드럽고 정확한 pose 예측을 하는 것이 목표다. 이 논문의 방법은 neural 렌더링을 통해 높은 효율로 pose tracking 태스크로 자연스럽게 확장된다. 기존 tracking 방법도 물체 지식의 가정에 따라 instance-level, category-level, model-based novel object tracking, model-free novel object tracking으로 나뉜다. 이 논문은 model-based와 model-free 두 setup 모두에서 공개 데이터셋 기록을 새로 세우며, instance-level 학습이 필요한 state-of-the-art 방법까지 능가한다.

## 3. Approach

전체 시스템은 Figure 2와 같고, 각 구성 요소는 아래 소절에서 설명한다.

![FoundationPose 전체 framework 개요](/assets/images/foundationpose-fig2.png)
*[Figure 2] LLM 보조 합성 데이터 생성, neural object modeling, pose hypothesis generation, pose selection으로 구성된 전체 framework*

### 3.1 Language-aided Data Generation at Scale

강한 일반화를 위해서는 다양한 물체와 장면의 대규모 학습 데이터가 필요하다. 실제 세계에서 이런 데이터를 모으고 정확한 6D pose ground-truth를 다는 것은 시간·비용상 불가능에 가깝고, 기존 합성 데이터는 3D asset의 규모와 다양성이 부족하다. 이 논문은 대규모 3D model 데이터베이스, LLM, diffusion model이라는 최근 자원·기술을 활용한 새로운 합성 데이터 생성 pipeline을 개발해, 이전 연구 대비 데이터의 양과 다양성을 크게 확장했다.

**3D Assets.** 학습 asset은 Objaverse와 GSO 같은 최근 대규모 3D 데이터베이스에서 얻는다. Objaverse에서는 1156개 LVIS 카테고리에 속한 4만 개 이상의 물체로 이뤄진 Objaverse-LVIS subset을 골랐다. 일상 물체 위주로 품질이 좋고 모양·외형이 다양하며, 물체마다 카테고리를 설명하는 태그가 있어 다음 단계의 LLM 프롬프트 자동 생성에 유리하다.

![random texture blending과 LLM 보조 texture 증강 비교](/assets/images/foundationpose-fig3.png)
*[Figure 3] FS6D의 random texture blending(위)과 ChatGPT 프롬프트로 자동 생성한 LLM 보조 texture 증강(아래)의 비교*

**LLM-aided Texture Augmentation.** Objaverse 물체 대부분은 모양 품질은 좋지만 texture 품질 편차가 크다. FS6D는 ImageNet이나 MS-COCO 이미지를 무작위로 붙여 texture를 증강하는데, 무작위 UV mapping 때문에 이음새(seam) 같은 artifact가 생기고(Figure 3 위), 장면 전체 이미지를 물체에 입히면 비현실적인 결과가 나온다. 반면 이 논문은 텍스트 프롬프트, 물체 모양, 무작위로 초기화한 noisy texture를 TexFusion에 주어 증강된 textured model을 만든다. 프롬프트를 수작업으로 주는 것은 확장성이 없으므로 2단계 계층적 프롬프트 전략을 도입한다. 먼저 ChatGPT에게 물체의 가능한 외형을 설명해 달라고 프롬프트를 주는데, 이 프롬프트는 템플릿화되어 있어 Objaverse-LVIS가 제공하는 물체 태그만 바꿔 끼우면 된다. ChatGPT의 답변이 diffusion model의 texture 합성 프롬프트가 된다. 전 과정이 완전 자동이라 대규모의 다양한 데이터 생성이 가능하다.

**Data Generation.** 합성 데이터 생성은 NVIDIA Isaac Sim에서 path tracing 기반의 photo-realistic 렌더링으로 구현했다. 중력·물리 시뮬레이션으로 물리적으로 그럴듯한 장면을 만들고, 각 장면에서 원본과 texture 증강 버전을 포함한 물체를 무작위로 샘플링한다. 물체 크기, 재질, 카메라 pose, 조명도 무작위화한다.

### 3.2 Neural Object Modeling

CAD model이 없는 model-free setup의 핵심 과제는 downstream 모듈에 충분한 품질의 이미지를 효율적으로 렌더링할 수 있게 물체를 표현하는 것이다. neural implicit representation은 novel view synthesis에 효과적이면서 GPU에서 병렬화되므로, 여러 pose hypothesis를 렌더링할 때 계산 효율이 높다.

**Field Representation.** 물체를 두 함수로 표현한다. 기하 함수 $$\Omega: x \mapsto s$$ 는 3D 점 $$x \in \mathbb{R}^3$$ 을 입력받아 signed distance 값 $$s \in \mathbb{R}$$ 을 출력한다. 외형 함수 $$\Phi: (f_{\Omega(x)}, n, d) \mapsto c$$ 는 기하 네트워크의 중간 feature 벡터 $$f_{\Omega(x)}$$, 점의 normal $$n \in \mathbb{R}^3$$, 시선 방향 $$d \in \mathbb{R}^3$$ 을 입력받아 색 $$c \in \mathbb{R}^3_+$$ 을 출력한다. 실제로는 $$x$$ 에 multi-resolution hash encoding을 적용해 네트워크에 넣고, $$n$$ 과 $$d$$ 는 고정된 2차 spherical harmonics 계수로 embedding한다. 물체 표면은 signed distance field(SDF)의 zero level set으로 얻는다.

$$
S = \left\{ x \in \mathbb{R}^3 \mid \Omega(x) = 0 \right\}
$$

NeRF와 비교하면 SDF 표현은 depth 렌더링 품질이 더 높고, density threshold를 수동으로 고를 필요가 없다.

**Field Learning.** texture 학습은 표면 근처 절단 영역에 대한 volumetric 렌더링을 따른다.

$$
c(r) = \int_{z(r)-\lambda}^{z(r)+0.5\lambda} w(x_i)\, \Phi(f_{\Omega(x_i)}, n(x_i), d(x_i))\, dt
$$

$$
w(x_i) = \frac{1}{1+e^{-\alpha \Omega(x_i)}} \cdot \frac{1}{1+e^{\alpha \Omega(x_i)}}
$$

여기서 $$w(x_i)$$ 는 점 $$x_i$$ 에서 implicit 표면까지의 signed distance $$\Omega(x_i)$$ 에 의존하는 종 모양(bell-shaped) 확률밀도함수로, 표면 교차점에서 최대가 된다. $$\alpha$$ 는 이 분포의 퍼짐 정도를 조절한다. $$z(r)$$ 은 depth 이미지에서 얻은 ray $$r$$ 의 depth 값이고 $$\lambda$$ 는 절단 거리다. 표면에서 $$\lambda$$ 이상 떨어진 빈 공간의 기여는 무시해 학습 효율을 높이고, 자기 occlusion을 modeling하기 위해 침투 거리 $$0.5\lambda$$ 까지만 적분한다. 학습 중에는 이 값을 참조 RGB 이미지와 비교해 색 supervision을 준다.

$$
\mathcal{L}_c = \frac{1}{\lvert \mathcal{R} \rvert} \sum_{r \in \mathcal{R}} \left\| c(r) - \bar{c}(r) \right\|_2
$$

여기서 $$\mathcal{R}$$ 은 샘플링한 ray의 집합, $$\bar{c}(r)$$ 은 ray $$r$$ 이 지나는 픽셀의 ground-truth 색이다.

기하 학습은 공간을 두 영역으로 나눠 SDF를 학습하는 hybrid SDF model을 채택해, 빈 공간 loss와 표면 근처 loss를 쓴다. 표면 근처 SDF에는 eikonal 정규화도 적용한다.

$$
\mathcal{L}_e = \frac{1}{\lvert \mathcal{X}_e \rvert} \sum_{x \in \mathcal{X}_e} \lvert \Omega(x) - \lambda \rvert
$$

$$
\mathcal{L}_s = \frac{1}{\lvert \mathcal{X}_s \rvert} \sum_{x \in \mathcal{X}_s} \left( \Omega(x) + d_x - d_D \right)^2
$$

$$
\mathcal{L}_{eik} = \frac{1}{\lvert \mathcal{X}_s \rvert} \sum_{x \in \mathcal{X}_s} \left( \left\| \nabla \Omega(x) \right\|_2 - 1 \right)^2
$$

여기서 $$x$$ 는 나뉜 공간에서 ray를 따라 샘플링한 3D 점이고, $$\mathcal{X}_e$$ 는 빈 공간, $$\mathcal{X}_s$$ 는 표면 근처 영역의 샘플 집합이다. $$d_x$$ 와 $$d_D$$ 는 각각 ray 원점에서 샘플 점까지, 관측된 depth 점까지의 거리다. $$\mathcal{L}_e$$ 는 빈 공간의 SDF 값을 절단 거리 $$\lambda$$ 로 밀고, $$\mathcal{L}_s$$ 는 표면 근처에서 SDF가 관측 depth와 일치하게 하며, $$\mathcal{L}_{eik}$$ 는 SDF gradient의 크기가 1이 되도록 정규화한다. 참조 이미지가 미리 촬영되는 model-free setup이므로 uncertain free-space loss는 쓰지 않는다. 전체 학습 loss는 다음과 같다.

$$
\mathcal{L} = w_c \mathcal{L}_c + w_e \mathcal{L}_e + w_s \mathcal{L}_s + w_{eik} \mathcal{L}_{eik}
$$

여기서 $$w_c, w_e, w_s, w_{eik}$$ 는 각 항의 가중치다. 학습은 사전 지식 없이 물체별로 최적화되며 수 초 안에 끝난다. novel object마다 neural field를 한 번만 학습하면 된다.

**Rendering.** 학습된 neural field는 기존 그래픽스 pipeline의 드롭인 대체재로서, 이후의 render-and-compare 반복을 위해 물체를 효율적으로 렌더링한다. RGBD 기반 pose estimation과 tracking을 위해 색뿐 아니라 depth 렌더링도 필요하다. 이를 위해 SDF의 zero level set에서 Marching Cubes로 textured mesh를 추출하고 색을 투영하는데, 물체마다 한 번만 수행하면 된다. 추론 시에는 물체 pose가 주어지면 rasterization 과정을 따라 RGBD 이미지를 렌더링한다. sphere tracing으로 $$\Omega$$ 에서 직접 depth를 렌더링할 수도 있지만, 병렬로 렌더링할 pose hypothesis가 많을 때 효율이 떨어지는 것을 확인했다.

### 3.3 Pose Hypothesis Generation

**Pose Initialization.** RGBD 이미지가 주어지면 Mask R-CNN이나 CNOS 같은 기성 방법으로 물체를 검출한다. translation은 검출된 2D bounding box 안 median depth에 위치한 3D 점으로 초기화한다. rotation은 물체를 중심으로 한 icosphere에서 카메라가 중심을 바라보는 $$N_s$$ 개 시점을 균일하게 샘플링하고, 여기에 $$N_i$$ 개의 이산화된 in-plane rotation을 더해 $$N_s \cdot N_i$$ 개의 전역 pose 초기값을 만들어 pose refiner에 입력한다.

**Pose Refinement.** 앞 단계의 coarse pose 초기값은 잡음이 많으므로 품질을 높이는 refinement 모듈이 필요하다. pose refinement 네트워크는 coarse pose로 렌더링한 물체 이미지와 카메라 관측의 crop을 입력받아 pose 업데이트를 출력한다. coarse pose 주변 여러 view를 렌더링해 anchor 점을 찾는 MegaPose와 달리, coarse pose에 해당하는 단일 view 렌더링으로 충분함을 확인했다. 관측 입력은 고정된 2D detection 기준 crop 대신 **pose-conditioned cropping** 전략을 쓴다. 물체 원점을 이미지 공간에 투영해 crop 중심을 정하고, 약간 키운 물체 지름(물체 표면 위 두 점 사이의 최대 거리)을 투영해 crop 크기를 정한다. crop이 coarse pose에 조건화되므로 네트워크가 crop과 관측이 더 잘 정렬되게 translation을 업데이트하도록 유도한다. refinement는 최신 pose를 다시 입력해 여러 번 반복할 수 있다.

네트워크 구조는 Figure 2와 같다. 두 RGBD 입력 branch에서 공유된 단일 CNN encoder로 feature map을 추출하고, 이어붙인 뒤 residual 연결이 있는 CNN 블록에 넣고, patch로 나눠 position embedding과 함께 token화한다. 마지막으로 translation 업데이트 $$\Delta \boldsymbol{t} \in \mathbb{R}^3$$ 와 rotation 업데이트 $$\Delta \boldsymbol{R} \in \mathbb{SO}(3)$$ 를 각각 별도의 transformer encoder로 처리하고 출력 차원으로 선형 투영해 예측한다. $$\Delta \boldsymbol{t}$$ 는 카메라 좌표계에서 물체의 이동량, $$\Delta \boldsymbol{R}$$ 은 카메라 좌표계에서 표현한 물체의 방향 업데이트다. rotation은 axis-angle 표현을 쓴다(6D 표현도 실험했으나 결과가 비슷했다). 입력 coarse pose $$[\boldsymbol{R} \mid \boldsymbol{t}] \in \mathbb{SE}(3)$$ 는 다음과 같이 업데이트된다.

$$
\boldsymbol{t}^{+} = \boldsymbol{t} + \Delta \boldsymbol{t}
$$

$$
\boldsymbol{R}^{+} = \Delta \boldsymbol{R} \otimes \boldsymbol{R}
$$

여기서 $$\otimes$$ 는 $$\mathbb{SO}(3)$$ 위의 업데이트다. 단일 homogeneous pose 업데이트 대신 이렇게 분리된(disentangled) 표현을 쓰면 translation 업데이트가 갱신된 orientation에 의존하지 않게 된다. 업데이트와 입력 관측이 모두 카메라 좌표계로 통일되어 학습이 단순해진다. 네트워크 학습은 $$L_2$$ loss로 지도한다.

$$
\mathcal{L}_{\text{refine}} = w_1 \left\| \Delta \boldsymbol{t} - \Delta \bar{\boldsymbol{t}} \right\|_2 + w_2 \left\| \Delta \boldsymbol{R} - \Delta \bar{\boldsymbol{R}} \right\|_2
$$

여기서 $$\Delta \bar{\boldsymbol{t}}$$ 와 $$\Delta \bar{\boldsymbol{R}}$$ 은 ground-truth 업데이트이고, $$w_1, w_2$$ 는 두 loss의 균형을 맞추는 가중치로 경험적으로 1로 둔다.

### 3.4 Pose Selection

refinement를 거친 pose hypothesis 목록이 주어지면, 계층적 pose ranking 네트워크로 각 hypothesis의 점수를 계산하고 최고 점수의 pose를 최종 추정으로 선택한다.

![계층적 비교 유무에 따른 pose ranking 결과](/assets/images/foundationpose-fig4.jpg)
*[Figure 4] 모든 pose hypothesis의 전역 문맥을 활용하는 계층적 비교의 pose ranking 시각화(빨간 원이 true best pose)*

**Hierarchical Comparison.** 네트워크는 2단계 비교 전략을 쓴다. 1단계에서는 각 pose hypothesis에 대해 렌더링 이미지를 pose-conditioned cropping으로 자른 입력 관측과 비교한다. 이 비교는 refinement 네트워크와 같은 backbone을 쓰는 pose ranking encoder로 수행하며, 추출된 feature를 이어붙이고 token화해 multi-head self-attention 모듈에 넣어 이미지 전역 문맥을 활용한다. pose ranking encoder는 average pooling으로 렌더링과 관측 사이의 정렬 품질을 나타내는 feature embedding $$\mathcal{F} \in \mathbb{R}^{512}$$ 를 출력한다. 여기서 흔히 하듯 $$\mathcal{F}$$ 를 바로 유사도 scalar로 투영할 수도 있지만, 그러면 다른 pose hypothesis들을 무시한 채 절대 점수를 매겨야 해서 학습이 어렵다.

더 나은 결정을 위해 모든 pose hypothesis의 전역 문맥을 활용하는 2단계 비교를 도입한다. $$K$$ 개 hypothesis 전체의 embedding을 이어붙인 $$\mathbf{F} = [\mathcal{F}_0, \ldots, \mathcal{F}_{K-1}]^\top \in \mathbb{R}^{K \times 512}$$ 에 multi-head self-attention을 수행한다. $$\mathbf{F}$$ 를 sequence로 다루므로 $$K$$ 의 길이가 달라져도 자연스럽게 일반화되고, permutation에 무관하도록 position encoding은 적용하지 않는다. attention을 거친 feature는 선형 투영되어 각 hypothesis에 배정될 점수 $$\mathbf{S} \in \mathbb{R}^K$$ 가 된다. 이 계층적 비교 전략의 효과는 Figure 4의 예시에서 볼 수 있다.

**Contrast Validation.** pose ranking 네트워크 학습을 위해 pose-conditioned triplet loss를 제안한다.

$$
\mathcal{L}(i^{+}, i^{-}) = \max(\mathbf{S}(i^{-}) - \mathbf{S}(i^{+}) + \alpha,\, 0)
$$

여기서 $$\alpha$$ 는 contrastive margin이고, $$i^{-}$$ 와 $$i^{+}$$ 는 각각 negative·positive pose 샘플로, ground truth와의 ADD metric을 계산해 정한다. 표준 triplet loss와 달리 anchor 샘플을 positive와 negative가 공유하지 않는데, 입력 crop이 각 pose hypothesis에 조건화되어 있기 때문이다. 이 loss를 목록의 모든 쌍에 대해 계산할 수도 있지만, 두 pose가 모두 ground truth에서 멀면 비교가 모호해진다. 따라서 positive 샘플의 시점이 ground truth에 충분히 가까운 쌍만 남긴다.

$$
\mathbb{V}^{+} = \{ i : D(\boldsymbol{R}_i, \bar{\boldsymbol{R}}) < d \}
$$

$$
\mathbb{V}^{-} = \{ 0, 1, 2, \ldots, K-1 \}
$$

$$
\mathcal{L}_{\text{rank}} = \sum_{i^{+}, i^{-}} \mathcal{L}(i^{+}, i^{-})
$$

여기서 합은 $$i^{+} \in \mathbb{V}^{+}$$, $$i^{-} \in \mathbb{V}^{-}$$, $$i^{+} \neq i^{-}$$ 에 대해 계산한다. $$\boldsymbol{R}_i$$ 와 $$\bar{\boldsymbol{R}}$$ 은 각각 hypothesis와 ground truth의 rotation이고, $$D(\cdot)$$ 는 rotation 사이의 geodesic 거리, $$d$$ 는 미리 정한 threshold다. InfoNCE loss도 실험했지만 더 나쁜 성능을 보였는데(4.5절), InfoNCE가 가정하는 완벽한 translation이 이 setup에서는 성립하지 않기 때문으로 본다.

## 4. Experiments

### 4.1 Dataset and Setup

LINEMOD, Occluded-LINEMOD, YCB-Video, T-LESS, YCBInEOAT의 5개 데이터셋을 사용한다. 밀집한 clutter, 다중 instance, 정적·동적 장면, 탁자 위·로봇 manipulation 등 다양한 시나리오와, textureless·반짝임·대칭·다양한 크기 등 다양한 성질의 물체를 포함한다.

framework가 통합되어 있으므로 두 setup(model-free, model-based)과 두 태스크(6D pose estimation, tracking)의 조합인 총 4개 태스크를 평가한다. model-free setup은 데이터셋 학습 split에서 pose ground-truth가 달린 참조 이미지 몇 장을 고르고, model-based setup은 novel object의 CAD model이 주어진다. ablation을 제외한 모든 평가에서 fine-tuning 없이 같은 학습된 모델과 설정을 사용한다.

### 4.2 Metric

각 setup의 baseline protocol을 그대로 따르기 위해 다음 metric을 사용한다.

- ADD와 ADD-S의 area under the curve(AUC)
- 물체 지름의 0.1 미만인 ADD의 recall(ADD-0.1d)
- BOP challenge에서 도입된 VSD, MSSD, MSPD metric의 average recall(AR)

### 4.3 Pose Estimation Comparison

**Model-free.** Table 1은 YCB-Video 데이터셋에서 state-of-the-art RGBD 방법들과의 비교다. 모든 방법은 공정한 비교를 위해 perturb된 ground-truth bounding box를 2D detection으로 받는다. Table 2는 LINEMOD 데이터셋 비교다. RGB 기반 방법들은 depth가 없는 것을 보상하기 위해 훨씬 많은 참조 이미지를 쓰는 특혜를 받고, RGBD 방법 중 FS6D는 대상 데이터셋에서 fine-tuning이 필요하다. 이 논문의 방법은 대상 데이터셋 fine-tuning이나 ICP refinement 없이 두 데이터셋 모두에서 기존 방법을 크게 능가한다.

| 객체 | PREDATOR | LoFTR | FS6D-DPM | Ours |
| --- | --- | --- | --- | --- |
| Ref. images | 16 | 16 | 16 | 16 |
| Finetune-free | ✓ | ✓ | ✗ | ✓ |
| 002_master_chef_can | 73.0 / 17.4 | 87.2 / 50.6 | 92.6 / 36.8 | 96.9 / 91.3 |
| 003_cracker_box | 41.7 / 8.3 | 71.8 / 25.5 | 83.9 / 24.5 | 97.5 / 96.2 |
| 004_sugar_box | 53.7 / 15.3 | 63.9 / 13.4 | 95.1 / 43.9 | 97.5 / 87.2 |
| 005_tomato_soup_can | 81.2 / 44.4 | 77.1 / 52.9 | 93.0 / 54.2 | 97.6 / 93.3 |
| 006_mustard_bottle | 35.5 / 5.0 | 84.5 / 59.0 | 97.0 / 71.1 | 98.4 / 97.3 |
| 007_tuna_fish_can | 78.2 / 34.2 | 72.6 / 55.7 | 94.5 / 53.9 | 97.7 / 73.7 |
| 008_pudding_box | 73.5 / 24.2 | 86.5 / 68.1 | 94.9 / 79.6 | 98.5 / 97.0 |
| 009_gelatin_box | 81.4 / 37.5 | 71.6 / 45.2 | 98.3 / 32.1 | 98.5 / 97.3 |
| 010_potted_meat_can | 62.0 / 20.9 | 67.4 / 45.1 | 87.6 / 54.9 | 96.6 / 82.3 |
| 011_banana | 57.7 / 9.9 | 24.2 / 1.6 | 94.0 / 69.1 | 98.1 / 95.4 |
| 019_pitcher_base | 83.7 / 18.1 | 58.7 / 22.3 | 91.1 / 40.4 | 97.9 / 96.6 |
| 021_bleach_cleanser | 88.3 / 48.1 | 36.9 / 16.7 | 89.4 / 44.1 | 97.4 / 93.3 |
| 024_bowl | 73.2 / 17.4 | 32.7 / 1.4 | 74.7 / 0.9 | 94.9 / 89.7 |
| 025_mug | 84.8 / 29.5 | 47.3 / 23.6 | 86.5 / 39.2 | 96.2 / 75.8 |
| 035_power_drill | 60.6 / 12.3 | 18.8 / 1.3 | 73.0 / 19.8 | 98.0 / 96.3 |
| 036_wood_block | 70.5 / 10.0 | 49.9 / 1.4 | 94.7 / 27.9 | 97.4 / 94.7 |
| 037_scissors | 75.5 / 25.0 | 32.3 / 14.6 | 74.2 / 27.7 | 97.8 / 95.5 |
| 040_large_marker | 81.8 / 38.9 | 20.7 / 8.4 | 97.4 / 74.2 | 98.6 / 96.5 |
| 051_large_clamp | 83.0 / 34.4 | 24.1 / 11.2 | 82.7 / 34.7 | 96.9 / 92.7 |
| 052_extra_large_clamp | 72.9 / 24.1 | 15.0 / 1.8 | 65.7 / 10.1 | 97.6 / 94.1 |
| 061_foam_brick | 79.2 / 35.5 | 59.4 / 31.4 | 95.7 / 45.8 | 98.1 / 93.4 |
| MEAN | 71.0 / 24.3 | 52.5 / 26.2 | 88.4 / 42.1 | **97.4** / **91.5** |

*[Table 1] YCB-Video에서 AUC of ADD-S / ADD로 측정한 model-free pose estimation 결과(각 셀은 ADD-S / ADD, 굵게 최고). "Finetune-free ✗"는 테스트 데이터셋의 물체 instance group split으로 fine-tuning했음을 의미*

| Method | Modality | Finetune-free | Ref. images | ape | benchwise | cam | can | cat | driller | duck | eggbox | glue | holepuncher | iron | lamp | phone | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gen6D | RGB | ✗ | 200 | - | 77 | 66.1 | - | 60.7 | 67.4 | 40.5 | 95.7 | 87.2 | - | - | - | - | - |
| Gen6D* | RGB | ✓ | 200 | - | 62.1 | 45.6 | - | 40.9 | 48.8 | 16.2 | - | - | - | - | - | - | - |
| OnePose | RGB | ✓ | 200 | 11.8 | 92.6 | 88.1 | 77.2 | 47.9 | 74.5 | 34.2 | 71.3 | 37.5 | 54.9 | 89.2 | 87.6 | 60.6 | 63.6 |
| OnePose++ | RGB | ✓ | 200 | 31.2 | 97.3 | 88.0 | 89.8 | 70.4 | 92.5 | 42.3 | 99.7 | 48.0 | 69.7 | 97.4 | 97.8 | 76.0 | 76.9 |
| LatentFusion | RGBD | ✓ | 16 | 88.0 | 92.4 | 74.4 | 88.8 | 94.5 | 91.7 | 68.1 | 96.3 | 94.9 | 82.1 | 74.6 | 94.7 | 91.5 | 87.1 |
| FS6D | RGBD | ✗ | 16 | 74.0 | 86.0 | 88.5 | 86.0 | 98.5 | 81.0 | 68.5 | **100.0** | 99.5 | 97.0 | 92.5 | 85.0 | 99.0 | 88.9 |
| FS6D + ICP | RGBD | ✗ | 16 | 78.0 | 88.5 | 91.0 | 89.5 | 97.5 | 92.0 | 75.5 | 99.5 | 99.5 | 96.0 | 87.5 | 97.0 | 97.5 | 91.5 |
| Ours | RGBD | ✓ | 16 | **99.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **99.4** | **100.0** | **100.0** | **99.9** | **100.0** | **100.0** | **100.0** | **99.9** |

*[Table 2] LINEMOD에서 ADD-0.1d로 측정한 model-free pose estimation 결과(굵게 최고). Gen6D\* 는 fine-tuning 없는 변형*

Figure 5는 정성적 비교다. FS6D는 코드가 공개되지 않아 pose 예측을 얻을 수 없었다. glue의 심한 자기 occlusion과 texture 부족은 OnePose++와 LatentFusion에게 큰 어려움이 되지만, 이 논문의 방법은 pose를 성공적으로 추정한다.

![LINEMOD model-free 정성 비교](/assets/images/foundationpose-fig5.jpg)
*[Figure 5] model-free setup에서 LINEMOD의 pose estimation 정성 비교*

**Model-based.** Table 3은 BOP의 3개 core 데이터셋(Occluded-LINEMOD, YCB-Video, T-LESS)에서 RGBD 방법들과의 비교다. 모든 방법은 Mask R-CNN으로 2D detection을 한다. novel object를 다루는 기존 model-based 방법들과 instance-level 방법인 SurfEmb까지 큰 차이로 능가한다.

| Method | Unseen objects | LM-O | T-LESS | YCB-V | Mean |
| --- | --- | --- | --- | --- | --- |
| SurfEmb + ICP | ✗ | 75.8 | 82.8 | 80.6 | 79.7 |
| OSOP + ICP | ✓ | 48.2 | - | 57.2 | - |
| (PPF, Sift) + Zephyr | ✓ | 59.8 | - | 51.6 | - |
| MegaPose-RGBD | ✓ | 58.3 | 54.3 | 63.3 | 58.6 |
| OVE6D | ✓ | 49.6 | 52.3 | - | - |
| GCPose | ✓ | 65.2 | 67.9 | - | - |
| Ours | ✓ | **78.8** | **83.0** | **88.0** | **83.3** |

*[Table 3] 대표 BOP 데이터셋에서 AR score로 측정한 model-based pose estimation 결과(굵게 최고, 모두 RGBD modality)*

### 4.4 Pose Tracking Comparison

별도 명시가 없으면 장기 tracking 강건성을 평가하기 위해 tracking을 놓쳐도 re-initialization을 적용하지 않는다.

급격한 out-of-plane rotation, 동적 외부 occlusion, 분리된 카메라 움직임 같은 어려움을 종합적으로 비교하기 위해, 동적 로봇 manipulation 비디오를 포함한 YCBInEOAT 데이터셋에서 평가한다. model-based setup의 결과가 Table 4다. 이 논문의 방법이 최고 성능을 달성하며, ground-truth pose로 초기화한 instance별 학습 방법(se(3)-TrackNet)까지 능가한다. 나아가 통합 framework 덕분에 외부 pose 초기화 없이 end-to-end로 pose estimation과 tracking을 수행할 수 있는 유일한 방법이다(표의 Ours†).

| 객체 | Metric | se(3)-TrackNet | RGF | BundleTrack | BundleSDF | Wüthrich | Ours | Ours† |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novel object | | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Initial pose | | GT | GT | GT | GT | GT | GT | Est. |
| cracker_box | ADD-S | 94.06 | 55.44 | 89.41 | 90.63 | 88.13 | **95.10** | **94.92** |
| | ADD | 90.76 | 34.78 | 85.07 | 85.37 | 79.00 | **91.32** | **91.54** |
| bleach_cleanser | ADD-S | 94.44 | 45.03 | 94.72 | 94.28 | 68.96 | **95.96** | **96.36** |
| | ADD | 89.58 | 29.40 | 89.34 | 87.46 | 61.47 | **91.45** | **92.63** |
| sugar_box | ADD-S | 94.80 | 16.87 | 90.22 | 93.81 | 92.75 | **96.67** | **96.61** |
| | ADD | 92.43 | 15.82 | 85.56 | 88.62 | 86.78 | **94.14** | **93.96** |
| tomato_soup_can | ADD-S | 96.95 | 26.44 | 95.13 | 95.24 | 93.17 | **96.58** | **96.54** |
| | ADD | 93.40 | 15.13 | 86.00 | 83.10 | 63.71 | **91.71** | **91.85** |
| mustard_bottle | ADD-S | 97.92 | 60.17 | 95.35 | 95.75 | 95.31 | **97.89** | **97.77** |
| | ADD | 97.00 | 56.49 | 92.26 | 89.87 | 91.31 | **96.34** | **95.95** |
| All | ADD-S | 95.53 | 39.90 | 92.53 | 93.77 | 89.18 | **96.42** | **96.40** |
| | ADD | 92.66 | 29.98 | 87.34 | 86.95 | 78.28 | **93.09** | **93.22** |

*[Table 4] YCBInEOAT에서 AUC of ADD와 ADD-S로 측정한 RGBD 방법의 pose tracking 결과(굵게 최고). Ours† 는 pose estimation 모듈로 pose를 초기화하는 통합 pipeline*

Table 5는 YCB-Video에서의 pose tracking 비교다. baseline 중 DeepIM, se(3)-TrackNet, PoseRBPF는 같은 물체 instance에 대한 학습이 필요한 반면, Wüthrich et al., RGF, ICG와 이 논문의 방법은 CAD model만 주어지면 novel object에 즉시 적용할 수 있다.

| 객체 | DeepIM | se(3)-TrackNet | PoseRBPF + SDF | Wüthrich | RGF | ICG | Ours | Ours† |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial pose | GT | GT | PoseCNN | GT | GT | GT | GT | GT |
| Re-initialization | Yes (290) | No | Yes (2) | No | No | No | No | No |
| Novel object | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Object setup | Model-based | Model-based | Model-based | Model-based | Model-based | Model-based | Model-based | Model-free |
| 002_master_chef_can | 89.0 / 93.8 | 93.9 / 96.3 | 89.3 / 96.7 | 55.6 / 90.7 | 46.2 / 90.2 | 66.4 / 89.7 | 93.6 / 97.0 | 91.2 / 96.9 |
| 003_cracker_box | 88.5 / 93.0 | 96.5 / 97.2 | 96.0 / 97.1 | 96.4 / 97.2 | 57.0 / 72.3 | 82.4 / 92.1 | 96.9 / 97.8 | 96.2 / 97.5 |
| 004_sugar_box | 94.3 / 96.3 | 97.6 / 98.1 | 94.0 / 96.4 | 97.1 / 97.9 | 50.4 / 72.7 | 96.1 / 98.4 | 96.9 / 98.2 | 94.5 / 97.4 |
| 005_tomato_soup_can | 89.1 / 93.2 | 95.0 / 97.2 | 87.2 / 95.2 | 64.7 / 89.5 | 72.4 / 91.6 | 73.2 / 97.3 | 96.3 / 98.1 | 94.3 / 97.9 |
| 006_mustard_bottle | 92.0 / 95.1 | 95.8 / 97.4 | 98.3 / 98.5 | 97.1 / 98.0 | 87.7 / 98.2 | 96.2 / 98.4 | 97.3 / 98.4 | 97.3 / 98.5 |
| 007_tuna_fish_can | 92.0 / 96.4 | 86.5 / 91.1 | 86.8 / 93.6 | 69.1 / 93.3 | 28.7 / 52.9 | 73.2 / 95.8 | 96.9 / 98.5 | 84.0 / 97.8 |
| 008_pudding_box | 80.1 / 88.3 | 97.9 / 98.4 | 60.9 / 87.1 | 96.8 / 97.9 | 12.7 / 18.0 | 73.8 / 88.9 | 97.8 / 98.5 | 96.9 / 98.5 |
| 009_gelatin_box | 92.0 / 94.4 | 97.8 / 98.4 | 98.2 / 98.6 | 97.5 / 98.4 | 49.1 / 70.7 | 97.2 / 98.8 | 97.7 / 98.5 | 97.6 / 98.5 |
| 010_potted_meat_can | 78.0 / 88.9 | 77.8 / 84.2 | 76.4 / 83.5 | 83.7 / 86.7 | 44.1 / 45.6 | 93.3 / 97.3 | 95.1 / 97.7 | 94.8 / 97.5 |
| 011_banana | 81.0 / 90.5 | 94.9 / 97.2 | 92.8 / 97.7 | 86.3 / 96.1 | 93.3 / 97.7 | 95.6 / 98.4 | 96.4 / 98.4 | 95.6 / 98.1 |
| 019_pitcher_base | 90.4 / 94.7 | 96.8 / 97.5 | 97.7 / 98.1 | 97.3 / 97.7 | 97.9 / 98.2 | 97.0 / 98.8 | 96.7 / 98.0 | 96.8 / 98.0 |
| 021_bleach_cleanser | 81.7 / 90.5 | 95.9 / 97.2 | 95.9 / 97.0 | 95.2 / 97.2 | 95.9 / 97.3 | 92.6 / 97.5 | 95.5 / 97.8 | 94.7 / 97.5 |
| 024_bowl | 38.8 / 90.6 | 80.9 / 94.5 | 34.0 / 93.0 | 30.4 / 97.2 | 24.2 / 82.4 | 74.4 / 98.4 | 95.2 / 97.6 | 90.5 / 95.3 |
| 025_mug | 83.2 / 92.0 | 91.5 / 96.9 | 86.9 / 96.7 | 83.2 / 93.3 | 60.0 / 71.2 | 95.6 / 98.5 | 95.6 / 97.9 | 91.5 / 96.1 |
| 035_power_drill | 85.4 / 92.3 | 96.4 / 97.4 | 97.8 / 98.2 | 97.1 / 97.8 | 97.9 / 98.3 | 96.7 / 98.5 | 96.9 / 98.2 | 96.3 / 97.9 |
| 036_wood_block | 44.3 / 75.4 | 95.2 / 96.7 | 37.8 / 93.6 | 95.5 / 96.9 | 45.7 / 62.5 | 93.5 / 97.2 | 93.2 / 97.0 | 92.9 / 97.0 |
| 037_scissors | 70.3 / 84.5 | 95.7 / 97s | 72.7 / 85.5 | 4.2 / 16.2 | 20.9 / 38.6 | 93.5 / 97.3 | 94.8 / 97.5 | 95.5 / 97.8 |
| 040_large_marker | 80.4 / 91.2 | 92.2 / 96.0 | 89.2 / 97.3 | 35.6 / 53.0 | 12.2 / 18.9 | 88.5 / 97.8 | 96.9 / 98.6 | 96.6 / 98.6 |
| 051_large_clamp | 73.9 / 84.1 | 94.7 / 96.9 | 90.1 / 95.5 | 61.2 / 72.3 | 62.8 / 80.1 | 91.8 / 96.9 | 93.6 / 97.3 | 92.5 / 96.7 |
| 052_extra_large_clamp | 49.3 / 90.3 | 91.7 / 95.8 | 84.4 / 94.1 | 93.7 / 96.6 | 67.5 / 69.7 | 85.9 / 94.3 | 94.4 / 97.5 | 93.4 / 97.3 |
| 061_foam_brick | 91.6 / 95.5 | 93.7 / 96.7 | 96.1 / 98.3 | 96.8 / 98.1 | 70.0 / 86.5 | 96.2 / 98.5 | 97.9 / 98.6 | 96.8 / 98.3 |
| All Frames | 82.3 / 91.9 | 93.0 / 95.7 | 87.5 / 95.2 | 78.0 / 90.2 | 59.2 / 74.3 | 86.4 / 96.5 | **96.0** / **97.9** | **93.7** / **97.5** |

*[Table 5] YCB-Video에서 AUC of ADD와 ADD-S로 측정한 RGBD 방법의 pose tracking 결과(각 셀은 ADD / ADD-S, 굵게 최고). Ours† 는 참조 이미지를 쓰는 model-free setup*

### 4.5 Analysis

**Ablation Study.** Table 6은 핵심 설계 선택의 ablation으로, model-free(참조 이미지 16장) setup에서 YCB-Video의 AUC of ADD·ADD-S로 평가했다. *W/o LLM texture augmentation* 은 합성 학습에서 LLM 보조 texture 증강을 제거한 것이고, *W/o transformer* 는 transformer 기반 구조를 비슷한 파라미터 수의 convolution·linear layer로 바꾼 것이다. *W/o hierarchical comparison* 은 2단계 계층 비교 없이 렌더링과 crop 입력만 비교하도록 pose-conditioned triplet loss로 학습해, 테스트 때 각 hypothesis를 독립적으로 관측과 비교한다. *Ours-InfoNCE* 는 contrast validation 쌍별 loss를 InfoNCE loss로 바꾼 것이다.

| | ADD | ADD-S |
| --- | --- | --- |
| Ours (proposed) | 91.52 | 97.40 |
| W/o LLM texture augmentation | 90.83 | 97.38 |
| W/o transformer | 90.77 | 97.33 |
| W/o hierarchical comparison | 89.05 | 96.67 |
| Ours-InfoNCE | 89.39 | 97.29 |

*[Table 6] 핵심 설계 선택의 ablation 결과*

**참조 이미지 수의 효과.** YCB-Video에서 참조 이미지 수에 따른 AUC of ADD·ADD-S 변화를 보면, 특히 ADD-S에서 참조 이미지 수에 강건하고 두 metric 모두 12장에서 포화된다. 참조 이미지가 4장뿐이어도 16장을 쓴 FS6D보다 강한 성능을 낸다.

**학습 데이터 scaling law.** 이론적으로 합성 데이터는 무한정 생산할 수 있다. 학습 데이터 양에 따른 성능은 약 100만 개 근처에서 이득이 포화된다.

**Running time.** Intel i9-10980XE CPU와 NVIDIA RTX 3090 GPU에서 측정했다. pose estimation은 물체 하나에 약 1.3초가 걸리며, pose initialization 4ms, refinement 0.88초, pose selection 0.42초다. tracking은 pose refinement만 필요하고 여러 hypothesis가 없으므로 약 32Hz로 훨씬 빠르다. 실제로는 초기화로 pose estimation을 한 번 수행한 뒤 tracking 모드로 전환해 실시간 성능을 낼 수 있다.

## 5. Conclusion

novel object의 6D pose estimation과 tracking을 위한 통합 foundation model을 제시했으며, model-based와 model-free setup을 모두 지원한다. 4개 태스크 조합에 대한 광범위한 실험은 이 방법이 다재다능할 뿐 아니라 각 태스크에 특화 설계된 기존 state-of-the-art 방법을 상당한 차이로 능가함을 보여준다. instance-level 학습이 필요한 방법에 필적하는 결과까지 달성한다. 향후 연구로는 단일 rigid object를 넘어선 상태 추정 탐구가 관심사다.
