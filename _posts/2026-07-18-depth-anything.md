---
title: "[논문리뷰] Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data (CVPR 2024)"
excerpt: "62M장의 unlabeled 이미지 self-training과 DINOv2 feature alignment로 만든 zero-shot Monocular Depth Estimation foundation model, Depth Anything 논문 리뷰"
date: 2026-07-18
categories:
  - 논문리뷰
  - CV
tags:
  - Depth Anything
  - monocular depth estimation
  - self-training
  - semi-supervised learning
  - DINOv2
  - zero-shot
toc: true
toc_sticky: true
---

> **depth label이 있는 이미지는 백만 장 수준에서 더 늘리기 어렵다. 그럼 label 없는 이미지 6200만 장은 어떻게 써먹을 수 있을까?**
> Depth Anything은 새 모듈 없이 — pseudo label + 강한 perturbation + DINOv2 feature alignment 만으로 — 어떤 이미지에서든 동작하는 Monocular Depth Estimation foundation model을 만든다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891) |
| 저자 | Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao |
| 소속 / 연도 | HKU · TikTok · CUHK · ZJU, CVPR 2024 (arXiv:2401.10891) |
| 분야 | CV / monocular depth estimation |
| 코드 | [GitHub](https://github.com/LiheYoung/Depth-Anything) |

## 1. Introduction

컴퓨터 비전과 자연어 처리는 "foundation model"의 등장으로 다양한 downstream 시나리오에서 강력한 zero-/few-shot 성능을 보이는 혁명기를 지나고 있고, 그 성공은 데이터 분포를 폭넓게 덮는 대규모 학습 데이터에 기반한다. 로보틱스·자율주행·VR에 두루 쓰이는 **Monocular Depth Estimation(MDE)** 역시 한 장의 이미지에서 depth를 추정하는 foundation model이 필요하지만, 수천만 장 규모의 depth label 데이터셋을 만들기 어려워 이 방향은 미개척으로 남아 있었다. MiDaS가 여러 labeled 데이터셋을 섞어 학습하는 선구적 연구를 했으나, 데이터 커버리지의 한계로 일부 시나리오에서는 성능이 크게 무너진다.

이 논문의 목표는 **어떤 상황의 어떤 이미지에도 고품질 depth를 내는 MDE foundation model**이고, 접근은 **데이터셋 스케일업**이다. 전통적으로 depth 데이터셋은 depth 센서, stereo matching, SfM으로 만드는데 비용이 크거나 특정 상황에서는 아예 불가능하다. 저자들은 대신 처음으로 **대규모 unlabeled 데이터**에 주목한다. 단안(monocular) unlabeled 이미지는 세 가지 장점이 있다.

- **획득이 쉽고 싸다** — 특수 장비 없이 어디에나 존재
- **다양하다** — 훨씬 넓은 장면을 덮어 일반화·확장성에 결정적
- **annotation이 쉽다** — 사전학습된 MDE 모델의 forward 한 번이면 depth label이 생기고, LiDAR보다 dense하며 stereo matching 계산도 필요 없음

저자들은 unlabeled 이미지에 depth annotation을 자동 생성하는 **data engine**을 설계한다. SA-1B, Open Images, BDD100K 등 8개 공개 데이터셋에서 **62M 장**의 다양한 이미지를 모으고(label은 일절 쓰지 않음), annotation 도구로 쓸 초기 MDE 모델을 6개 공개 데이터셋의 **1.5M 장** labeled 이미지로 학습한다. unlabeled 이미지는 자동 annotation된 뒤 labeled 이미지와 함께 self-training 방식으로 공동 학습된다.

다만 labeled 이미지가 이미 충분하고 사전학습이 강한 상황에서 이런 대규모 unlabeled 이미지를 유익하게 쓰는 것은 간단치 않다 — 예비 실험에서 labeled + pseudo labeled를 단순 결합해서는 labeled만 쓴 baseline을 넘지 못했다. 이런 나이브한 self-teaching으로 얻는 추가 지식이 제한적이라고 보고, **pseudo label을 배울 때 student 모델에 더 어려운 최적화 목표를 던지는** 방법을 제안한다. 다양한 강한 perturbation 아래서 student가 추가적인 시각 지식을 찾고 robust한 표현을 배우도록 강제하는 것이다.

나아가 보조 semantic segmentation 과제가 MDE에 도움이 된다는 기존 연구를 따라가되, MDE 모델이 이미 충분히 강하면 그런 보조 과제로는 추가 이득이 어렵다는 것을 관찰한다. 이미지를 이산적인 class 공간으로 decode하면서 semantic 정보가 심하게 손실되기 때문이라 보고, semantic 관련 과제에서 뛰어난 **DINOv2의 풍부한 semantic prior를 단순한 feature alignment loss로 유지**하는 방법을 제안한다. 이는 MDE 성능을 높일 뿐 아니라 중간 수준·고수준 인식 과제를 아우르는 multi-task encoder를 만들어 낸다.

![어두운 실내, 복잡한 식탁 장면, 저조도 공연장, 건축물, 안개 낀 도시, 항공 뷰까지 다양한 미학습 장면 사진과 Depth Anything의 depth 예측 결과](/assets/images/depth-anything-fig1.jpg)
*[Figure 1] COCO(왼쪽 두 열)·SA-1B(가운데 두 열)·직접 촬영 사진(오른쪽 두 열) 등 광범위한 미학습 장면에 대한 일반화 능력 — 저조도, 복잡한 장면, 안개, 초원거리까지 robust한 동작*

기여를 요약하면 다음과 같다.

- 값싸고 다양한 대규모 unlabeled 이미지의 데이터 스케일업 가치를 부각
- 대규모 labeled + unlabeled 공동 학습의 핵심 실천법 제시 — unlabeled 이미지를 그대로 배우는 대신 **더 어려운 최적화 목표**로 모델에 도전
- 보조 segmentation 과제 대신 **사전학습 encoder의 semantic prior를 상속**하는 방법 제안
- MiDaS-BEiT$$_{\textrm{L-512}}$$ 보다 강한 zero-shot 능력을 보이고, metric depth로 fine-tuning하면 ZoeDepth를 크게 능가

## 2. Related Work

**Monocular depth estimation (MDE).** 초기 연구는 handcrafted feature와 전통 비전 기법에 의존했고, 명시적 depth 단서에 기대다 보니 가려짐(occlusion)이나 텍스처 없는 영역이 있는 복잡한 장면에 약했다. 딥러닝 기반 방법은 정밀하게 annotation된 데이터셋에서 depth 표현을 학습하며 이 분야를 혁신했다 — Eigen et al.의 multi-scale fusion 네트워크를 시작으로, regression을 classification으로 재설계하거나 prior와 목적 함수를 개선하는 연구가 이어졌다. 다만 이들은 미학습 도메인으로의 일반화가 어렵다.

**Zero-shot depth estimation.** 이 논문이 속한 연구 방향으로, 다양한 학습 데이터로 MDE 모델을 학습해 어떤 이미지든 depth를 예측하는 것이 목표다. 초기 연구들은 학습 이미지를 늘렸지만 supervision이 매우 sparse했다. 이정표가 된 MiDaS는 **affine-invariant loss**로 데이터셋마다 다른 depth의 scale·shift를 무시해 다중 데이터셋 공동 학습을 가능하게 했고, 따라서 **relative depth**를 제공한다. 최근에는 metric depth까지 추정하는 연구도 있으나 저자들의 실험에서는 MiDaS(특히 최신 버전)보다 일반화가 나빴다. ZoeDepth가 보였듯 강한 relative depth 모델은 metric depth 정보로 fine-tuning하면 일반화 가능한 metric depth 추정에도 잘 동작하므로, 이 논문도 MiDaS를 따라 relative depth 추정을 택하되 대규모 단안 unlabeled 이미지의 가치를 부각해 이를 더 강화한다.

**Leveraging unlabeled data.** semi-supervised learning의 영역이다. 기존 연구는 대개 이미지가 소량만 있는 상황을 가정하고, labeled 이미지가 이미 충분한데 더 큰 규모의 unlabeled 이미지도 있는 도전적이지만 현실적인 시나리오는 거의 다루지 않았다. 이 논문은 zero-shot MDE에서 이 방향을 택해, unlabeled 이미지가 데이터 커버리지를 크게 넓혀 모델의 일반화와 robustness를 끌어올릴 수 있음을 보인다.

## 3. Depth Anything

labeled 집합과 unlabeled 집합을 각각 $$\mathcal{D}^l = \{(x_i, d_i)\}_{i=1}^{M}$$, $$\mathcal{D}^u = \{u_i\}_{i=1}^{N}$$ 로 표기한다. 여기서 $$x_i$$ 는 이미지, $$d_i$$ 는 그 depth label, $$u_i$$ 는 label 없는 이미지이고 $$M = 1.5\text{M}$$, $$N = 62\text{M}$$ 이다. 흐름은 세 단계다 — ① $$\mathcal{D}^l$$ 에서 **teacher 모델 $$T$$** 를 학습하고, ② $$T$$ 로 $$\mathcal{D}^u$$ 에 pseudo depth label을 달고, ③ labeled + pseudo labeled 결합 집합에서 **student 모델 $$S$$** 를 학습한다.

![labeled 이미지는 실선, unlabeled 이미지는 점선 흐름으로 그려진 학습 pipeline. unlabeled 이미지에 강한 perturbation S를 가하고, teacher의 pseudo label로 supervision하며, frozen encoder와의 semantic preservation 제약이 함께 표시됨](/assets/images/depth-anything-fig2.png)
*[Figure 2] 전체 pipeline — 실선은 labeled 이미지, 점선은 unlabeled 이미지의 흐름, $$\mathcal{S}$$ 는 강한 perturbation*

### 3.1 Learning Labeled Images

이 과정은 MiDaS의 학습과 유사하다(MiDaS는 코드를 공개하지 않아 저자들이 먼저 재현했다). depth 값 $$t$$ 를 $$d = 1/t$$ 로 **disparity 공간**으로 바꾼 뒤 depth map마다 0~1로 정규화한다. 다중 데이터셋 공동 학습을 위해 각 샘플의 미지의 scale·shift를 무시하는 **affine-invariant loss**를 쓴다:

$$
\mathcal{L}_l = \frac{1}{HW}\sum_{i=1}^{HW} \rho(d_i^*, d_i),
$$

여기서 $$H, W$$ 는 이미지의 높이·너비(합은 전체 픽셀에 대해 돈다), $$d_i^*$$ 와 $$d_i$$ 는 픽셀 $$i$$ 의 예측과 ground truth다. $$\rho$$ 는 affine-invariant mean absolute error로 $$\rho(d_i^*, d_i) = \lvert \hat{d}_i^* - \hat{d}_i \rvert$$ 인데, $$\hat{d}_i^*$$ 와 $$\hat{d}_i$$ 는 예측과 ground truth를 각각 scale·shift 정규화한 값이다:

$$
\hat{d}_i = \frac{d_i - t(d)}{s(d)},
$$

$$
t(d) = \operatorname{median}(d), \qquad s(d) = \frac{1}{HW}\sum_{i=1}^{HW} \lvert d_i - t(d) \rvert.
$$

$$t(d)$$ 는 depth map의 중앙값(translation을 0으로), $$s(d)$$ 는 중앙값으로부터의 평균 절대 편차(scale을 1로)다. 이렇게 정렬하면 데이터셋마다 depth의 절대 크기·기준점이 달라도 같은 loss로 비교할 수 있다.

robust한 모델을 위해 6개 공개 데이터셋에서 1.5M labeled 이미지를 모았다(Table 1 상단). MiDaS v3.1의 12개보다 적은 이유는 ① NYUv2·KITTI를 zero-shot 평가용으로 남겨 두었고, ② Movies·WSVD 등 일부는 더 이상 구할 수 없으며, ③ RedWeb 등 일부는 품질이 낮기 때문이다. labeled 이미지가 적어도, 얻기 쉽고 다양한 unlabeled 이미지가 데이터 커버리지를 보완해 일반화와 robustness를 크게 높인다. teacher $$T$$ 를 강화하기 위해 encoder는 **DINOv2 사전학습 가중치**로 초기화하고, 실전에서는 사전학습된 semantic segmentation 모델로 하늘 영역을 찾아 disparity를 0(가장 먼 곳)으로 설정한다.

| Dataset | Indoor | Outdoor | Label | # Images |
| --- | :-: | :-: | :-: | --- |
| **Labeled Datasets** | | | | |
| BlendedMVS | ✓ | ✓ | Stereo | 115K |
| DIML | ✓ | ✓ | Stereo | 927K |
| HRWSI | ✓ | ✓ | Stereo | 20K |
| IRS | ✓ | | Stereo | 103K |
| MegaDepth | | ✓ | SfM | 128K |
| TartanAir | ✓ | ✓ | Stereo | 306K |
| **Unlabeled Datasets** | | | | |
| BDD100K | | ✓ | None | 8.2M |
| Google Landmarks | | ✓ | None | 4.1M |
| ImageNet-21K | ✓ | ✓ | None | 13.1M |
| LSUN | ✓ | | None | 9.8M |
| Objects365 | ✓ | ✓ | None | 1.7M |
| Open Images V7 | ✓ | ✓ | None | 7.8M |
| Places365 | ✓ | ✓ | None | 6.5M |
| SA-1B | ✓ | ✓ | None | 11.1M |

*[Table 1] 1.5M labeled 이미지와 62M unlabeled 이미지의 공동 학습 구성*

### 3.2 Unleashing the Power of Unlabeled Images

이 논문의 핵심이다. 다양한 labeled 데이터셋을 공들여 구축하던 기존 연구와 달리 **unlabeled 이미지로 데이터 커버리지를 넓히는 것**의 가치를 부각한다. 인터넷이나 여러 과제의 공개 데이터셋에서 다양하고 큰 unlabeled 집합을 실질적으로 만들 수 있고, 잘 학습된 MDE 모델에 forward만 하면 dense한 depth map을 손쉽게 얻는다 — stereo matching이나 SfM 재구성보다 훨씬 간편하고 효율적이다. 장면이 다양한 8개 대규모 공개 데이터셋(총 62M 장 이상, Table 1 하단)을 unlabeled 소스로 골랐다.

기술적으로는, 앞서 얻은 teacher $$T$$ 로 $$\mathcal{D}^u$$ 를 예측해 pseudo labeled 집합 $$\hat{\mathcal{D}}^u$$ 를 얻는다:

$$
\hat{\mathcal{D}}^u = \{(u_i,\, T(u_i)) \mid u_i \in \mathcal{D}^u\}_{i=1}^{N}.
$$

$$T(u_i)$$ 는 unlabeled 이미지 $$u_i$$ 에 대한 teacher의 depth 예측, 즉 pseudo label이다. labeled와 pseudo labeled의 결합 $$\mathcal{D}^l \cup \hat{\mathcal{D}}^u$$ 에서 student $$S$$ 를 학습하는데, 선행 연구를 따라 $$S$$ 를 $$T$$ 에서 fine-tuning하지 않고 **재초기화**한다.

그런데 pilot study에서 이런 self-training pipeline으로는 개선을 얻지 못했다 — labeled 이미지가 소량일 때의 관찰과 상반되는 결과다. 저자들의 추측은 이렇다: labeled 이미지가 이미 충분한 상황에서는 추가 unlabeled 이미지에서 얻는 지식이 제한적이고, 특히 teacher와 student가 같은 사전학습·구조를 공유하므로 명시적 self-training 없이도 unlabeled 집합에서 비슷하게 맞고 비슷하게 틀린다.

해법은 **student에게 더 어려운 최적화 목표를 던지는 것**이다. 학습 중 unlabeled 이미지에 강한 perturbation을 주입해, student가 추가 시각 지식을 능동적으로 찾고 불변(invariant) 표현을 얻도록 강제한다. perturbation은 두 형태다 — **강한 색 왜곡**(color jittering, Gaussian blurring)과 **강한 공간 왜곡**(CutMix). 단순한 두 변경만으로 대규모 unlabeled 이미지가 labeled baseline을 크게 개선하게 된다.

CutMix는 원래 image classification용으로 제안돼 MDE에서는 거의 쓰이지 않았다. 먼저 무작위 unlabeled 이미지 쌍 $$u_a, u_b$$ 를 공간적으로 보간한다:

$$
u_{ab} = u_a \odot M + u_b \odot (1 - M),
$$

여기서 $$M$$ 은 직사각형 영역이 1인 binary mask이고 $$\odot$$ 는 픽셀별 곱이다 — 즉 $$u_{ab}$$ 는 $$u_b$$ 위에 $$u_a$$ 의 직사각형 조각을 붙인 합성 이미지다. unlabeled loss $$\mathcal{L}_u$$ 는 $$M$$ 과 $$1-M$$ 이 정의하는 유효 영역에서 affine-invariant loss를 따로 계산한 뒤:

$$
\mathcal{L}_u^M = \rho\big(S(u_{ab}) \odot M,\; T(u_a) \odot M\big),
$$

$$
\mathcal{L}_u^{1-M} = \rho\big(S(u_{ab}) \odot (1-M),\; T(u_b) \odot (1-M)\big),
$$

(간결함을 위해 합 기호와 픽셀 첨자 $$i$$ 는 생략) 두 영역의 면적 비율로 가중 평균한다:

$$
\mathcal{L}_u = \frac{\sum M}{HW}\,\mathcal{L}_u^M + \frac{\sum(1-M)}{HW}\,\mathcal{L}_u^{1-M}.
$$

$$\sum M$$ 은 mask가 1인 픽셀 수이므로, 각 항은 해당 영역이 이미지에서 차지하는 비율만큼 기여한다. student 입력은 합성·색 왜곡된 이미지지만, **pseudo label을 만드는 teacher $$T$$ 에는 왜곡 없는 깨끗한 이미지**를 넣는다. CutMix는 50% 확률로 적용한다.

### 3.3 Semantic-Assisted Perception

보조 semantic segmentation 과제로 depth 추정을 개선한 연구들이 있고, 고수준 semantic 정보로 depth 모델을 무장시키는 것은 유익해 보인다. 게다가 unlabeled 이미지를 쓰는 이 논문의 맥락에서는 다른 과제의 보조 supervision이 pseudo depth label의 노이즈에 대항하는 역할도 할 수 있다.

첫 시도로 RAM + GroundingDINO + HQ-SAM 조합으로 unlabeled 이미지에 semantic segmentation label을 정성껏 달았다(후처리 후 4K class). 공동 학습 단계에서 shared encoder + 개별 decoder 둘로 depth와 segmentation을 모두 예측하게 했지만, 시행착오 끝에도 원래 MDE 모델의 성능을 끌어올리지 못했다. 이미지를 **이산 class 공간으로 decode하면서 semantic 정보가 너무 많이 손실**되고, 이미 매우 경쟁력 있는 depth 모델을 제한된 정보의 semantic mask로 더 밀어올리기는 어렵다는 추측이다.

그래서 더 정보량이 많은 semantic 신호를 찾는다. DINOv2가 frozen 가중치 그대로도 image retrieval·semantic segmentation 등에서 보이는 강력한 성능에 주목해, 그 semantic 능력을 **보조 feature alignment loss**로 depth 모델에 전이한다. feature 공간은 고차원·연속이라 이산 mask보다 풍부한 semantic 정보를 담는다:

$$
\mathcal{L}_{feat} = 1 - \frac{1}{HW}\sum_{i=1}^{HW} \cos(f_i,\, f'_i),
$$

여기서 $$\cos(\cdot,\cdot)$$ 는 두 feature 벡터의 cosine similarity, $$f$$ 는 depth 모델 $$S$$ 가 추출한 feature, $$f'$$ 는 **frozen DINOv2 encoder**의 feature다. 픽셀 $$i$$ 마다 두 feature가 닮을수록 loss가 작아진다. 일부 연구처럼 online feature $$f$$ 를 새 공간으로 projection하지 않는 이유는, 무작위 초기화된 projector가 초기 학습에서 큰 alignment loss로 전체 loss를 지배하기 때문이다.

feature alignment의 또 하나의 핵심: DINOv2 같은 semantic encoder는 물체의 서로 다른 부분(예: 자동차 앞면과 뒷면)에 비슷한 feature를 내는 경향이 있다. 그러나 depth 추정에서는 같은 부분 안의 픽셀들조차 depth가 다를 수 있으므로, frozen encoder와 **완전히 같은 feature를 강요하는 것은 이롭지 않다**. 이를 위해 **tolerance margin $$\alpha$$** 를 둔다 — $$f_i$$ 와 $$f'_i$$ 의 cosine similarity가 $$\alpha$$ 를 넘으면 그 픽셀은 $$\mathcal{L}_{feat}$$ 에서 제외한다. 이로써 DINOv2의 semantic-aware 표현과 depth supervision의 part-level 판별 표현을 모두 누린다. 부수 효과로, 만들어진 encoder는 downstream MDE뿐 아니라 semantic segmentation에서도 강력해 중간·고수준 인식을 아우르는 범용 multi-task encoder의 잠재력을 보인다.

최종 loss는 세 loss $$\mathcal{L}_l$$(labeled), $$\mathcal{L}_u$$(unlabeled), $$\mathcal{L}_{feat}$$(feature alignment)의 평균 결합이다.

## 4. Experiment

### 4.1 Implementation Details

feature 추출에는 DINOv2 encoder, depth regression에는 MiDaS를 따라 **DPT decoder**를 쓴다. labeled 데이터셋은 re-sampling 없이 단순 결합한다. 1단계에서 teacher를 labeled 이미지로 20 epoch 학습하고, 2단계 공동 학습에서 student가 전체 unlabeled 이미지를 한 번 훑는다. unlabeled 이미지의 annotation은 ViT-L encoder의 최고 성능 teacher가 담당한다. 배치 안 labeled:unlabeled 비율은 1:2. 두 단계 모두 사전학습 encoder의 base learning rate는 5e-6, 무작위 초기화된 decoder는 그 10배다. AdamW optimizer에 linear schedule로 learning rate를 감쇠하고, labeled 이미지의 augmentation은 horizontal flip만 쓴다. feature alignment의 tolerance margin은 $$\alpha = 0.85$$.

### 4.2 Zero-Shot Relative Depth Estimation

6개의 대표적인 미학습 데이터셋 — KITTI, NYUv2, Sintel, DDAD, ETH3D, DIODE — 에서 zero-shot depth 추정 능력을 종합 검증한다. 비교 대상은 더 많은 labeled 이미지를 쓴 최신 MiDaS v3.1의 최고 모델 DPT-BEiT$$_{\textrm{L-512}}$$ 다. 지표는 **AbsRel**(absolute relative error: $$\lvert d^* - d \rvert / d$$, 낮을수록 좋음)과 **$$\delta_1$$**($$\max(d^*/d,\, d/d^*) < 1.25$$ 인 픽셀 비율, 높을수록 좋음)이다.

| Method | Encoder | KITTI AbsRel↓ | KITTI $$\delta_1$$↑ | NYUv2 AbsRel↓ | NYUv2 $$\delta_1$$↑ | Sintel AbsRel↓ | Sintel $$\delta_1$$↑ | DDAD AbsRel↓ | DDAD $$\delta_1$$↑ | ETH3D AbsRel↓ | ETH3D $$\delta_1$$↑ | DIODE AbsRel↓ | DIODE $$\delta_1$$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiDaS v3.1 | ViT-L | 0.127 | 0.850 | 0.048 | <u>0.980</u> | 0.587 | 0.699 | 0.251 | 0.766 | 0.139 | 0.867 | 0.075 | 0.942 |
| **Depth Anything** | ViT-S | 0.080 | 0.936 | 0.053 | 0.972 | 0.464 | 0.739 | 0.247 | 0.768 | 0.127 | **0.885** | 0.076 | 0.939 |
| | ViT-B | <u>0.080</u> | <u>0.939</u> | <u>0.046</u> | 0.979 | **0.432** | <u>0.756</u> | <u>0.232</u> | <u>0.786</u> | **0.126** | <u>0.884</u> | <u>0.069</u> | <u>0.946</u> |
| | ViT-L | **0.076** | **0.947** | **0.043** | **0.981** | <u>0.458</u> | **0.760** | **0.230** | **0.789** | <u>0.127</u> | 0.882 | **0.066** | **0.952** |

*[Table 2] Zero-shot relative depth estimation — 굵게 최고, 밑줄 2위. MiDaS v3.1은 KITTI·NYUv2 학습 이미지를 사용해 이 둘에서는 엄밀한 zero-shot이 아님. 세 모델 규모는 ViT-S(24.8M), ViT-B(97.5M), ViT-L(335.3M)*

같은 ViT-L encoder에서 Depth Anything은 광범위한 장면에서 MiDaS 최강 모델을 크게 앞선다 — 예컨대 자율주행 데이터셋 DDAD에서 AbsRel 0.251 → 0.230, $$\delta_1$$ 0.766 → 0.789. 더욱이 **ViT-B 모델이 훨씬 큰 ViT-L 기반 MiDaS보다 이미 명백히 우세**하고, MiDaS의 1/10 크기도 안 되는 **ViT-S 모델조차** Sintel·DDAD·ETH3D 등 여러 데이터셋에서 MiDaS를 능가한다 — 계산 제약 시나리오에서의 잠재력을 보여준다. 가장 널리 쓰이는 벤치마크 KITTI·NYUv2에서는 MiDaS v3.1이 해당 학습 이미지를 쓴(더 이상 zero-shot이 아닌) 반면 Depth Anything은 한 장도 쓰지 않고도 명백히 우세하다(KITTI AbsRel 0.127 vs. 0.076, $$\delta_1$$ 0.850 vs. 0.947).

### 4.3 Fine-tuned to Metric Depth Estimation

zero-shot relative depth를 넘어, **downstream metric depth 추정의 가중치 초기화**로서의 가치를 검증한다. downstream MDE 모델의 encoder를 사전학습된 Depth Anything encoder(ViT-L)로 초기화하고 decoder는 무작위 초기화한 뒤, 해당 metric depth 정보로 fine-tuning한다. 두 시나리오를 본다 — ① 같은 도메인에서 학습·평가하는 in-domain, ② 한 도메인(예: NYUv2)에서 학습해 다른 도메인(예: SUN RGB-D)에서 평가하는 zero-shot metric depth다.

#### 4.3.1 In-Domain Metric Depth Estimation

| Method | $$\delta_1$$↑ | $$\delta_2$$↑ | $$\delta_3$$↑ | AbsRel↓ | RMSE↓ | log10↓ |
| --- | --- | --- | --- | --- | --- | --- |
| AdaBins | 0.903 | 0.984 | 0.997 | 0.103 | 0.364 | 0.044 |
| DPT | 0.904 | 0.988 | 0.998 | 0.110 | 0.357 | 0.045 |
| P3Depth | 0.898 | 0.981 | 0.996 | 0.104 | 0.356 | 0.043 |
| SwinV2-L | 0.949 | 0.994 | 0.999 | 0.083 | 0.287 | 0.035 |
| AiT | 0.954 | 0.994 | 0.999 | 0.076 | 0.275 | 0.033 |
| VPD | 0.964 | 0.995 | 0.999 | 0.069 | 0.254 | 0.030 |
| ZoeDepth* | 0.951 | 0.994 | 0.999 | 0.077 | 0.282 | 0.033 |
| **Ours** | **0.984** | **0.998** | **1.000** | **0.056** | **0.206** | **0.024** |

*[Table 3] NYUv2 fine-tuning·평가 결과 — \*: 저자 재현*

| Method | $$\delta_1$$↑ | $$\delta_2$$↑ | $$\delta_3$$↑ | AbsRel↓ | RMSE↓ | RMSE log↓ |
| --- | --- | --- | --- | --- | --- | --- |
| AdaBins | 0.964 | 0.995 | 0.999 | 0.058 | 2.360 | 0.088 |
| DPT | 0.959 | 0.995 | 0.999 | 0.062 | 2.573 | 0.092 |
| P3Depth | 0.953 | 0.993 | 0.998 | 0.071 | 2.842 | 0.103 |
| NeWCRFs | 0.974 | 0.997 | 0.999 | 0.052 | 2.129 | 0.079 |
| SwinV2-L | 0.977 | 0.998 | 1.000 | 0.050 | 1.966 | 0.075 |
| NDDepth | 0.978 | 0.998 | 0.999 | 0.050 | 2.025 | 0.075 |
| GEDepth | 0.976 | 0.997 | 0.999 | 0.048 | 2.044 | 0.076 |
| ZoeDepth* | 0.971 | 0.996 | 0.999 | 0.054 | 2.281 | 0.082 |
| **Ours** | **0.982** | **0.998** | **1.000** | **0.046** | **1.896** | **0.069** |

*[Table 4] KITTI fine-tuning·평가 결과 — \*: 저자 재현*

NYUv2에서 종전 최고 VPD를 크게 넘어 $$\delta_1$$ 0.964 → 0.984, AbsRel 0.069 → 0.056을 달성하고, KITTI에서도 $$\delta_1$$ 0.978 → 0.982로 비슷한 개선을 보인다. 이 시나리오에는 비교적 기본적인 depth 모델인 ZoeDepth framework를 썼으므로, 더 발전된 구조를 쓰면 결과가 더 좋아질 수 있다고 본다.

#### 4.3.2 Zero-Shot Metric Depth Estimation

ZoeDepth를 따라 zero-shot metric depth를 평가한다. ZoeDepth는 MiDaS 사전학습 encoder를 NYUv2(실내) 또는 KITTI(실외)의 metric depth로 fine-tuning하는데, 여기서는 MiDaS encoder를 Depth Anything encoder로 바꾸기만 하고 나머지는 그대로 둔다.

| Method | SUN RGB-D AbsRel↓ | SUN RGB-D $$\delta_1$$↑ | iBims-1 AbsRel↓ | iBims-1 $$\delta_1$$↑ | HyperSim AbsRel↓ | HyperSim $$\delta_1$$↑ | vKITTI 2 AbsRel↓ | vKITTI 2 $$\delta_1$$↑ | DIODE Outdoor AbsRel↓ | DIODE Outdoor $$\delta_1$$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ZoeDepth | 0.520 | 0.545 | 0.169 | 0.656 | 0.407 | 0.302 | 0.106 | 0.844 | 0.814 | 0.237 |
| **Depth Anything** | **0.500** | **0.660** | **0.150** | **0.714** | **0.363** | **0.361** | **0.085** | **0.913** | **0.794** | **0.288** |

*[Table 5] Zero-shot metric depth estimation — 앞 세 test set은 실내, 뒤 둘은 실외. ZoeDepth를 따라 실내 일반화에는 NYUv2 학습 모델, 실외 평가에는 KITTI 학습 모델 사용, ZoeDepth 결과는 저자 환경에서 재현한 수치*

실내·실외를 아우르는 광범위한 미학습 데이터셋에서 Depth Anything encoder가 MiDaS 기반 원본 ZoeDepth보다 나은 metric depth 모델을 만든다.

### 4.4 Fine-tuned to Semantic Segmentation

feature alignment 제약으로 사전학습 encoder의 semantic prior를 상속한 MDE encoder의 **semantic 능력**을 검증한다 — MDE encoder를 downstream semantic segmentation 데이터셋에 fine-tuning한다.

| Method | Encoder | mIoU (s.s.) | m.s. |
| --- | --- | --- | --- |
| Segmenter | ViT-L | - | 82.2 |
| SegFormer | MiT-B5 | 82.4 | 84.0 |
| Mask2Former | Swin-L | 83.3 | 84.3 |
| OneFormer | Swin-L | 83.0 | 84.4 |
| OneFormer | ConvNeXt-XL | 83.6 | 84.6 |
| DDP | ConvNeXt-L | 83.2 | 83.9 |
| Ours | ViT-L | **84.8** | **86.2** |

*[Table 7] Cityscapes semantic segmentation 전이 결과 — s.s./m.s.는 single-/multi-scale 평가, Mapillary 사전학습 미사용*

| Method | Encoder | mIoU |
| --- | --- | --- |
| Segmenter | ViT-L | 51.8 |
| SegFormer | MiT-B5 | 51.0 |
| Mask2Former | Swin-L | 56.4 |
| UperNet | BEiT-L | 56.3 |
| ViT-Adapter | BEiT-L | 58.3 |
| OneFormer | Swin-L | 57.4 |
| OneFormer | ConNeXt-XL | 57.4 |
| Ours | ViT-L | **59.4** |

*[Table 8] ADE20K semantic segmentation 전이 결과 — segmentation 모델은 Mask2Former*

Cityscapes에서 대규모 MDE 학습으로 얻은 encoder(86.2 mIoU)가 ImageNet-21K 대규모 사전학습 기반의 기존 encoder들 — Swin-L(84.3), ConvNeXt-XL(84.6) — 을 앞선다. ADE20K에서도 종전 최고 58.3 → 59.4로 같은 관찰이 성립한다. MDE와 semantic segmentation 모두에서의 우세를 볼 때, 이 사전학습 encoder는 중간·고수준 시각 인식 시스템의 범용 multi-task encoder로서 큰 잠재력이 있다.

### 4.5 Ablation Studies

별도 언급이 없으면 ViT-L encoder를 쓴다.

**각 학습 데이터셋의 zero-shot 전이.** 하나의 학습 데이터셋으로 relative MDE 모델을 학습해 6개 미학습 데이터셋에서 평가한다. 범용 MDE 시스템을 만들려는 후속 연구에 통찰을 주기 위한 실험이다.

| Training set | KITTI AbsRel↓ | KITTI $$\delta_1$$↑ | NYUv2 AbsRel↓ | NYUv2 $$\delta_1$$↑ | Sintel AbsRel↓ | Sintel $$\delta_1$$↑ | DDAD AbsRel↓ | DDAD $$\delta_1$$↑ | ETH3D AbsRel↓ | ETH3D $$\delta_1$$↑ | DIODE AbsRel↓ | DIODE $$\delta_1$$↑ | Mean AbsRel↓ | Mean $$\delta_1$$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BlendedMVS | *0.089* | 0.918 | 0.068 | 0.958 | *0.556* | 0.689 | *0.305* | *0.731* | 0.148 | 0.845 | 0.092 | 0.921 | *0.210* | *0.844* |
| DIML | 0.099 | 0.907 | <u>0.055</u> | *0.969* | 0.573 | 0.722 | 0.381 | 0.657 | <u>0.142</u> | <u>0.859</u> | 0.107 | 0.908 | 0.226 | 0.837 |
| HRWSI | 0.095 | 0.917 | 0.062 | 0.966 | <u>0.502</u> | <u>0.731</u> | <u>0.270</u> | <u>0.750</u> | 0.186 | 0.775 | <u>0.087</u> | <u>0.935</u> | <u>0.200</u> | <u>0.846</u> |
| IRS | 0.105 | 0.892 | *0.057* | <u>0.970</u> | 0.568 | 0.714 | 0.328 | 0.691 | 0.143 | 0.845 | 0.088 | 0.926 | 0.215 | 0.840 |
| MegaDepth | 0.217 | 0.741 | 0.071 | 0.953 | 0.632 | 0.660 | 0.479 | 0.566 | *0.142* | *0.852* | 0.104 | 0.910 | 0.274 | 0.780 |
| TartanAir | <u>0.088</u> | <u>0.920</u> | 0.061 | 0.964 | 0.602 | *0.723* | 0.332 | 0.690 | 0.160 | 0.818 | *0.088* | *0.928* | 0.222 | 0.841 |
| All labeled data | **0.085** | **0.934** | **0.053** | **0.971** | **0.492** | **0.748** | **0.245** | **0.771** | **0.134** | **0.874** | **0.070** | **0.945** | **0.180** | **0.874** |

*[Table 6] 각 labeled 학습 데이터셋(왼쪽)의 6개 미학습 데이터셋(위)으로의 zero-shot 전이 성능 — test set별 1·2·3위를 각각 굵게·밑줄·이탤릭으로 표시*

6개 학습 데이터셋 중 **HRWSI가 겨우 20K 장으로도 가장 강한 일반화**를 준다 — **데이터 다양성이 결정적**이라는 뜻이고, unlabeled 이미지를 쓰는 이 논문의 동기와 정확히 맞닿는다. MegaDepth처럼 이 6개 test set에서는 성능이 낮아도 고유한 강점(초원거리 건물의 거리 추정, Figure 1)이 있어 항공 장비 등에는 매우 유용한 데이터셋도 있다.

**① unlabeled 학습 시 student에 대한 도전과 ② semantic 제약의 효과.**

| $$\mathcal{L}_l$$ | $$\mathcal{L}_u$$ | $$\mathcal{S}$$ | $$\mathcal{L}_{feat}$$ | KI | NY | SI | DD | ET | DI |
| :-: | :-: | :-: | :-: | --- | --- | --- | --- | --- | --- |
| ✓ | | | | 0.085 | 0.053 | 0.492 | 0.245 | 0.134 | 0.070 |
| ✓ | ✓ | | | 0.085 | 0.054 | 0.481 | 0.242 | 0.138 | 0.073 |
| ✓ | ✓ | ✓ | | 0.081 | 0.048 | 0.469 | 0.235 | 0.134 | 0.068 |
| ✓ | ✓ | ✓ | ✓ | **0.076** | **0.043** | **0.458** | **0.230** | **0.127** | **0.066** |

*[Table 9] 강한 perturbation $$\mathcal{S}$$ 와 semantic 제약 $$\mathcal{L}_{feat}$$ 의 ablation — 지면 제약으로 AbsRel(↓)만 보고, 데이터셋명은 앞 두 글자로 축약*

pseudo label을 단 unlabeled 이미지를 단순 추가하는 것($$\mathcal{L}_u$$ 만)은 이득을 보장하지 않는다 — labeled 이미지가 이미 충분하기 때문이다. 그러나 re-training에서 unlabeled 이미지에 **강한 perturbation $$\mathcal{S}$$** 를 가하면 student가 추가 시각 지식을 찾고 더 robust한 표현을 배우도록 도전받아, 대규모 unlabeled 이미지가 일반화 능력을 크게 끌어올린다. 나아가 **semantic 제약 $$\mathcal{L}_{feat}$$** 은 unlabeled 이미지의 힘을 depth 추정에 더 증폭시키고, 4.4절에서 강조했듯 encoder가 multi-task 시각 시스템의 핵심 부품이 되게 한다.

**MiDaS 학습 encoder와의 downstream 비교.** MiDaS v3.1 학습 encoder와 downstream fine-tuning 성능을 비교하면(부록 Table 10), depth 추정과 semantic segmentation 모두에서 이 논문의 encoder가 크게 앞선다 — 예: NYUv2 $$\delta_1$$ 0.951 vs. 0.984, ADE20K mIoU 52.4 vs. 59.4.

**DINOv2와의 downstream 비교.** 최종 encoder는 DINOv2에서 fine-tuning한 것이므로 원본 DINOv2 encoder와도 비교한다(부록 Table 11). downstream metric depth 추정과 semantic segmentation 모두에서 원본 DINOv2보다 낫다 — DINOv2 가중치가 이미 매우 강한 초기화임에도, 대규모·고품질 MDE 학습이 downstream 전이 성능을 더 끌어올린다.

### 4.6 Qualitative Results

6개 미학습 데이터셋에 대한 예측을 시각화하면(Figure 3) 다양한 도메인의 test 이미지에 robust하다.

![KITTI, NYUv2, Sintel, DDAD, ETH3D, DIODE 여섯 데이터셋의 입력 이미지와 Depth Anything의 depth 예측](/assets/images/depth-anything-fig3.jpg)
*[Figure 3] 6개 미학습 데이터셋에서의 정성 결과*

MiDaS와의 비교(Figure 4)에서 더 정확한 depth 추정을 보이고, 예측 depth map을 조건으로 ControlNet으로 새 이미지를 합성하면 합성 품질도 더 낫다. 더 정확한 합성을 위해 Depth Anything 기반의 더 나은 depth-conditioned ControlNet을 재학습해, 이미지 합성과 video editing에 더 좋은 제어 신호를 제공한다.

![입력 이미지에 대한 Depth Anything과 MiDaS의 depth 예측 비교, 그리고 각 depth map을 조건으로 ControlNet이 합성한 새 이미지](/assets/images/depth-anything-fig4.jpg)
*[Figure 4] MiDaS와의 depth 예측 비교 및 depth 조건 ControlNet 이미지 합성 결과*

## 5. Conclusion

이 논문은 robust한 monocular depth estimation을 위한 매우 실용적인 해법 **Depth Anything**을 제시한다. 기존 연구와 달리 값싸고 다양한 unlabeled 이미지의 가치를 특별히 부각하고, 그 가치를 온전히 끌어내는 단순하지만 효과적인 두 전략 — ① unlabeled 이미지를 배울 때 더 도전적인 최적화 목표를 부과하고, ② 사전학습 모델의 풍부한 semantic prior를 보존 — 을 설계했다. 그 결과 Depth Anything 모델은 탁월한 zero-shot depth 추정 능력을 보이고, downstream metric depth 추정과 semantic segmentation 과제의 유망한 초기화로도 기능한다.
