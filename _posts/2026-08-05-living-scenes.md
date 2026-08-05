---
title: "[논문리뷰] Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments (CVPR 2024)"
excerpt: "SE(3)-equivariant 표현 하나로 시간차 scan의 instance matching·registration·reconstruction을 함께 푸는 MoRE² 논문 리뷰"
date: 2026-08-05 19:00:00 +0900
categories:
  - 논문리뷰
  - CV
tags:
  - 3D scene understanding
  - point cloud registration
  - SE(3) equivariance
  - neural implicit representation
  - instance matching
toc: true
toc_sticky: true
---

> 시간이 흐르며 물체가 옮겨지는 실내 환경을 "living scene"으로 보고, 드문드문 찍힌 scan들로부터 물체 단위의 relocalization과 reconstruction을 함께 푸는 MoRE²(CVPR 2024 Highlight) 논문을 리뷰한다. 핵심은 encoder-decoder 네트워크 하나의 equivariant 표현으로 matching, registration, reconstruction 세 태스크를 모두 해결하는 것이다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments](https://arxiv.org/abs/2312.09138) |
| 저자 | Liyuan Zhu, Shengyu Huang, Konrad Schindler, Iro Armeni |
| 소속 / 연도 | Stanford University, ETH Zurich, 2024 (arXiv:2312.09138, CVPR 2024 Highlight) |
| 분야 | Dynamic 3D Scene Understanding |
| 코드 | [GitHub](https://github.com/GradientSpaces/LivingScenes) |

## 1. Introduction

3D scene reconstruction은 mixed reality, 내비게이션, embodied perception 같은 컴퓨터 비전·로보틱스 응용의 기반이다. 이 응용들 다수는 같은 환경에서 청소나 물건 찾기 같은 비슷한 작업을 반복하므로, 서로 다른 시점에 획득한 여러 3D scan에 걸쳐 누적된 환경 이해가 있으면 이득을 본다. 이런 누적적 scene 이해는 시간이 지날수록(특히 이전에 안 보이던 부분이 드러날수록) 물체의 기하적 완전성과 정확도를 높여 상호작용을 개선하고, 물체가 환경 안에서 어떻게 재배치되는지에 대한 이해를 키울 수 있다. 이를 **multi-object relocalization과 reconstruction** 문제로 정식화할 수 있다. 여기서 object relocalization은 두 scan 사이에 물체가 겪은 6DoF 움직임을 배경 기준으로 추정하는 것이다.

기존 동적 scene 연구 대부분은 실시간 관측을 다뤄 relocalization이 tracking으로 환원된다. 반면 센서 데이터가 상시가 아니라 불규칙한 시점에만 얻어지는 **장기(long-term) 동역학**을 다루는 연구는 적다. 촬영 사이 간격이 길어 중간 움직임의 modeling이 불가능하므로, 대안적 방법들은 점·물체 matching과 물체별 registration으로 relocalization을 푼다. 시간에 걸친 relocalization과 reconstruction을 **함께** 다루는 것은 거의 간과되어 왔고, 태스크마다 별도 방법을 쓰면 오차가 커진다.

![living scene 개념과 MoRE²의 네 단계](/assets/images/living-scenes-fig1.jpg)
*[Figure 1] 두 시점 scan(a)에서 instance matching(b), registration & reconstruction(c), object 중심 accumulation(d)으로 이어지는 living scene 구성*

이 간극을 메우기 위해 논문은 **MoRE²** 를 제안한다. 긴 시간 간격과 희소한 관측으로부터 진화하는 환경의 multi-object relocalization과 reconstruction을 수행해 living scene을 만드는 방법이다. 건물을 살아있는 유기체처럼 다뤄 생애 전체의 디지털 트윈을 유지하려는 living building information model 개념에서 영감을 받았다. Figure 1(a)처럼 서로 다른 시점에 획득되고 instance로 분할된 여러 3D point cloud를 입력받아, 모든 instance의 matching(b), registration(c 위), reconstruction(c 아래)이라는 연결된 세 태스크를 풀어 living scene을 만든다. 합성 데이터로만 학습한 단일 encoder-decoder 네트워크로 실세계 scan에서 이 태스크들을 풀 수 있으며, 중심에는 SE(3)-equivariant 표현이 있다. 또한 같은 instance에서 나온 point cloud들을 여러 scan에 걸쳐 누적하는 최적화 기법을 도입한다. ShapeNet 물체로 생성한 합성 데이터셋과 실세계 3RScan에서 평가해 end-to-end와 개별 subtask 모두에서 state-of-the-art를 달성한다.

기여는 다음과 같다.

1. 진화하는 3D 실내 환경을 living scene으로 해석하는 새로운 object 중심 정식화. instance matching, relocalization, reconstruction을 포함
2. 세 태스크를 동시에 다루는 새로운 compact object 표현. 합성 데이터로만 학습해도 실제 scene으로 일반화
3. 데이터가 누적될수록 point cloud registration과 reconstruction 성능을 점진적으로 개선하는 joint optimization 알고리즘

## 2. Related Work

**Dynamic point cloud understanding.** 동적 scene은 주로 여러 움직이는 instance와 정적 배경으로 구성된다. 이런 환경의 modeling은 보통 점별 움직임, 즉 scene flow 추정에서 시작하지만, 기존 scene flow 방법은 instance 개념이 없어 downstream 태스크에 쓸 고수준 표현이 부족하다. multi-body scene과 물체 articulation을 위해 분할과 움직임 추정을 분리하는 방법들, detection과 motion model을 결합한 3D multi-object tracking 방법들도 있으나, 이들은 자율주행 데이터셋처럼 짧은 시간 범위의 고빈도 관측에 의존한다. 장기 변화 실내 환경 연구를 위해 3RScan, ReScan, NSS 같은 데이터셋이 구축되었는데, 관측 사이 변화가 극심해 tracking 기반 방법을 적용할 수 없다. Adam et al.은 기하 변환 일관성 기반 3D 변화 감지를 다뤘지만 instance·semantic 개념이 없고, Halber et al.은 시공간 모델로 instance 분할을 개선했지만 relocalization·reconstruction은 다루지 않았다. Wald et al.은 두 시점 point cloud에서 instance relocalization을 다루며 patch 사이 3D keypoint 대응을 찾는 triplet 네트워크를 도입했다. 이에 비해 MoRE²는 국소 matching을 넘어 instance 전체 수준의 6DoF 변환을 추론하고, 기존의 어떤 장기 변화 실내 환경 방법도 다루지 않은 reconstruction까지 수행한다.

**SO(3)-equivariant networks.** 무작위로 회전된 point cloud는 좌표 기반 네트워크의 예측을 불안정하게 만든다. 학습 중 강한 데이터 증강으로 일부 완화할 수 있지만, 회전을 보존하는 SO(3)-equivariance 자체가 바람직한 성질이 된다. Thomas et al.은 spherical harmonics로 SE(3) equivariance를 제약했고, SO(3)-transformer는 self-attention에 equivariance를 도입해 효율을 크게 높였다. Deng et al.의 **Vector Neurons(VNN)** 는 scalar neuron을 벡터화해 MLP 기반 네트워크를 SO(3)-equivariant로 만드는 일반 framework다. GraphOnet은 이를 SE(3)로 확장했고, Assaad et al.은 rotation equivariant attention을 적용했다. EFEM은 VNN에 shape prior를 저장해 비지도 물체 분할을 수행했고, 이외에 물체 articulation·다중 물체 분할, 로봇 policy 학습으로의 확장이 있다. 이 논문은 VNN을 여러 물체의 동역학 이해에 활용한다.

**Point cloud registration.** pose가 다른 point cloud 정렬은 3D 인식과 mapping에 필수다. FPFH, SHOT 같은 수작업 descriptor에서 출발해 3DMatch, PerfectMatch, RPMNet 같은 학습 기반 descriptor로 발전했다. Predator는 낮은 overlap 영역의 3D 대응 탐색에 attention을 도입했고, 이후 transformer 구조가 superpoint matching으로 정제되었으며, 표면 곡률·2D 이미지 overlap·scene 구조 같은 추가 prior를 쓰는 연구도 있다. 대응 matching과 별개로, equivariant 표현을 학습해 상대 pose를 직접 푸는 연구들이 있다(group equivariant 학습 기반 descriptor, rotation invariant transformer, rotation equivariant embedding으로 pairwise rotation을 직접 풀기 등). 이 논문도 equivariant 표현을 쓰지만, 추가로 neural implicit surface를 활용해 테스트 시점 최적화로 두 point cloud를 정렬한다.

**Multi-object reconstruction.** neural implicit reconstruction의 등장은 학습된 shape prior와 테스트 시점 최적화를 위한 미분 가능성 덕에 물체 reconstruction의 성능과 유연성을 끌어올렸다. FroDo는 detection과 multi-view 최적화로 물체 shape을 복원하고, ELLIPSDF와 ODAM은 superquadric으로 shape primitive를 표현해 multi-view 최적화를 제약한다. Irshad et al.은 기존 detector 없이 물체 pose·shape·외형을 single-shot으로 regression하고, BundleSDF는 단일 동적 물체에 집중해 graph 최적화로 미지 물체에 일반화한다. 이 논문의 reconstruction은 DeepSDF 위에 구축되며, 여러 관측을 모으는 shape·pose joint optimization을 추가로 도입한다.

## 3. Living Scenes

living scene을 동적·정적 물체가 있는 건조 환경으로 정의한다. 그 reconstruction은 시간에 걸친 scan들로부터 누적적으로 이뤄지며 환경이 어떻게 사용되어 왔는지 보여준다. MoRE²는 물체의 rigid 움직임과 기하를 이해하도록 설계되어, 각 3D 물체를 개별적으로 재구성하고, scan이 쌓일수록 정확도와 완전성이 올라간다.

**Problem Setting.** 불규칙한 간격으로 촬영된 scan 집합 $$\{\mathbf{S}^t\}_{t=1}^{T}$$ 을 생각하자. scan $$\mathbf{S}^t$$ 는 시각 $$t$$ 에 관측된 환경이고 point cloud 목록 $$\{\mathbf{X}_i^t\}_{i=1}^{N}$$ 을 담는다. 이후 물체 수준 point cloud를 point cloud, scene 수준 point cloud를 scan이라 부른다. 첫 scan $$\mathbf{S}^1$$ 을 reference scan, 이후를 temporal rescan $$\{\mathbf{S}^t \mid t > 1\}$$ 이라 한다. 목표는 두 가지다.

1. **Multi-object relocalization**: reference scan과 rescan에서 같은 instance에 속한 point cloud 사이의 6DoF rigid 변환 $$\{\mathbf{T}_i^t \in \mathrm{SE}(3) \mid t > 1\}$$ 을 계산한다. 구체적으로 point cloud의 matching과 registration 두 단계로 정식화한다
2. **Object reconstruction**: 누적된 point cloud $$\{\mathbf{X}^1 \circ \mathbf{T}_i^2 \mathbf{X}^2 \circ \cdots \circ \mathbf{T}_i^t \mathbf{X}^t \mid t > 1\}$$ 로부터 각 instance를 재구성한다. $$\circ$$ 는 이어붙이기(concatenation) 연산이다

방법이 instance 수준에서 동작하므로 instance segmentation mask가 주어진다고 가정한다(semantic label에는 무관하며, 예측 mask를 쓰는 실험도 제공한다).

**Method Overview.** MoRE²는 물체 reconstruction 태스크만으로 학습된 encoder-decoder 네트워크(3.1절)에서 얻은 단일 compact 표현으로 instance matching(3.2절), registration(3.3절), reconstruction(3.4절)을 차례로 해결한다. 최종 instance별 누적 point cloud는 shape-pose joint optimization(3.5절)으로 얻는다.

![MoRE² pipeline 개요](/assets/images/living-scenes-fig2.png)
*[Figure 2] VN encoder embedding으로 Hungarian matching(a), Kabsch 초기화 + 최적화 registration(b), joint optimization(c), SDF decoding reconstruction(d)을 수행하는 전체 pipeline*

### 3.1 Encoder-decoder Network

**Vector Neuron Encoder (VN).** point cloud에서 회전 equivariant feature를 얻기 위해 Deng et al.을 따라 neuron 표현을 scalar에서 벡터로 "끌어올려" 순전파 중 회전을 보존한다. 함수 $$f$$ 의 SO(3) equivariance와 invariance는 다음과 같이 표현된다.

$$
f(\mathbf{R}\mathbf{X}) = \mathbf{R} f(\mathbf{X}), \qquad f(\mathbf{R}\mathbf{X}) = f(\mathbf{X})
$$

여기서 $$\mathbf{R}$$ 은 입력 point cloud $$\mathbf{X}$$ 에 가한 회전이다. 앞 식이 equivariance(입력을 돌리면 출력도 똑같이 돈다), 뒷 식이 invariance(입력을 돌려도 출력이 그대로)다. EFEM은 스케일 인자와 무게중심을 추가로 추정해 VNN을 SO(3)에서 SIM(3) equivariance로 확장했다. encoder $$\Phi$$ 는 point cloud $$\mathbf{X}$$ 를 입력받아 $$\mathbf{F} = (\mathbf{F_{inv}} \in \mathbb{R}^{256}, \mathbf{F_{eqv}} \in \mathbb{R}^{3 \times 256}, \mathbf{F_s} \in \mathbb{R}_+, \mathbf{F_c} \in \mathbb{R}^3)$$ 을 출력한다. 네 성분은 각각 invariant embedding, equivariant embedding, 스케일 인자, 무게중심이다. SIM(3) equivariance는 다음으로 달성된다.

$$
\Phi(s \mathbf{R} \mathbf{X} + \mathbf{t}) = (\mathbf{F_{inv}},\; \mathbf{R} \mathbf{F_{eqv}},\; s \mathbf{F_s},\; \mathbf{F_c} + \mathbf{t})
$$

여기서 $$(s, \mathbf{R}, \mathbf{t}) \in \mathrm{SIM}(3)$$ 은 스케일·회전·이동이다. 즉 입력에 similarity 변환을 가하면 equivariant embedding은 같은 회전을, 스케일과 무게중심은 같은 스케일·이동을 그대로 따라가고 invariant embedding은 변하지 않는다. embedding $$\mathbf{F}$$ 는 query 점 $$\mathbf{p} \in \mathbb{R}^3$$ 을 정준화(canonicalize)하는 데 쓰인다.

$$
\mathbf{F_q} = \langle \mathbf{F_{eqv}},\; (\mathbf{p} - \mathbf{F_c}) / \mathbf{F_s} \rangle
$$

여기서 $$\langle \cdot, \cdot \rangle$$ 는 채널별 내적이다. 무게중심을 빼고 스케일로 나눈 query를 equivariant embedding과 내적하므로, $$\mathbf{F_q}$$ 는 물체의 pose와 무관한 정준 좌표의 feature가 된다. 이것이 decoder에 입력된다.

**Neural Implicit Decoder.** decoder로는 DeepSDF를 쓴다. latent code와 query 좌표를 입력받아 그 위치의 SDF 값을 출력하는 auto-decoder 네트워크다.

$$
\mathbf{SDF}(\mathbf{p}) = \Psi(\mathbf{F_{inv}}, \mathbf{F_q})
$$

여기서 $$\Psi$$ 는 DeepSDF decoder다. DeepSDF로 학습된 latent 공간은 shape interpolation과 테스트 시점 최적화의 튼튼한 기반이 된다.

**Training.** 개별 shape에 대한 L1 reconstruction loss로 학습한다.

$$
\mathcal{L}_{\mathrm{recon}} = \frac{1}{K} \sum_{i=1}^{K} \lvert \overline{\mathbf{SDF}}(\mathbf{p}_i) - \mathbf{SDF}(\mathbf{p}_i) \rvert
$$

여기서 $$\overline{\mathbf{SDF}}(\mathbf{p})$$ 는 ground-truth SDF 값, $$\mathbf{SDF}(\mathbf{p})$$ 는 예측 SDF 값, $$\mathbf{p}_i$$ 는 샘플링된 점, $$K$$ 는 SDF 샘플 수다. DeepSDF·MendNet과 달리 여러 클래스를 한꺼번에 학습해 category-agnostic으로 만든다. 학습이 끝나면 가중치를 고정하고, 이후에는 네트워크와 embedding을 유연하게 적응시켜 relocalization·reconstruction 태스크를 푼다.

### 3.2 Instance Matching

무작위 방향의 두 point cloud 집합 $$\{\mathbf{X}_i^{t_1}\}_{i=1}^{N}, \{\mathbf{X}_j^{t_2}\}_{j=1}^{M}$$ (크기 $$N, M$$)이 주어지면 이들을 시간에 걸쳐 연결해야 한다.

먼저 모든 invariant embedding 쌍 $$\{\langle \mathbf{F_{inv}}_i^{t_1}, \mathbf{F_{inv}}_j^{t_2} \rangle\}_{i,j=1}^{N,M}$$ 의 cosine 유사도 행렬 $$\mathbf{\Lambda} \in \mathbb{R}^{N \times M}_+$$ 을 초기 score 행렬로 계산한다. 다음으로, equivariant embedding $$\mathbf{F_{eqv}}$$ 는 feature 차원을 따라 일대일 대응이 있는 latent 공간의 3D 좌표로 볼 수 있으므로, 모든 equivariant embedding 쌍 사이의 회전 $$\mathbf{R}_{i,j}$$ 를 Kabsch 알고리즘으로 추출해 쌍마다 회전을 제거한다. 그 후 정렬 잔차 행렬 $$\mathbf{E} \in \mathbb{R}^{N \times M}_+$$ 을 계산한다.

$$
\mathbf{E}(i, j) = \| \mathbf{R}_{i,j} \mathbf{F_{eqv}}_{,i}^{t_1} - \mathbf{F_{eqv}}_{,j}^{t_2} \|_2
$$

이 값은 $$\mathbf{R}_{i,j}$$ 로 SO(3) feature 공간에서 거칠게 정렬한 뒤 equivariant 쌍이 얼마나 안 맞는지(inverse fitness)를 나타낸다. 마지막으로 종합 matching score 행렬 $$\mathbf{H} = \mathbf{\Lambda} \oslash \mathbf{E}$$ 를 계산한다. $$\oslash$$ 는 행렬의 원소별 나눗셈이다. 이제 총 matching score $$\sum_{i,j} \mathbf{H}_{i,j} \mathbf{P}_{i,j}$$ 를 최대화하는 할당($$\mathbf{P}$$ 는 할당 행렬)을 찾으면 되는데, 기존 2D/3D feature matching 방식처럼 Hungarian matching으로 이 선형 할당을 푼다. 두 집합의 instance 수가 다를 수 있어 매칭되지 않는 instance가 남으면, 시간상 등장 순서에 따라 제거(removed) 또는 추가(added)로 처리한다.

### 3.3 Instance Registration

매칭된 쌍 $$(\mathbf{X}^{t_1} \in \mathbb{R}^{3 \times N_1}, \mathbf{X}^{t_2} \in \mathbb{R}^{3 \times N_2})$$ 에 대해 source $$\mathbf{X}^{t_2}$$ 를 target $$\mathbf{X}^{t_1}$$ 에 정렬하는 상대 변환 $$\mathbf{T} = (\mathbf{R}, \mathbf{t})$$ 를 추정한다. 이를 위해 최적화 기반 registration을 제안한다. 먼저 각 point cloud의 SE(3)-equivariant embedding $$\mathbf{F}_{se3} = \mathbf{F_{eqv}} + \mathbf{F_c}$$ 를 계산하고 Kabsch 알고리즘으로 $$(\mathbf{R}, \mathbf{t})$$ 를 풀어 초기화로 삼는다. 다음으로 $$\mathcal{L}_{\mathrm{reg}}$$ 를 최소화해 최적 변환을 얻는다.

$$
(\mathbf{R}^*, \mathbf{t}^*) = \operatorname*{arg\,min}_{(\mathbf{R}, \mathbf{t})} \mathcal{L}_{\mathrm{reg}}(\mathbf{X}^{t_1}, \mathbf{X}^{t_2})
$$

$$\mathcal{L}_{\mathrm{reg}}$$ 는 다음과 같이 정의된다.

$$
\mathcal{L}_{\mathrm{reg}}(\mathbf{X}^{t_1}, \mathbf{X}^{t_2}) = \underbrace{\| \Psi(\mathbf{F}^{t_1}_{\mathbf{inv}},\; \langle \mathbf{F}^{t_1}_{\mathbf{eqv}},\; (\mathbf{R}_{\mathrm{i}} \mathbf{X}^{t_2} + \mathbf{t}_{\mathrm{i}} - \mathbf{F}^{t_1}_{\mathbf{c}}) / \mathbf{F}^{t_1}_{\mathbf{s}} \rangle) \|_1}_{\mathcal{L}_{\mathrm{sdf}}} + \underbrace{\tilde{CD}(\mathbf{R}_{\mathrm{i}} \mathbf{X}^{t_2} + \mathbf{t}_{\mathrm{i}},\; \mathbf{X}^{t_1})}_{\mathcal{L}_{\mathrm{cd}}}
$$

여기서 $$\mathcal{L}_{\mathrm{sdf}}$$ 는 point cloud와 zero level-set 사이의 어긋남이다. $$i$$ 번째 반복의 현재 추정 $$(\mathbf{R}_{\mathrm{i}}, \mathbf{t}_{\mathrm{i}})$$ 로 변환한 source를 target embedding으로 정준화해 decoder $$\Psi$$ 에 넣으면, target의 SDF가 0이어야 할 표면 위 점들의 SDF 값이 나오므로 그 절댓값이 곧 정렬 오차가 된다. $$\mathcal{L}_{\mathrm{cd}}$$ 는 현재 추정과 target point cloud 사이의 Chamfer loss($$\tilde{CD}$$)다. $$(\mathbf{R}, \mathbf{t})$$ 는 TorchLie를 사용한 SE(3) manifold 위의 역전파로 직접 최적화해 더 빠르고 안정적으로 수렴시킨다. 이 최적화는 source point cloud를 target의 zero level-set에 반복적으로 정렬하면서 점별 어긋남도 함께 줄인다. 최적화 후 iterative closest point(ICP)로 정렬을 정제해 최종 출력을 얻는다. 나아가 변환 거리에 threshold를 걸어 scene의 정적/동적 물체를 분류할 수 있다.

### 3.4 Instance Reconstruction

매칭·정렬된 point cloud 쌍 $$\{(\mathbf{X}^{t_1}, \mathbf{X}^{t_2}, \mathbf{R}, \mathbf{t})_i\}_{i=0}^{M}$$ 을 얻은 뒤 재구성을 진행한다. 먼저 누적 point cloud를 farthest point sampling(FPS)으로 downsampling하고, downsampling된 point cloud에서 새 embedding $$\mathbf{F}_*$$ 를 계산한다. 마지막으로 $$\mathbf{F}_*$$ 와 DeepSDF decoder로 $$64^3$$ 해상도 voxel grid의 SDF 값을 query하고, Multi-resolution IsoSurface Extraction(MISE)으로 zero level-set을 추출해 물체 reconstruction으로 삼는다.

### 3.5 Joint Optimization for Accumulation

지금까지는 두 시점 scan 사이의 relocalization과 reconstruction이었다. 여러 scan의 관측을 활용하기 위해, registration과 reconstruction을 정제하고 시간에 따라 기하적 정확도·완전성이 증가하도록 point cloud를 누적하는 joint optimization 알고리즘을 제안한다.

**Initialization.** 매칭·정합된 point cloud들 $$\{\mathbf{X}^t\}_{t=t_1}^{t_K}$$ 과 VN-encoder의 equivariant·invariant embedding을 생각하자. 각 point cloud의 $$\mathcal{L}_{\mathrm{sdf}}$$ 값을 계산해, 자신의 embedding이 정의하는 zero level set과 가장 잘 맞는 $$\mathbf{X}^*$$ 를 초기화로 고른다. 구체적으로 $$\mathbf{F}$$ 를 그 equivariant embedding $$\mathbf{F}_{\mathbf{eqv}}^*$$ 로 초기화하고, pose graph $$\mathbf{G} = \{\mathbf{T}^t\}_{t=t_1}^{t_K}$$ 를 구성한다. $$\mathbf{T}^t$$ 는 $$\mathbf{X}^t$$ 를 $$\mathbf{X}^*$$ 에 정렬하는 변환으로, 앞의 registration 방법으로 계산한다.

**Optimization Objectives.** 공유 equivariant embedding $$\mathbf{F}$$ 와 pose graph $$\mathbf{G}$$ 를 $$\mathcal{L}_{joint} = \mathcal{L}_{\mathrm{sdf}} + \mathcal{L}_z$$ 를 최소화하며 함께 최적화한다. $$\mathcal{L}_{\mathrm{sdf}}$$ 는 누적 point cloud의 SDF 오차이고, $$\mathcal{L}_z = \| \mathbf{F}' - \mathbf{F} \|_2$$ 는 초기 $$\mathbf{F}$$ 대비 변화를 제약하는 정규화 항이다($$\mathbf{F}'$$ 은 현재 embedding). registration과 마찬가지로 pose graph는 SE(3) manifold에서, embedding은 Adam optimizer로 200회 반복 최적화한다.

### 3.6 Implementation Details

encoder와 decoder로 각각 VN Transformer와 DeepSDF를 쓴다. PyTorch로 구현해 NVIDIA A100(80GB) GPU 한 장에서 batch size 64로 $$2 \times 10^5$$ 회 학습하고, learning rate 0.0001을 $$1.2 \times 10^5$$, $$1.5 \times 10^5$$, $$1.8 \times 10^5$$ 회에서 0.3배씩 감쇠한다.

## 4. Experiments

multi-object relocalization과 reconstruction의 end-to-end 성능(4.3절)과 세 subtask 각각(4.4~4.6절)을 평가한다. end-to-end 평가에서는 앞 태스크의 출력을 다음 태스크의 입력으로 넘겨 noise와 오차를 그대로 물려받게 한다. 개별 태스크 평가에서는 ground truth를 입력으로 준다(registration에는 올바른 match를, reconstruction에는 올바른 registration 쌍을). 태스크별 최고 성능 baseline들을 조합해 end-to-end baseline으로 삼는다.

### 4.1 Datasets

MoRE²와 baseline 모두 합성 데이터셋으로 학습하고 합성·실세계 데이터셋 모두에서 테스트해 일반화 능력을 평가한다. 장기 변화를 보이는 합성 실내 3D 데이터셋이 없어 직접 합성했다.

**FlyingShape.** ShapeNet subset(chair, table, sofa, pillow, bench, couch, trash can의 7개 카테고리)으로 합성한 데이터셋이다. subset의 테스트셋에서 물체를 무작위로 뽑아 100개의 고유한 3D scene을 구성한다. 물체가 바닥에 닿도록 무작위 pose를 주고, 장기 동역학을 모사하기 위해 모든 물체 pose에 무작위 변화를 주며 scene당 5개의 temporal scan을 순차 생성한다.

**3RScan.** 물체 instance relocalization benchmark용 실세계 데이터셋으로, 478개 실내 scene의 rescan을 포함한 1428개 RGB-D 시퀀스로 구성된다. temporal scan 사이의 instance segmentation, 연결 관계, 변환이 주석되어 있다. 3RScan의 validation set으로 평가하며(테스트셋 ground truth 서버는 더 이상 유지되지 않음), 종합 태스크 평가를 위해 3RScan을 instance matching·reconstruction benchmark로 확장했다.

### 4.2 Evaluation Metrics

instance matching은 이미지 feature matching 평가에서 착안해 올바른 match 비율인 instance 수준 matching recall을 계산한다. scene 수준 matching recall도 계산하는데, 두 scan 사이 전체 match 중 올바른 match의 최소 허용 비율로 25%, 50%, 75%, 100%를 threshold로 쓴다. point cloud registration은 표준 관행을 따라 registration recall(RR), 중앙값 rotation error(MedRE), 변환 오차(RMSE), 중앙값 Chamfer Distance(MedCD)를 보고한다. RR의 rotation error threshold는 FlyingShape에서 $$5\degree$$, ground truth 주석 정확도가 낮은 3RScan에서 $$10\degree$$ 다. instance reconstruction은 Chamfer Distance와 volumetric IoU를 보고하고, instance 수준 reconstruction 성공 비율을 나타내는 SDF recall을 새로 도입한다.

**End-to-end Metrics.** end-to-end 성능 평가를 위해 MR recall과 MRR recall 두 결합 metric을 제안한다. MR은 relocalization의 end-to-end recall로

$$
\mathrm{P}(R_1, M) = \mathrm{P}(R_1 \mid M)\, \mathrm{P}(M)
$$

여기서 $$M$$ 과 $$R_1$$ 은 각각 instance가 올바르게 매칭·정합되는 사건, $$\mathrm{P}(\cdot)$$ 는 사건의 확률이다. matching 출력을 registration에 넘겨 계산한 RR이 MR recall이 되어 두 태스크의 오차를 모두 포함한다. 비슷하게 MRR은

$$
\mathrm{P}(M, R_1, R_2) = \mathrm{P}(R_2 \mid R_1, M)\, \mathrm{P}(R_1 \mid M)\, \mathrm{P}(M)
$$

여기서 $$R_2$$ 는 instance가 올바르게 재구성되는 사건이다. 예측 match와 그로부터의 registration을 reconstruction에 넘기고 SDF recall을 MRR로 쓴다. 이렇게 세 태스크 전체의 성능이 하나의 metric으로 평가된다.

### 4.3 End-to-end Performance

태스크별 최고 baseline인 VN-ONet, GeoTransformer, ConvONet을 조합해 end-to-end baseline(Baseline†)으로 삼는다. Table 1처럼 MoRE²가 모든 metric에서 일관되게 앞선다. FlyingShape에서 3RScan으로 갈 때 두 방법 모두 domain gap으로 비슷한 성능 하락을 보인다. 주목할 점은 개별 태스크 결과(4.4~4.6절) 대비 end-to-end 평가에서 baseline과의 격차가 더 벌어진다는 것이다. 이는 shape과 pose 정보를 유지하는 단일 네트워크·표현을 쓰는 통합 접근 덕분으로, 서로 다른 네트워크·표현 3개를 쓰는 조합 baseline은 태스크 사이 일관성이 없다.

| Method | FlyingShape MR Recall ↑ | FlyingShape MRR Recall ↑ | 3RScan MR Recall ↑ | 3RScan MRR Recall ↑ |
| --- | --- | --- | --- | --- |
| Baseline† | 67.32 | 54.30 | 44.02 | 30.77 |
| Ours | **74.39** | **62.00** | **49.07** | **40.74** |

*[Table 1] end-to-end 성능(굵게 최고). MR은 matching+registration, MRR은 세 태스크 전체 평가*

![여러 scan의 end-to-end 누적 reconstruction 정성 결과](/assets/images/living-scenes-fig3.jpg)
*[Figure 3] 세 시점 $$t_1, t_2, t_3$$ 의 scan을 $$t_1$$ 에 누적한 end-to-end reconstruction 비교($$t_3$$ 에서 제거된 의자도 처리)*

![3RScan에서의 multi-object relocalization 정성 비교](/assets/images/living-scenes-fig4.jpg)
*[Figure 4] 3RScan에서 source instance들을 target scan에 매칭·정합한 multi-object relocalization 결과 비교*

### 4.4 Instance Matching

MendNet, VN-ONet, VN-DGCNN 세 baseline과 비교한다(앞 둘은 point cloud reconstruction, 마지막은 분류 네트워크). Table 2와 Table 3이 각각 FlyingShape와 3RScan 결과다. MoRE²가 모든 metric에서 앞서는데, encoder가 큰 pose 변화를 다루는 표현력 있는 전역 invariant feature를 내고 equivariant feature가 입력 point cloud의 고주파 디테일을 modeling하기 때문이다. 같은 카테고리 안의 물체들도 미세한 기하 차이를 포착해 구분한다.

| Method | Instance-level Recall ↑ | R@50% ↑ | R@75% ↑ | R@100% ↑ |
| --- | --- | --- | --- | --- |
| MendNet | 83.69 | <u>96.75</u> | 68.25 | 60.75 |
| VN-DGCNN<sub>cls</sub> | 61.37 | 73.50 | 32.25 | 27.75 |
| VN-ONet<sub>recon</sub> | <u>86.63</u> | 96.00 | <u>74.50</u> | <u>67.75</u> |
| Ours | **88.75** | **97.50** | **78.00** | **72.00** |

*[Table 2] FlyingShape에서의 instance matching 결과(굵게 최고, 밑줄 차선. 원문은 색상 하이라이트)*

| Method | Static | Dynamic | All | R@25% | R@50% | R@75% |
| --- | --- | --- | --- | --- | --- | --- |
| MendNet | <u>60.32</u> | 63.76 | 62.20 | 80.68 | 64.77 | 37.50 |
| VN-DGCNN<sub>cls</sub> | 43.39 | 49.34 | 46.65 | 72.32 | 53.41 | 29.55 |
| VN-ONet<sub>recon</sub> | 56.08 | <u>72.05</u> | <u>64.83</u> | <u>86.36</u> | <u>71.59</u> | <u>44.32</u> |
| Ours | **60.32** | **87.50** | **71.77** | **87.50** | **78.41** | **50.00** |

*[Table 3] 3RScan에서의 instance matching 결과(왼쪽 3열 instance 수준, 오른쪽 3열 scene 수준 recall. 굵게 최고, 밑줄 차선)*

### 4.5 Point Cloud Registration

물체 수준 registration을 겨냥한 학습 기반 방법 RPMNet, equivariant embedding으로 회전을 푸는 FreeReg, 3DMatch·3DLoMatch state-of-the-art인 GeoTransformer와 비교한다(Table 4). FreeReg는 거친 registration만 제공해 큰 partiality 변화에서 잘 동작하지 않는다. RPMNet·GeoTransformer와 달리 MoRE²는 이산적인 점별 대응에 의존하지 않고 연속적인 signed distance field로 기하를 표현해 최적화로 point cloud를 field에 정렬한다. 이 분석은 MoRE²의 ECDF 곡선이 가장 높다는 점으로도 뒷받침된다.

| Dataset | Method | RR ↑ | MedRE ↓ | RMSE ↓ | MedCD ↓ |
| --- | --- | --- | --- | --- | --- |
| FlyingShape | RPMNet | 23.17 | 2.37 | 31.77 | 0.0249 |
| | FreeReg | 47.50 | 2.44 | 33.84 | 0.0760 |
| | GeoTransformer | 77.67 | 1.36 | **16.66** | 0.0271 |
| | Ours w/o optim | <u>83.00</u> | <u>0.86</u> | 20.83 | <u>0.0171</u> |
| | Ours full | **83.83** | **0.74** | <u>18.47</u> | **0.0168** |
| 3RScan | RPMNet | 9.40 | 3.78 | 15.91 | 0.0248 |
| | FreeReg | 26.06 | 5.76 | 11.05 | 0.0082 |
| | GeoTransformer | 51.71 | **3.46** | 6.51 | 0.0141 |
| | Ours w/o optim | <u>58.12</u> | 3.77 | <u>5.49</u> | <u>0.0032</u> |
| | Ours full | **61.11** | <u>3.77</u> | **4.74** | **0.0030** |

*[Table 4] 두 데이터셋에서의 point cloud registration 결과(굵게 최고, 밑줄 차선)*

### 4.6 Instance Reconstruction

MendNet, VN-ONet, ConvONet과 비교한다(Table 5). FlyingShape에서는 2-way Chamfer를, 물체 mesh가 불완전(non-watertight)한 3RScan에서는 1-way Chamfer만 보고한다. joint optimization을 포함한 full 방법이 두 데이터셋 대부분의 metric에서 baseline을 앞서고, 없으면 FlyingShape에서 ConvONet과 비슷하다. 이는 noisy하고 무작위 방향인 point cloud에 대한 최적화 알고리즘의 적응력을 보여준다. 표면 reconstruction만 수행하는 baseline들과 달리, pose graph $$\mathbf{G}$$ 와 공유 embedding $$\mathbf{F}$$ 가 누적 point cloud 사이의 최적화 메시지 전달과 융합을 가능하게 해 registration과 reconstruction 성능을 함께 끌어올린다.

| Method | FlyingShape L1-Chamfer2 ↓ | FlyingShape IoU ↑ | FlyingShape SDF Rec. ↑ | 3RScan L1-Chamfer1 ↓ | 3RScan SDF Rec. ↑ |
| --- | --- | --- | --- | --- | --- |
| MendNet | 25.27 | 47.79 | 6.17 | 17.73 | 20.99 |
| VN-ONet | 8.55 | 34.47 | 65.00 | 10.65 | 51.91 |
| ConvONet | 6.64 | 36.99 | <u>80.67</u> | <u>7.61</u> | **64.89** |
| Ours w/o optim | <u>6.27</u> | <u>49.98</u> | 78.00 | 9.28 | 56.87 |
| Ours full | **6.11** | **66.73** | **83.33** | **6.16** | <u>64.12</u> |

*[Table 5] instance reconstruction 결과(L1-Chamfer는 $$\times 10^{-3}$$. 굵게 최고, 밑줄 차선)*

## 5. Ablation Study

**Predicted instance segmentation.** Mask3D의 noisy하고 불완전한 instance segmentation mask를 MoRE²에 입력한 결과가 Table 6이다. GT mask 대비 모든 태스크·조합에서 MoRE²가 조합 baseline을 앞선다. 특히 두 방법의 matching·registration recall이 비슷한데도 baseline의 MR·MRR recall이 크게 낮다. 또 GT에서 Mask3D로 바뀔 때 baseline의 MR·MRR recall 하락은 약 33%인 반면 MoRE²는 약 18%다. 이는 태스크 간 공유 표현과 joint optimization이 noisy한 조건에서도 효과적임을 보여준다.

| Method | Ins. Seg. | Mat. Rec. ↑ | Reg. Rec. ↑ | MR Rec. ↑ | MRR Rec. ↑ |
| --- | --- | --- | --- | --- | --- |
| Baseline† | GT | 64.83 | 51.71 | 44.02 | 30.77 |
| Ours | GT | **71.77** | **61.11** | **49.07** | **40.74** |
| Baseline† | Mask3D | 43.43 | 47.74 | 27.86 | 20.89 |
| Ours | Mask3D | **45.76** | **51.27** | **40.14** | **33.80** |

*[Table 6] 3RScan에서 Mask3D 예측 mask를 입력으로 한 결과(굵게 최고)*

**Benefit of accumulation.** Table 4의 결과는 두 시점 point cloud 쌍에 대한 것이다. 여기서는 누적에 쓰는 multi-temporal scan 수를 늘려가며 FlyingShape에서 registration(RE)과 reconstruction(CD) 성능을 본다. 누적 없음(point cloud 1개)에서 4개까지 두 metric 모두 크게 개선되고 이후 포화된다. 포화는 누적 point cloud가 완전한 shape을 덮는 비율로 설명된다. 4번째 scan 시점에 완전성이 75%에 가까워, 추가 scan이 registration과 reconstruction에 미치는 영향이 작아진다.

## 6. Conclusion

세 개의 연속된 태스크로 장기 동적 scene(living scene)을 해석하는 새로운 접근 MoRE²를 제안했다. MoRE²는 equivariant embedding의 유연한 적응과 multi-temporal 누적을 가능하게 하는 joint optimization으로 세 태스크를 해결하며, 합성·실세계 데이터셋 모두에서 우수한 성능을 보인다. scene 안 3D asset의 누적적 이해를 가능하게 한다. 향후 연구 방향으로는 탄성 변형과 다수의 동일 물체가 있는 scene의 문제, 대규모 시공간 변화의 이해를 꼽는다.
