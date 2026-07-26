---
title: "[논문리뷰] Depth Anything 3: Recovering the Visual Space from Any Views"
excerpt: "depth와 ray map을 최소 prediction target으로 삼아 plain transformer 하나로 any-view geometry를 통일한 Depth Anything 3 논문 리뷰"
date: 2026-07-26 12:00:00 +0900
categories:
  - 논문리뷰
  - CV
tags:
  - Depth Anything
  - multi-view geometry
  - camera pose estimation
  - monocular depth estimation
  - 3D Gaussian Splatting
  - knowledge distillation
toc: true
toc_sticky: true
---

> **Depth Anything 시리즈가 monocular를 넘어 any-view로 확장된다.**
> Depth Anything 3(DA3)는 이미지가 한 장이든 여러 장이든, 카메라 pose가 있든 없든, **plain transformer 하나와 depth-ray라는 최소 prediction target만으로** 일관된 3D geometry를 복원한다. VGGT를 pose 정확도에서 평균 35.7%, geometry 정확도에서 23.6% 앞선다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647) |
| 저자 | Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang |
| 소속 / 연도 | ByteDance Seed, 2025 (arXiv:2511.10647) |
| 분야 | CV / visual geometry, monocular depth estimation |
| 코드 | [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3) |

## 1. Introduction

시각 입력으로부터 3D 공간 정보를 인지하는 능력은 공간 지능의 초석이고, robotics·mixed reality 같은 응용의 핵심 요건이다. 여기서 Monocular Depth Estimation, SfM(Structure from Motion), MVS(Multi-View Stereo), SLAM 같은 다양한 3D vision 과제가 파생됐다. 이 과제들은 입력 view 수 같은 요소 하나만 다를 뿐 개념적으로 크게 겹치는데도, 지금까지는 과제별로 고도로 특화된 모델을 만드는 것이 지배적 패러다임이었다. DUSt3R·VGGT처럼 여러 과제를 동시에 다루는 통합 모델 시도도 있었지만, 복잡한 맞춤 architecture에 의존하고 과제들의 joint optimization으로 처음부터 학습하느라 대규모 사전학습 모델을 효과적으로 활용하지 못하는 한계가 있다.

이 논문은 기존 3D 과제 정의에서 물러나, 사람의 공간 지능에서 영감을 받은 더 근본적인 목표로 돌아간다. 단일 이미지든, 한 장면의 여러 view든, 비디오 스트림이든 **임의의 시각 입력에서 3D 구조를 복원**하는 것이다. 정교한 architecture 설계를 버리고 두 가지 질문을 따라 최소 모델링(minimal modeling) 전략을 추구한다.

- **Q1**: 수많은 3D 과제의 joint modeling이 필요한가, 아니면 최소한의 prediction target 집합이 존재하는가?
- **Q2**: 이 목표에 plain transformer 하나면 충분한가?

논문은 둘 다 긍정으로 답한다. Depth Anything 3는 특별히 선택된 ray representation을 통해 **any-view depth와 pose 추정만을 위해 학습된 단일 transformer 모델**이고, 이 최소한의 접근이 카메라 pose 유무와 무관하게 임의 장수의 이미지에서 시각 공간(visual space)을 복원하기에 충분함을 보인다.

DA3는 geometry 복원을 dense prediction 과제로 정식화한다. $$N$$ 장의 입력 이미지에 대해 각 입력과 픽셀 정렬된 $$N$$ 개의 depth map과 ray map을 출력하도록 학습한다. backbone은 DINOv2 같은 표준 사전학습 vision transformer를 그대로 쓰고, 임의의 view 수를 다루기 위해 forward 중 선택된 layer에서 token을 동적으로 재배열해 모든 view 사이의 정보 교환을 가능하게 하는 **input-adaptive cross-view self-attention**을 도입한다. 최종 예측에는 같은 feature를 서로 다른 fusion 파라미터로 처리해 depth와 ray를 함께 출력하는 **Dual-DPT head**를 제안한다. 알려진 카메라 pose는 간단한 camera encoder로 선택적으로 주입할 수 있다.

학습은 **teacher-student 패러다임**을 쓴다. depth 카메라 실측, 3D reconstruction, synthetic 데이터 등 다양한 형식의 학습 데이터를 통일하기 위해, synthetic 데이터만으로 강력한 monocular depth teacher를 학습하고 모든 실사 데이터에 dense한 고품질 pseudo depth를 생성한다. 이때 geometry 무결성을 지키기 위해 dense pseudo depth를 원래의 sparse·노이즈 depth에 정렬(align)하는 것이 핵심이다.

평가를 위해 pose 정확도와 geometry 정확도를 직접 재는 **visual geometry benchmark**를 구축했다. object 수준부터 실내·실외까지 89개 이상의 장면, 5개 데이터셋으로 구성되고, DA3는 20개 설정 중 18개에서 state-of-the-art를 달성한다. 표준 monocular 벤치마크에서도 Depth Anything 2(DA2)를 앞선다. 나아가 feed-forward novel view synthesis(FF-NVS) 벤치마크(160개 이상 장면)를 도입하고 DPT head 하나를 더해 fine-tuning했더니, ① geometry foundation model의 fine-tuning이 고도로 특화된 과제 전용 모델을 크게 앞서고 ② geometry 복원 능력이 좋을수록 FF-NVS 성능이 좋아 DA3가 이 과제의 최적 backbone임이 확인됐다.

![다수의 이미지와 선택적 카메라 pose를 받아 일관된 depth·ray map을 예측하고 point cloud와 3D Gaussian으로 융합하는 Depth Anything 3의 결과 모음](/assets/images/depth-anything-3-fig1.jpg)
*[Figure 1] 임의 장수의 이미지와 선택적 camera pose로부터 visual space를 복원하는 Depth Anything 3*

## 2. Related Work

**Multi-view visual geometry estimation.** 전통적 시스템은 재구성을 feature 검출·matching, robust한 relative pose 추정, bundle adjustment를 동반한 SfM, dense MVS로 분해했다. texture가 좋은 장면에서는 여전히 강하지만, 모듈성과 취약한 correspondence 때문에 low-texture·반사·큰 시점 변화에서 robust하지 못하다. 초기 학습 기반 방법은 학습된 detector·descriptor·미분가능 최적화 layer로 구성 요소 수준의 robustness를 넣었고, cost-volume 네트워크가 hand-crafted 정규화를 3D CNN으로 대체했다. 전환점은 **DUSt3R**다. transformer로 두 view 사이의 point map을 직접 예측해 depth와 relative pose를 순수 feed-forward로 계산했고, 이후 multi-view 입력, 비디오 입력, robust correspondence, 카메라 파라미터 주입, 대규모 SfM, SLAM, 3D Gaussian view synthesis로 확장됐다. 그중 **VGGT**는 대규모 학습, multi-stage architecture, 설계상의 중복(redundancy)으로 정확도를 한 단계 끌어올렸다. 이와 달리 이 논문은 단순한 단일 transformer를 중심에 둔 최소 모델링 전략에 집중한다.

**Monocular depth estimation.** 초기 방법은 단일 도메인 데이터셋의 완전 지도학습에 의존해 실내 방 또는 실외 주행 장면에 특화됐고 새로운 환경에 일반화되지 못했다. Depth Anything 시리즈, MoGe, Depth Pro, Metric3D, Marigold 같은 현대의 generalist 접근은 대규모 multi-dataset 학습과 vision transformer·DiT 같은 architecture를 활용하고 affine-invariant depth 정규화 같은 기법을 쓴다. 이 논문의 방법은 통합 visual geometry 추정을 주 목표로 설계됐는데도 monocular depth에서 경쟁력 있는 성능을 보인다.

**Feed-Forward Novel View Synthesis.** NVS는 컴퓨터 비전·그래픽스의 오랜 핵심 문제이고, neural rendering의 부상과 함께 관심이 커졌다. 특히 장면별 최적화 없이 image-to-3D 네트워크 한 번의 forward로 3D representation을 만드는 feed-forward NVS가 유망하다. 초기에는 NeRF를 썼지만 최근에는 명시적 구조와 실시간 rendering이 가능한 3DGS로 옮겨 왔고, epipolar attention, cost volume, depth prior 같은 geometry prior로 개선해 왔다. 최근에는 multi-view geometry foundation model이 통합되고 있지만 기존 방법들은 하나의 선택된 foundation model에만 의존해 평가됐다. 이 논문은 서로 다른 geometry foundation model의 NVS 기여를 체계적으로 벤치마크하고, posed·pose-free 입력, 가변 view 수, 임의 해상도를 다루는 전략을 제안한다.

## 3. Depth Anything 3

단일 이미지, multi-view 모음, 비디오 등 다양한 시각 입력에서 일관된 3D geometry를 복원하고, 가능하면 알려진 카메라 pose도 선택적으로 활용한다.

![DINOv2 backbone에 input-adaptive cross-view attention을 적용하고 Dual-DPT head로 depth와 ray map을 예측하는 전체 pipeline](/assets/images/depth-anything-3-fig2.png)
*[Figure 2] 구조 수정 없는 단일 transformer(vanilla DINOv2)와 Dual-DPT head로 구성된 Depth Anything 3의 pipeline*

### 3.1 Formulation

입력은 $$\mathcal{I} = \{\mathbf{I}_i\}_{i=1}^{N_v}$$, 각 이미지는 $$\mathbf{I}_i \in \mathbb{R}^{H \times W \times 3}$$ 이다. $$N_v = 1$$ 이면 monocular 이미지, $$N_v > 1$$ 이면 비디오나 multi-view 집합이다. 각 이미지는 depth $$\mathbf{D}_i \in \mathbb{R}^{H \times W}$$, extrinsics $$[\mathbf{R}_i \mid \mathbf{t}_i]$$, intrinsics $$\mathbf{K}_i$$ 를 갖는다. 카메라는 translation $$\mathbf{t}_i \in \mathbb{R}^3$$, rotation quaternion $$\mathbf{q}_i \in \mathbb{R}^4$$, FOV 파라미터 $$\mathbf{f}_i \in \mathbb{R}^2$$ 를 이은 벡터 $$\mathbf{v}_i \in \mathbb{R}^9$$ 로도 표현할 수 있다. 픽셀 $$\mathbf{p} = (u, v, 1)^\top$$ 은 3D 점 $$\mathbf{P}$$ 로 투영된다.

$$
\mathbf{P} = \mathbf{R}_i \big( \mathbf{D}_i(u,v) \, \mathbf{K}_i^{-1} \mathbf{p} \big) + \mathbf{t}_i
$$

즉 픽셀을 intrinsics의 역행렬로 카메라 좌표계에 backprojection하고 depth를 곱한 뒤, rotation과 translation으로 world 좌표계로 옮기는 표준 식이다. 이 식을 통해 기저의 3D visual space를 충실히 복원할 수 있다.

**Depth-ray representation.** 유효한 rotation 행렬 $$\mathbf{R}_i$$ 를 직접 예측하는 것은 직교성 제약 때문에 어렵다. 이를 피하려고 카메라 pose를 입력 이미지·depth map과 정렬된 **픽셀별 ray map으로 암묵적으로 표현**한다. 각 픽셀 $$\mathbf{p}$$ 의 카메라 ray $$\mathbf{r} \in \mathbb{R}^6$$ 는 원점 $$\mathbf{t} \in \mathbb{R}^3$$ 와 방향 $$\mathbf{d} \in \mathbb{R}^3$$ 로 정의된다: $$\mathbf{r} = (\mathbf{t}, \mathbf{d})$$. 방향은 픽셀을 카메라 좌표계로 backprojection한 뒤 world 좌표계로 회전해 얻는다.

$$
\mathbf{d} = \mathbf{R} \mathbf{K}^{-1} \mathbf{p}
$$

dense ray map $$\mathbf{M} \in \mathbb{R}^{H \times W \times 6}$$ 은 모든 픽셀의 이 파라미터를 담는다. $$\mathbf{d}$$ 를 정규화하지 않으므로 크기가 투영 scale을 보존하고, 따라서 world 좌표의 3D 점은 단순히

$$
\mathbf{P} = \mathbf{t} + \mathbf{D}(u,v) \cdot \mathbf{d}
$$

가 된다. 예측된 depth와 ray map의 원소별(element-wise) 연산만으로 일관된 point cloud를 만들 수 있다는 뜻이다.

**Ray map에서 카메라 파라미터 유도.** ray map $$\mathbf{M}$$ 의 앞 세 채널은 픽셀별 ray 원점, 뒤 세 채널은 ray 방향이다. 카메라 중심 $$\mathbf{t}_c$$ 는 픽셀별 ray 원점의 평균으로 추정한다.

$$
\mathbf{t}_c = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} \mathbf{M}(h, w, :3)
$$

rotation $$\mathbf{R}$$ 과 intrinsics $$\mathbf{K}$$ 는 homography $$\mathbf{H}$$ 를 찾는 문제로 정식화한다. intrinsics가 단위행렬인 "identity" 카메라를 정의하면 픽셀 $$\mathbf{p}$$ 의 ray 방향은 그대로 $$\mathbf{d}_I = \mathbf{p}$$ 이고, 목표 카메라 좌표계의 ray 방향으로 가는 변환은 $$\mathbf{d}_{\text{cam}} = \mathbf{K} \mathbf{R} \mathbf{d}_I$$ 이다. 즉 두 ray 집합 사이에 $$\mathbf{H} = \mathbf{K}\mathbf{R}$$ 의 직접적인 homography 관계가 성립한다. 변환된 canonical ray와 예측된 목표 ray 사이의 기하 오차를 최소화해 이 homography를 푼다.

$$
\mathbf{H}^{*} = \arg\min_{\|\mathbf{H}\| = 1} \sum_{h=1}^{H} \sum_{w=1}^{W} \left\| \mathbf{H} \mathbf{p}_{h,w} \times \mathbf{M}(h, w, 3\!:\!) \right\|
$$

외적(cross product)이 0에 가까울수록 두 방향이 평행하다는 사실을 이용한 표준 least-squares 문제로, DLT(Direct Linear Transform) 알고리즘으로 효율적으로 풀 수 있다. $$\mathbf{K}$$ 는 상삼각행렬이고 $$\mathbf{R}$$ 은 직교행렬이므로, 최적 $$\mathbf{H}^{*}$$ 를 RQ decomposition으로 유일하게 분해해 $$\mathbf{K}, \mathbf{R}$$ 을 복원한다.

**최소 prediction target.** 최근의 통합 모델들은 point map만 쓰거나(DUSt3R), pose·local/global point map·depth의 중복 조합(VGGT 등)을 multitask로 학습한다. point map만으로는 일관성이 보장되지 않고, 중복 target은 pose 정확도를 높일 수 있어도 얽힘(entanglement)을 일으켜 오히려 정확도를 해친다. 반면 실험(Table 6)은 **depth-ray representation이 장면 구조와 카메라 운동을 모두 담는 최소이자 충분한 target 집합**이며 point map 등 대안을 능가함을 보인다. 다만 추론 시 ray map에서 카메라 pose를 복원하는 계산이 비싸므로, camera token에서 FOV $$\mathbf{f} \in \mathbb{R}^2$$, quaternion $$\mathbf{q} \in \mathbb{R}^4$$, translation $$\mathbf{t} \in \mathbb{R}^3$$ 를 예측하는 가벼운 camera head $$\mathcal{D}_C$$ 를 추가한다. view당 token 하나만 처리하므로 추가 비용은 무시할 수준이다.

### 3.2 Architecture

네트워크는 세 부분으로 구성된다. 단일 transformer backbone, pose conditioning을 위한 선택적 camera encoder, 예측을 만드는 Dual-DPT head다.

**Single transformer backbone.** 대규모 monocular 이미지 corpus로 사전학습된 $$L$$ 개 block의 Vision Transformer(예: DINOv2)를 쓴다. cross-view 추론은 구조 변경 없이 입력 token 재배열로 구현한 **input-adaptive self-attention**으로 가능하게 한다. transformer를 크기 $$L_s$$ 와 $$L_g$$ 의 두 그룹으로 나눠, 앞의 $$L_s$$ 개 layer는 이미지 내부(within-view) self-attention을 적용하고, 뒤의 $$L_g$$ 개 layer는 tensor 재배열을 통해 모든 token을 함께 다루며 cross-view attention과 within-view attention을 번갈아 수행한다. 실전에서는 $$L_s : L_g = 2 : 1$$ 로 두는데, ablation(Table 7)에서 성능·효율의 최적 균형이었다. 이 설계는 입력에 적응적이어서 이미지가 한 장이면 추가 비용 없이 자연스럽게 monocular depth estimation으로 환원된다.

**Camera condition injection.** posed·unposed 입력을 매끄럽게 다루기 위해 각 view 앞에 camera token $$\mathbf{c}_i$$ 를 붙인다. 카메라 파라미터 $$(\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i)$$ 가 있으면 가벼운 MLP $$\mathcal{E}_c$$ 로 $$\mathbf{c}_i = \mathcal{E}_c(\mathbf{f}_i, \mathbf{q}_i, \mathbf{t}_i)$$ 를 만들고, 없으면 공유 learnable token $$\mathbf{c}_l$$ 을 쓴다. patch token과 이어 붙여 모든 attention 연산에 참여시켜, 명시적 geometry 문맥 또는 일관된 학습형 placeholder를 제공한다.

![shared reassembly 모듈을 거친 feature가 depth branch와 ray branch의 서로 다른 fusion layer로 갈라지는 Dual-DPT head 구조](/assets/images/depth-anything-3-fig3.png)
*[Figure 3] reassembly 모듈을 공유해 출력 정렬을 개선하는 Dual-DPT head*

**Dual-DPT head.** 최종 예측 단계에서 dense depth와 ray 값을 함께 출력하는 Dual-DPT head를 제안한다. backbone의 feature 집합을 먼저 **공유 reassembly 모듈**로 처리한 뒤, depth branch와 ray branch **각각의 fusion layer**로 융합하고, 두 개의 출력 layer가 최종 depth·ray map을 만든다. 두 branch가 같은 feature 집합 위에서 동작하고 마지막 fusion 단계만 다르므로, 두 예측 과제 사이의 강한 상호작용을 유도하면서 중복된 중간 representation을 피한다.

### 3.3 Training

**Teacher-student 학습 패러다임.** 학습 데이터는 실사 depth 캡처, 3D reconstruction, synthetic 데이터셋 등 다양한 소스에서 온다. 실사 depth는 노이즈가 많고 불완전해(Figure 4) supervision 가치가 제한적이다. 이를 완화하기 위해 synthetic 데이터만으로 monocular relative depth teacher를 학습해 고품질 pseudo label을 생성하고, 이 dense pseudo depth를 RANSAC least squares로 원래의 sparse·노이즈 GT에 정렬한다. geometry 정확도를 잃지 않으면서 label의 디테일과 완전성을 높이는 것이다. 이 teacher를 **Depth-Anything-3-Teacher**라 부른다.

**학습 목적함수.** 모델 $$\mathcal{F}_\theta$$ 는 입력 $$\mathcal{I}$$ 를 depth map $$\hat{\mathbf{D}}$$, ray map $$\hat{\mathbf{R}}$$, 선택적 camera pose $$\hat{\mathbf{c}}$$ 로 사상한다. loss 계산 전에 모든 GT 신호를 공통 scale 인자로 정규화하는데, 이 scale은 유효한 reprojected point map $$\mathbf{P}$$ 의 평균 $$\ell_2$$ norm으로 정의해 modality 간 크기를 일치시키고 학습을 안정화한다. 전체 목적함수는 여러 항의 가중합이다.

$$
\mathcal{L} = \mathcal{L}_D(\hat{\mathbf{D}}, \mathbf{D}) + \mathcal{L}_M(\hat{\mathbf{R}}, \mathbf{M}) + \mathcal{L}_P(\hat{\mathbf{D}} \odot \mathbf{d} + \mathbf{t},\ \mathbf{P}) + \beta \mathcal{L}_C(\hat{\mathbf{c}}, \mathbf{v}) + \alpha \mathcal{L}_{\text{grad}}(\hat{\mathbf{D}}, \mathbf{D})
$$

$$\mathcal{L}_D$$ 는 depth loss, $$\mathcal{L}_M$$ 은 ray map loss, $$\mathcal{L}_P$$ 는 예측 depth와 ray를 결합해 만든 point와 GT point map $$\mathbf{P}$$ 사이의 loss, $$\mathcal{L}_C$$ 는 선택적 camera head의 pose loss($$\mathbf{v}$$ 는 9차원 GT pose 벡터), $$\mathcal{L}_{\text{grad}}$$ 는 depth gradient loss다. 모든 loss 항은 $$\ell_1$$ norm 기반이고 가중치는 $$\alpha = 1$$, $$\beta = 1$$ 이다. depth loss는 confidence를 함께 학습하는 형태다.

$$
\mathcal{L}_D(\hat{\mathbf{D}}, \mathbf{D} ; D_c) = \frac{1}{Z_\Omega} \sum_{p \in \Omega} m_p \left( D_{c,p} \, \lvert \hat{\mathbf{D}}_p - \mathbf{D}_p \rvert - \lambda_c \log D_{c,p} \right)
$$

여기서 $$\Omega$$ 는 픽셀 영역, $$m_p$$ 는 유효 마스크, $$D_{c,p}$$ 는 depth $$D_p$$ 의 confidence, $$Z_\Omega$$ 는 정규화 상수다. 오차가 큰 픽셀은 confidence를 낮춰 벌점을 줄이되, $$-\lambda_c \log D_{c,p}$$ 항이 confidence가 전부 0으로 가는 것을 막는다. gradient loss는 depth의 수평·수직 유한차분 $$\nabla_x, \nabla_y$$ 의 차이를 벌점한다.

$$
\mathcal{L}_{\text{grad}}(\hat{\mathbf{D}}, \mathbf{D}) = \| \nabla_x \hat{\mathbf{D}} - \nabla_x \mathbf{D} \|_1 + \| \nabla_y \hat{\mathbf{D}} - \nabla_y \mathbf{D} \|_1
$$

이 loss는 날카로운 edge를 보존하면서 평면 영역의 smoothness를 보장한다.

### 3.4 Implementation Details

**학습 데이터셋.** 학습·평가 데이터셋 구성은 Table 1과 같다. 학습과 test가 겹칠 수 있는 ScanNet++는 장면 수준에서 엄격히 분리했다.

| 용도 | Dataset | #Scenes | Data Type |
| --- | --- | --- | --- |
| Pose-geometry benchmark | HiRoom (ours) | 29 | Synthetic |
| | ETH3D | 11 | LiDAR |
| | DTU | 22 | LiDAR |
| | 7Scenes | 7 | LiDAR |
| | ScanNet++ | 20 | LiDAR |
| Pose-geometry training | AriaDigitalTwin | 237 | Synthetic |
| | AriaSyntheticENV | 99950 | Synthetic |
| | ArkitScenes | 4388 | LiDAR |
| | BlendedMVS | 503 | 3D Recon |
| | Co3dv2 | 30616 | Colmap |
| | DL3DV | 6379 | Colmap |
| | HyperSim | 344 | Synthetic |
| | MapFree | 921 | Colmap |
| | MegaDepth | 268 | Colmap |
| | MegaSynth | 6049 | Synthetic |
| | MvsSynth | 121 | Synthetic |
| | Objaverse | 505557 | Synthetic |
| | Omniobject | 5885 | Synthetic |
| | OmniWorld | 1039 | Synthetic |
| | PointOdyssey | 44 | Synthetic |
| | ReplicaVMAP | 17 | Synthetic |
| | ScanNet++ | 230 | LiDAR |
| | ScenenetRGBD | 16866 | Synthetic |
| | TartanAir | 355 | Synthetic |
| | Trellis | 557408 | Synthetic |
| | vKitti2 | 50 | Synthetic |
| | WildRGBD | 23050 | LiDAR |

*[Table 1] Depth Anything 3에 사용된 데이터셋(장면 수, 데이터 타입)*

**학습 세부.** 128장의 H100 GPU로 200K step 학습하고, 8K step warm-up과 peak learning rate $$2 \times 10^{-4}$$ 를 쓴다. 기본 해상도는 $$504 \times 504$$ 인데, 2·3·4·6·9·14로 나누어떨어져 2:3, 3:4, 9:16 같은 흔한 사진 비율과 호환성이 좋다. 학습 해상도는 8가지 조합에서 무작위로 샘플링하고, $$504 \times 504$$ 에서는 view 수를 [2, 18]에서 균일 샘플링한다. step당 token 수가 일정하도록 배치 크기를 동적으로 조정한다. supervision은 120K step에서 GT depth에서 teacher label로 전환되고, pose conditioning은 확률 0.2로 무작위 활성화된다.

## 4. Teacher-Student Learning

![sparse하거나 구멍이 많은 실사 데이터셋 depth label의 예시 모음](/assets/images/depth-anything-3-fig4.jpg)
*[Figure 4] 품질이 낮은 실사 데이터셋 depth의 예*

Figure 4처럼 실사 데이터셋의 depth 품질이 낮으므로, teacher 모델을 synthetic 데이터만으로 학습해 실사 데이터의 supervision을 제공한다. teacher는 monocular relative depth 예측기로 학습되고, 추론·supervision 시 노이즈 GT depth로 scale·shift 파라미터를 구해 예측 relative depth를 절대 depth 측정치에 정렬할 수 있다.

### 4.1 Constructing the Teacher Model

Depth Anything 2를 기반으로 데이터와 representation 양면에서 확장한다. teacher의 backbone은 DA3 framework와 그대로 정렬되어 있다. DINOv2 vision transformer에 DPT decoder를 얹은 것뿐이고 특수한 구조 수정은 없다.

**Data scaling.** 정밀한 geometry 디테일을 위해 teacher는 synthetic 데이터만으로 학습한다. DA2의 synthetic 데이터셋이 비교적 제한적이었던 것과 달리, DA3는 Hypersim, TartanAir, IRS, vKITTI2, BlendedMVS, SPRING, MVSSynth, UnrealStereo4K, GTA-SfM, TauAgent, KenBurns, MatrixCity, EDEN, ReplicaGSO, UrbanSyn, PointOdyssey, Structured3D, Objaverse, Trellis, OmniObject로 corpus를 대폭 확장했다. 실내·실외·object 중심·in-the-wild 장면을 아우르며 teacher의 일반화를 높인다.

**Depth representation.** scale-shift-invariant disparity를 예측하는 DA2와 달리 teacher는 **scale-shift-invariant depth**를 출력한다. metric depth 추정이나 multi-view geometry처럼 disparity가 아니라 depth 공간에서 직접 동작하는 downstream 과제에 depth가 더 적합하기 때문이다. depth가 disparity보다 근거리 민감도가 낮은 문제는 선형 depth 대신 **exponential depth**를 예측해 가까운 거리의 분별력을 높이는 것으로 해결한다.

**학습 목적함수.** 표준 depth-gradient loss에 더해 MoGe가 제안한 ROE 정렬 기반 global-local loss를 쓴다. 국소 geometry를 더 다듬기 위해 **거리 가중 surface normal loss**를 도입한다. 각 중심 픽셀에 대해 이웃 점 4개를 샘플링해 비정규화 normal $$n_i$$ 를 계산하고, 다음 가중치로 가중한다.

$$
w_i = \sum_{j=0}^{4} \| n_j \| - \| n_i \|
$$

$$n_i$$ 의 크기는 이웃이 중심에서 멀수록 커지므로, 이 가중치는 중심에서 먼 이웃의 기여를 낮춰 평균 normal이 참 국소 surface normal에 가까워지게 한다.

$$
n_m = \sum_{i=0}^{4} w_i \frac{n_i}{\| n_i \|}
$$

최종 normal loss는 평균 normal과 개별 normal의 각도 오차 $$\mathcal{E}$$ 의 합이다.

$$
\mathcal{L}_N = \mathcal{E}(\hat{n}_m, n_m) + \sum_{i=0}^{4} \mathcal{E}(\hat{n}_i, n_i)
$$

하늘 영역과 object 전용 데이터셋의 배경은 GT가 정의되지 않으므로, depth 출력과 정렬된 sky mask와 object mask를 MSE loss로 함께 예측한다. teacher의 전체 목적함수는

$$
\mathcal{L}_T = \alpha \mathcal{L}_{\text{grad}} + \mathcal{L}_{\text{gl}} + \mathcal{L}_N + \mathcal{L}_{\text{sky}} + \mathcal{L}_{\text{obj}}
$$

이고 $$\alpha = 0.5$$ 다. $$\mathcal{L}_{\text{grad}}, \mathcal{L}_{\text{gl}}, \mathcal{L}_{\text{sky}}, \mathcal{L}_{\text{obj}}$$ 는 각각 gradient loss, global-local loss, sky-mask loss, object-mask loss다.

### 4.2 Teaching Depth Anything 3

실사 데이터셋은 카메라 pose 추정의 일반화에 결정적이지만 깨끗한 depth를 거의 제공하지 않는다. teacher의 고품질 relative depth $$\tilde{\mathbf{D}}$$ 를 COLMAP이나 능동 센서의 노이즈 metric 측정치 $$\mathbf{D}$$ 에 robust한 RANSAC scale-shift 절차로 정렬한다. 유효 마스크 $$m_p$$ 아래에서 scale $$s$$ 와 shift $$t$$ 를 추정한다.

$$
(\hat{s}, \hat{t}) = \arg\min_{s > 0,\, t} \sum_{p \in \Omega} m_p \big( s\,\tilde{\mathbf{D}}_p + t - \mathbf{D}_p \big)^2, \quad \mathbf{D}^{T \rightarrow M} = \hat{s}\,\tilde{\mathbf{D}} + \hat{t}
$$

inlier 임계값은 잔차 중앙값으로부터의 평균 절대 편차로 둔다. 정렬된 $$\mathbf{D}^{T \rightarrow M}$$ 는 scale이 일관되고 pose-depth가 정합하는 supervision을 제공해, joint depth-ray 목적함수를 보완하고 실사 일반화를 개선한다(Figure 8).

### 4.3 Teaching Monocular Model

teacher-student 패러다임으로 monocular depth 모델도 추가 학습한다. DA2 framework를 따라 unlabeled 이미지에 teacher가 생성한 pseudo label로 student를 학습하되, 예측 target이 disparity(DA2)가 아니라 **depth**라는 점이 다르다. teacher와 같은 loss를 pseudo depth label에 적용해 추가 supervision한다. 이 monocular 모델도 relative depth를 예측하며, unlabeled 데이터와 teacher supervision만으로 표준 monocular depth 벤치마크에서 state-of-the-art를 달성한다(Table 10).

### 4.4 Teaching Metric Model

teacher 모델로 경계가 선명한 metric depth 모델도 학습할 수 있다. Metric3Dv2를 따라 canonical camera space 변환으로 focal length 변화에 따른 depth 모호성을 해소한다. GT depth를 비율 $$f^c / f$$ 로 rescale하는데, $$f^c$$ 는 canonical focal length(300으로 설정), $$f$$ 는 카메라 focal length다. 디테일 선명도를 위해 teacher 예측의 scale·shift를 GT metric depth에 정렬해 학습 label로 쓴다.

**학습 데이터셋.** Taskonomy, DIML(Outdoor), DDAD, Argoverse, Lyft, PandaSet, Waymo, ScanNet++, ARKitScenes, Map-free, DSEC, Driving Stereo, Cityscapes 등 14개 데이터셋으로 학습한다. stereo 데이터셋에는 FoundationStereo의 예측을 학습 label로 활용한다.

**구현 세부.** monocular teacher 학습을 대체로 따른다. 기본 해상도 504에 다양한 비율로 학습하고, AdamW로 encoder lr 5e-6, decoder lr 5e-5를 쓴다. 5% 확률로 90도 또는 270도 회전 augmentation을 적용한다. 20% 확률로는 원래 GT label을 쓴다. 배치 64로 160K iteration 학습하고, 목적함수는 depth loss, gradient loss, sky-mask loss의 가중합이다.

## 5. Application: Feed-Forward 3D Gaussian Splattings

### 5.1 Pose-Conditioned Feed-Forward 3DGS

일관된 depth 추정이 downstream 3D vision 과제를 크게 향상시킬 수 있다는 믿음 아래, 시연 과제로 feed-forward novel view synthesis(FF-NVS)를 고른다(3D representation은 3DGS). 최소 모델링 전략을 유지해, **GS-DPT head 하나를 추가해 픽셀 정렬된 3D Gaussian 파라미터를 추론**하도록 fine-tuning한다.

**GS-DPT head.** backbone에서 뽑은 view별 visual token으로 camera 공간의 3D Gaussian 파라미터 $$\{\sigma_i, \mathbf{q}_i, \mathbf{s}_i, \mathbf{c}_i\}_{i=1}^{H \times W}$$ 를 예측한다. $$\sigma_i$$ 는 opacity, $$\mathbf{q}_i$$ 는 rotation quaternion, $$\mathbf{s}_i \in \mathbb{R}^3$$ 는 scale, $$\mathbf{c}_i \in \mathbb{R}^3$$ 는 RGB 색이다. opacity는 confidence head가, 나머지는 GS-DPT 본체가 예측한다. 추정된 depth를 world 좌표로 unprojection해 3D Gaussian의 전역 위치 $$\mathbf{P}_i \in \mathbb{R}^3$$ 를 얻고, 이 primitive들을 rasterization해 주어진 카메라 pose의 novel view를 합성한다.

**학습 목적함수.** rendering된 novel view에 대한 photometric loss(MSE와 LPIPS), 그리고 관측 view의 추정 depth에 대한 scale-shift-invariant depth loss로 fine-tuning한다.

### 5.2 Pose-Adaptive Feed-Forward 3DGS

위의 pose-conditioned 버전이 DA3를 feed-forward 3DGS backbone으로 벤치마크하기 위한 것이라면, in-the-wild 평가에 더 적합한 대안도 제시한다. DA3와 동일한 사전학습 가중치로 매끄럽게 통합되어, **pose 유무와 무관하게, 다양한 해상도·view 수에서** novel view synthesis가 가능하다.

**Pose-adaptive 정식화.** 모든 입력이 uncalibrated라고 가정하는 기존 방법들과 달리, posed·unposed 입력을 모두 받는 pose-adaptive 설계를 채택한다. 두 가지 설계가 필요하다. ① 모든 3DGS 파라미터를 local camera 공간에서 예측한다. ② backbone이 posed·unposed 이미지를 매끄럽게 다뤄야 한다. DA3 backbone은 둘 다 만족한다. pose가 있으면 예측 depth와 camera 공간 3DGS를 Umeyama 정렬로 scale해 world 공간에 unprojection하고, 없으면 예측 pose를 그대로 쓴다. 정확한 surface geometry와 rendering 품질 사이의 trade-off를 줄이기 위해 GS-DPT head에서 추가 depth offset을 예측하고, in-the-wild robustness를 위해 Gaussian별 색을 spherical harmonics 계수로 바꿔 view-dependent surface를 모델링한다.

**강화된 학습 전략.** 불안정한 학습을 피하기 위해 DA3 backbone을 사전학습 가중치로 초기화하고 freeze한 채 GS-DPT head만 학습한다. 고해상도 입력에는 적은 context view를, 저해상도 입력에는 많은 view를 짝지어 학습해 안정성과 다양한 평가 시나리오를 함께 지원한다.

### 5.3 Implementation Details

**학습 데이터셋.** COLMAP pose가 있는 대규모 실사 데이터셋 DL3DV의 10,015개 장면으로 feed-forward 3DGS 모델을 학습한다. 벤치마크에 쓰는 140개 DL3DV 장면은 학습 셋과 완전히 분리해 데이터 누수를 막는다.

## 6. Visual Geometry Benchmark

geometry 예측 모델을 평가하기 위해 pose 정확도, 재구성 정확도를 통한 depth, 시각 rendering 품질을 직접 평가하는 벤치마크를 도입한다.

### 6.1 Benchmark Pipeline

**Pose estimation.** 장면마다 가용한 모든 이미지를 쓰되, 한도를 넘으면 고정 random seed로 100장을 샘플링한다. 선택된 이미지를 feed-forward 모델에 통과시켜 일관된 pose·depth를 얻고 pose 정확도를 계산한다.

**Geometry estimation.** 같은 이미지 집합에 대해 예측 pose와 예측 depth로 재구성을 수행한다. 예측 pose를 GT pose에 evo(Umeyama 정렬)로 맞춰 재구성을 GT 좌표계로 옮기는 변환을 얻는데, robustness를 위해 RANSAC 기반 정렬을 쓴다. 무작위 pose 부분집합에 evo를 반복 적용하고, translation 오차가 전체 pose 편차의 중앙값보다 작은 pose를 inlier로 세어 inlier가 가장 많은 변환을 고른다. 정렬된 예측 point cloud를 예측 depth map과 TSDF fusion으로 융합한 뒤, GT point cloud와 비교해 재구성 품질을 평가한다.

**Visual rendering.** 장면당 이미지는 대개 300~400장이고, 8장마다 1장을 평가용 target novel view로 샘플링한다. 나머지 시점에서 카메라 translation·rotation 거리를 함께 고려한 farthest point sampling으로 입력 context view 12장을 고른다.

### 6.2 Metrics

**Pose 지표.** VGGT·PoseDiffusion의 프로토콜을 따라 AUC를 보고한다. 두 이미지 간 rotation·translation의 각도 편차를 재는 RRA(Relative Rotation Accuracy)와 RTA(Relative Translation Accuracy)를 임계값 집합과 비교해 정확도 값을 얻고, 각 임계값에서 RRA와 RTA 중 작은 쪽으로 결정되는 정확도-임계값 곡선의 적분이 AUC다. 허용 오차 수준을 보이기 위해 임계값 3과 30의 결과를 주로 보고한다(Auc3, Auc30).

**재구성 지표.** GT 점집합을 $$\mathcal{G}$$, 평가 대상 재구성 점집합을 $$\mathcal{R}$$ 이라 할 때, 정확도는 $$\text{dist}(\mathcal{R} \rightarrow \mathcal{G})$$, 완전성은 $$\text{dist}(\mathcal{G} \rightarrow \mathcal{R})$$ 로 재고, 둘의 평균이 Chamfer Distance(CD)다. 거리 임계값 $$d$$ 에 대해 precision은 $$\mathcal{R}$$ 의 점 중 GT까지 거리가 $$d$$ 미만인 비율, recall은 $$\mathcal{G}$$ 의 점 중 재구성까지 거리가 $$d$$ 미만인 비율이고, 둘의 조화평균인 F1-score를 보고한다.

$$
\text{F1} = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
$$

### 6.3 Datasets

벤치마크는 다섯 데이터셋으로 구성된다. **HiRoom**은 전문 아티스트가 만든 실내 거실 장면 30개의 Blender 렌더링 synthetic 데이터셋이다(F1 임계값 0.05m, TSDF voxel 0.007m). **ETH3D**는 레이저 센서 GT depth가 있는 고해상도 실내·실외 이미지로, 11개 장면을 선택했다(임계값 0.25, voxel 0.039m). **DTU**는 124개 물체를 49개 view로 기록한 실내 데이터셋으로, 22개 평가 scan을 쓰고 배경 픽셀은 RMBG 2.0으로 제거한다. **7Scenes**는 심한 motion blur의 저해상도 실내 이미지로 구성된 어려운 실사 데이터셋이다(11배 frame downsampling, 임계값 0.05m). **ScanNet++**는 고해상도 이미지와 레이저 스캔 재구성 depth를 제공하는 대규모 실내 데이터셋으로, 20개 장면을 쓴다(5배 downsampling, 임계값 0.05m, voxel 0.02m).

**Visual rendering 품질.** DL3DV 140개, Tanks and Temples 6개, MegaDepth 19개 장면으로 새 NVS 벤치마크를 구축한다. COLMAP으로 추정한 GT 카메라 pose를 직접 써서 공정한 비교를 보장하고, rendering된 novel view의 PSNR, SSIM, LPIPS를 보고한다.

## 7. Experiments

### 7.1 Comparison with State of the Art

**Baselines.** VGGT는 한 장 또는 여러 장의 view에서 카메라 파라미터·depth·3D point를 함께 예측하는 end-to-end transformer다. Pi3는 permutation-equivariant 설계로 순서 없는 이미지에서 affine-invariant 카메라와 scale-invariant point map을 복원한다. MapAnything은 카메라 pose를 입력으로도 받을 수 있는 feed-forward framework, Fast3R은 point map 회귀를 수백~수천 장으로 확장한 모델, DUSt3R은 uncalibrated 이미지 쌍의 point map을 회귀해 전역 정렬하는 모델이다.

**Pose estimation.** Table 2처럼 DA3-Giant는 DTU의 Auc30 하나를 빼고 거의 모든 지표에서 최고 성능이다. Auc3에서는 모든 경쟁 방법 대비 최소 8%의 상대 개선을 보이고, ScanNet++에서는 2위 모델보다 33% 상대 개선을 달성한다.

| Methods | Params | HiRoom Auc3 | HiRoom Auc30 | ETH3D Auc3 | ETH3D Auc30 | DTU Auc3 | DTU Auc30 | 7Scenes Auc3 | 7Scenes Auc30 | ScanNet++ Auc3 | ScanNet++ Auc30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DUSt3R | 0.57B | 17.6 | 54.3 | 4.30 | 27.3 | 4.00 | 74.3 | 6.90 | 61.6 | 8.10 | 33.9 |
| Fast3R | 0.65B | 25.9 | 77.0 | 8.10 | 44.4 | 9.50 | 79.1 | 19.0 | 78.6 | 17.9 | 72.5 |
| MapAnything | 0.56B | 17.9 | 82.8 | 19.2 | 77.4 | 6.50 | 72.7 | 12.6 | 79.7 | 20.2 | 84.1 |
| Pi3 | 0.96B | 67.0 | 94.8 | 35.2 | 87.3 | 62.5 | 94.9 | 25.5 | 86.3 | 50.7 | 92.1 |
| VGGT | 1.19B | 49.1 | 88.0 | 26.3 | 80.8 | 79.2 | **99.8** | 23.9 | 85.0 | 62.6 | 95.1 |
| DA3-Giant | 1.10B | **80.3** | **95.9** | **48.4** | **91.2** | **94.1** | 99.4 | 28.5 | **86.8** | **85.0** | **98.1** |
| DA3-Large | 0.36B | 58.7 | 94.2 | 32.2 | 86.9 | 70.2 | 96.7 | **29.2** | 86.6 | 60.2 | 94.7 |
| DA3-Base | 0.11B | 19.0 | 83.2 | 15.1 | 74.6 | 60.1 | 95.9 | 20.1 | 82.9 | 25.1 | 83.4 |
| DA3-Small | 0.03B | 9.49 | 75.2 | 8.59 | 62.1 | 30.6 | 91.2 | 14.0 | 78.7 | 10.9 | 71.9 |

*[Table 2] SOTA 방법들과의 pose 정확도 비교(Auc3↑, Auc30↑, 굵게 최고)*

**Geometry estimation.** Table 3처럼 DA3-Giant는 다섯 pose-free 설정 전부에서 모든 경쟁자를 앞서며 새 SOTA를 세운다. 평균적으로 VGGT 대비 25.1%, Pi3 대비 21.5%의 상대 개선이다. 더 주목할 점은 3배 작은 DA3-Large(0.30B)가 열 설정 중 다섯에서 기존 SOTA VGGT(1.19B)를 넘어선다는 것이다. pose가 주어지면 DA3와 MapAnything은 이를 직접 활용하고 다른 방법들도 GT pose fusion의 이득을 본다. 제한된 비디오 설정으로 이미 성능이 포화된 7Scenes를 제외하면 대부분의 데이터셋에서 뚜렷한 향상이 있다. pose conditioning 시 모델 크기 스케일링의 이득이 pose-free보다 작은데, pose 추정이 depth 추정보다 스케일링에 강하게 반응해 더 큰 모델을 요구함을 시사한다.

| Methods | Params | HiRoom w/o p. | HiRoom w/ p. | ETH3D w/o p. | ETH3D w/ p. | DTU w/o p. | DTU w/ p. | 7Scenes w/o p. | 7Scenes w/ p. | ScanNet++ w/o p. | ScanNet++ w/ p. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DUSt3R | 0.57B | 30.1 | 39.5 | 19.7 | 18.8 | 7.60 | 7.97 | 26.6 | 39.8 | 18.9 | 27.3 |
| Fast3R | 0.65B | 40.7 | 48.2 | 38.5 | 50.3 | 6.88 | 8.20 | 41.0 | 49.8 | 37.1 | 53.7 |
| MapAnything | 0.56B | 32.4 | 69.2 | 54.8 | 71.9 | 7.91 | 3.97 | 44.8 | 55.2 | 39.4 | 71.3 |
| Pi3 | 0.96B | 75.8 | 85.0 | 72.7 | 80.6 | 3.28 | 1.72 | 44.2 | **57.5** | 63.1 | 73.3 |
| VGGT | 1.19B | 56.7 | 70.2 | 57.2 | 66.7 | 2.05 | 1.44 | 47.9 | 51.4 | 66.4 | 70.7 |
| DA3-Giant | 1.10B | **85.1** | **95.6** | **79.0** | **87.1** | **1.85** | 1.85 | 53.5 | 56.5 | **77.0** | **79.3** |
| DA3-Large | 0.36B | 69.5 | 87.1 | 65.8 | 75.2 | 2.08 | **1.23** | **56.3** | 49.2 | 67.9 | 75.7 |
| DA3-Base | 0.11B | 25.9 | 71.4 | 49.5 | 66.7 | 2.87 | 2.36 | 49.9 | 50.6 | 47.2 | 67.8 |
| DA3-Small | 0.03B | 18.3 | 52.2 | 41.6 | 63.4 | 5.83 | 2.49 | 41.0 | 46.8 | 32.3 | 53.8 |

*[Table 3] SOTA 방법들과의 재구성 정확도 비교(DTU는 CD↓ mm, 나머지는 F1↑, w/o p.와 w/ p.는 GT pose 제공 여부, 굵게 최고)*

![여러 장면에서 DA3와 경쟁 모델들의 point cloud를 비교한 정성 결과](/assets/images/depth-anything-3-fig6.jpg)
*[Figure 6] 다른 방법들보다 기하적으로 정돈되고 노이즈가 훨씬 적은 DA3의 point cloud 품질 비교*

**Monocular depth.** monocular depth 정확도도 geometry 품질을 반영한다. Table 4처럼 DA2 벤치마크의 표준 monocular depth 데이터셋에서 DA3는 VGGT와 DA2를 능가한다. 참고로 teacher 모델의 결과도 함께 싣는다.

| Method | KITTI | NYU | SINTEL | ETH3D | DIODE | Rank |
| --- | --- | --- | --- | --- | --- | --- |
| DA2 | 94.6 | **97.9** | 77.2 | 86.5 | 95.2 | 2.60 |
| VGGT | 91.7 | **97.9** | 67.9 | 97.5 | 95.3 | 3.75 |
| DA3 | 95.3 | 97.4 | 75.5 | 98.6 | 95.4 | 2.20 |
| Teacher | **97.2** | **97.9** | **81.4** | **99.8** | **96.6** | **1.00** |

*[Table 4] Monocular depth 비교($$\delta_1$$↑, 굵게 최고)*

**Visual rendering.** FF-NVS 평가를 위해 pixelSplat, MVSplat, DepthSplat의 세 3DGS 모델과, geometry backbone을 Fast3R·MV-DUSt3R·VGGT로 바꾼 framework들을 비교한다. 모든 모델을 DL3DV-10K 학습 셋에서 통일된 프로토콜로 학습했다. Table 5에서 모든 모델이 다른 데이터셋보다 DL3DV에서 훨씬 좋은데, 3DGS 기반 NVS가 장면 내용보다 DL3DV로 표준화된 trajectory·pose 분포에 민감함을 시사한다. 두 그룹을 비교하면 geometry 모델 기반 framework가 특화된 feed-forward 모델을 일관되게 앞서, 단순한 backbone과 DPT head가 epipolar transformer·cost volume·cascade 모듈 같은 복잡한 과제 전용 설계를 능가함을 보여준다. 그룹 안에서는 NVS 성능이 geometry 추정 능력과 상관되어 DA3가 가장 강한 backbone이 된다.

| Methods | DL3DV PSNR↑ | DL3DV SSIM↑ | DL3DV LPIPS↓ | T&T PSNR↑ | T&T SSIM↑ | T&T LPIPS↓ | MegaDepth PSNR↑ | MegaDepth SSIM↑ | MegaDepth LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelSplat | 16.55 | 0.456 | 0.480 | 13.81 | 0.347 | 0.558 | 13.87 | 0.367 | 0.561 |
| MVSplat | 18.13 | 0.559 | 0.393 | 14.81 | 0.391 | 0.508 | 14.67 | 0.398 | 0.533 |
| DepthSplat | 19.24 | 0.620 | 0.322 | 15.80 | 0.474 | 0.418 | 15.90 | 0.471 | 0.450 |
| Fast3R | 19.30 | 0.604 | 0.320 | 16.24 | 0.478 | 0.409 | 16.43 | 0.493 | 0.421 |
| MV-DUSt3R | 20.01 | 0.645 | 0.294 | 17.04 | 0.529 | 0.370 | 16.20 | 0.484 | 0.437 |
| VGGT | 20.96 | 0.697 | 0.253 | 17.18 | 0.550 | 0.347 | 16.45 | 0.500 | 0.417 |
| DAv3 (Ours) | **21.33** | **0.711** | **0.241** | **18.10** | **0.578** | **0.311** | **17.89** | **0.561** | **0.351** |

*[Table 5] NVS 과제에서의 SOTA 비교(context view 12장, 해상도 270×480, 굵게 최고)*

### 7.2 Analysis for Depth Anything 3

DA3-Giant 학습에는 128장의 H100으로 약 10일이 필요하다. 계산 비용을 줄이기 위해 이 절의 모든 ablation은 ViT-L backbone과 최대 10 view로 수행한다(32장의 H100으로 약 4일).

#### 7.2.1 Sufficiency of the Depth-Ray Representation

depth-ray representation을 검증하기 위해 prediction 조합들을 비교한다(Table 6). 모든 모델은 ViT-L backbone과 동일한 학습 설정을 쓴다. 네 가지 head를 평가한다. ① dense depth map을 위한 depth, ② 직접 3D point cloud를 위한 pcd, ③ 9-DoF 카메라 pose를 위한 cam, ④ 제안하는 픽셀별 ray map을 위한 ray다. **최소 조합인 depth + ray가 depth + pcd + cam과 depth + cam을 모든 데이터셋·지표에서 일관되게 앞서고**, Auc3에서는 depth + cam 대비 거의 100%의 상대 이득을 얻는다. 보조 cam head를 더해도(depth + ray + cam) 추가 이득이 없어 depth-ray representation의 충분성이 확인된다. camera head의 계산 비용이 backbone의 약 0.1%로 무시할 수준이므로, 최종 representation으로는 depth + ray + cam을 채택한다.

| Methods | HiRoom Auc3 | HiRoom F1 | ETH3D Auc3 | ETH3D F1 | DTU Auc3 | DTU CD | 7Scenes Auc3 | 7Scenes F1 | ScanNet++ Auc3 | ScanNet++ F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth + pcd + cam | 9.1 | 12.8 | 19.0 | 60.4 | 42.3 | 4.918 | 20.8 | 43.4 | 22.0 | 43.0 |
| depth + cam | 10.8 | 16.5 | 9.9 | 48.0 | 23.3 | 5.316 | 13.0 | 38.5 | 13.3 | 41.0 |
| depth + ray | **48.7** | **60.3** | **25.5** | **65.4** | 46.5 | 3.919 | 24.0 | **46.5** | **35.5** | 53.4 |
| depth + ray + cam | 37.2 | 45.4 | 22.3 | 59.4 | **56.3** | **3.066** | **25.7** | 45.6 | 34.1 | **56.5** |

*[Table 6] Prediction target 조합의 ablation(camera condition token 없이 실험, 굵게 최고)*

#### 7.2.2 Sufficiency of a Single Plain Transformer

표준 ViT-L backbone을, 두 개의 서로 다른 transformer를 쌓아 block 수가 3배인 VGGT 스타일 architecture와 비교한다. 공정한 용량 비교를 위해 VGGT 스타일 모델은 더 작은 ViT-B backbone을 써서 ViT-L과 비슷한 파라미터 크기로 맞췄다. Table 7처럼 VGGT 스타일 모델은 baseline의 79.8% 성능으로 떨어져, 비슷한 규모에서 단일 transformer 설계의 우월함이 확인된다. 이 격차는 backbone 전체가 사전학습된 것과 달리 VGGT는 block의 2/3가 미학습이기 때문으로 본다. 또한 모든 layer에서 cross-view/within-view attention을 번갈아 하는 Full Alt. 변형은 거의 모든 지표에서 성능이 떨어져, 부분 교대(partial alternation)가 더 효과적이고 robust한 전략임을 보인다.

| Methods | HiRoom Auc3 | HiRoom F1 | ETH3D Auc3 | ETH3D F1 | DTU Auc3 | DTU CD | 7Scenes Auc3 | 7Scenes F1 | ScanNet++ Auc3 | ScanNet++ F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a. Proposed Arch. | **39.2** | **47.0** | **21.0** | 55.4 | 45.8 | 3.82 | **26.2** | 47.6 | **30.3** | **51.1** |
| b. VGGT Style | 3.72 | 14.5 | 2.31 | 27.4 | 1.38 | 6.93 | 0.97 | 21.4 | 2.03 | 12.2 |
| c. Full Alt. | 24.7 | 29.3 | 13.1 | 51.9 | 44.6 | 4.23 | 21.1 | 48.6 | 27.7 | 47.5 |
| d. w/o Dual DPT | 5.59 | 11.5 | 13.6 | 33.4 | 21.7 | 5.14 | 14.2 | **49.4** | 26.5 | 46.6 |
| e. w/o Teacher | 11.2 | 16.0 | 16.2 | **57.6** | **52.5** | **3.29** | 23.3 | 40.3 | 26.2 | 47.7 |
| f. w/o Pose Cond.\* | | 65.8 | | 63.2 | | 3.65 | | **58.4** | | 62.8 |
| g. w/ Pose Cond.\* | | **73.8** | | **70.9** | | **2.14** | | 46.0 | | **65.7** |

*[Table 7] Ablation study. 비슷한 크기의 세 architecture(a-c), Dual-DPT head(d), teacher label supervision(e), pose conditioning 모듈(f-g)의 효과. \*는 GT pose fusion으로 평가(굵게 각 그룹 내 최고)*

#### 7.2.3 Ablation and Analysis

**Dual-DPT head.** depth와 ray를 별도의 DPT head 둘로 독립 예측하는 변형과 비교한다(Table 7의 (d)). Dual-DPT가 없으면 지표 전반에서 일관되게 하락해 설계의 효과가 확인된다.

**Teacher 모델 supervision.** teacher label supervision을 빼면(Table 7의 (e)) DTU에서는 약간 좋아지지만 7Scenes·ScanNet++에서 하락하고, 특히 HiRoom에서 크게 나빠진다. HiRoom이 synthetic이라 GT에 미세 구조가 풍부한데, teacher supervision이 student가 이런 디테일을 정확히 잡도록 돕기 때문으로 본다.

![teacher label 유무에 따른 depth map 디테일 비교](/assets/images/depth-anything-3-fig8.jpg)
*[Figure 8] teacher가 생성한 label로 supervision하면 훨씬 풍부한 디테일과 미세 구조의 depth map을 얻는 비교*

**Pose conditioning.** pose conditioning 모듈의 ablation은 Table 7의 (f)(g)다. 이 두 항목만 GT pose fusion으로 평가했는데, pose conditioning이 있는 구성이 지표 전반에서 일관되게 우수하다.

**실행 시간.** 파라미터, 최대 이미지 수, 실행 속도 분석은 Table 8과 같다. 최대 이미지 수는 80GB A100 기준이고, 속도는 32장 장면에서 이미지당 평균 속도다(해상도 504×336).

| Model | Max # of Images | Backbone | DualDPT | CameraHead | Running Speed |
| --- | --- | --- | --- | --- | --- |
| VGGT(Reference) | 400-500 | 0.91B | 0.064B | 0.22B | 34.1 FPS |
| DA3-Giant | 900-1000 | 1.130B | 0.050B | 0.018B | 37.6 FPS |
| DA3-Large | 1500-1600 | 0.300B | 0.047B | 0.008B | 78.37 FPS |
| DA3-Base | 2100-2200 | 0.086B | 0.015B | 0.004B | 126.5 FPS |
| DA3-Small | 4000-4100 | 0.022B | 0.003B | 0.001B | 160.5 FPS |

*[Table 8] 모델별 파라미터·최대 이미지 수·실행 속도 비교*

### 7.3 Analysis for Depth-Anything-3-Monocular

#### 7.3.1 Teacher Model

teacher의 지표는 Table 4에 있다. NYU에서 DA2와 동률인 것을 빼면 모든 데이터셋에서 DA2를 일관되게 앞선다. teacher ablation은 ViT-L backbone·배치 64로 수행하고 DA2 벤치마크 프로토콜을 따르며, GT로 정규화한 평균 제곱 오차인 SqRel(Squared Relative Error)을 추가 보고한다. Table 9처럼 geometry(예측 target) 중에서는 depth가 disparity·point map보다 효과적이고, 목적함수는 제안한 full teacher loss가 DA2 loss나 normal loss 없는 변형을 앞선다. 데이터 스케일링도 뚜렷이 기여해, 데이터셋을 V2에서 V3로 올리고 multi-resolution 학습 전략을 채택하면 일관된 개선이 있다.

| 구분 | 설정 | $$\delta_1$$↑ | AbsRel↓ | SqRel↓ |
| --- | --- | --- | --- | --- |
| Data | V2 | 0.919 | 0.087 | 0.596 |
| | V3 | 0.929 | 0.079 | 0.508 |
| | V3 + mr. | **0.938** | **0.072** | **0.452** |
| Geometry | Disparity | 0.919 | 0.095 | 1.033 |
| | Pointmap | 0.912 | 0.096 | 0.693 |
| | Depth | 0.918 | **0.089** | **0.637** |
| Loss | MAE-Loss | 0.918 | 0.089 | 0.637 |
| | w/o Dist. Nor. | 0.918 | 0.087 | 0.600 |
| | Full loss | **0.919** | **0.087** | **0.596** |

*[Table 9] Teacher 모델 ablation(KITTI·NYU·ETH3D·SUN-RGBD·DIODE 평균, 굵게 각 그룹 내 최고)*

#### 7.3.2 Student Model

Table 10처럼 ViT-L backbone의 monocular student는 모든 평가 데이터셋에서 DA2 student를 앞선다. 특히 ETH3D에서 10% 이상, 어려운 SINTEL에서도 +5.1%의 큰 개선을 보인다. 향상은 더 나은 geometry supervision을 주는 강화된 teacher와 확장된 학습 데이터(V3) 덕분이며, teacher-student 증류 framework의 효과를 입증한다.

| Method | KITTI | NYU | SINTEL | ETH3D | DIODE |
| --- | --- | --- | --- | --- | --- |
| DA2 | 94.6 | 97.9 | 77.2 | 86.5 | 95.2 |
| mono-student | **97.1** | **98.0** | **82.3** | **98.8** | **96.5** |

*[Table 10] Monocular student depth 비교($$\delta_1$$↑, 굵게 최고)*

### 7.4 Analysis for Depth-Anything-3-Metric

DepthPro, Metric3D v2, UniDepthv1, UniDepthv2와 NYUv2·KITTI·ETH3D·SUN-RGBD·DIODE(indoor)의 5개 벤치마크에서 비교한다. Table 11처럼 DA3-metric은 ETH3D에서 $$\delta_1 = 0.917$$ 로 2위 UniDepthv2(0.863)를 큰 폭으로 앞서는 SOTA를 달성하고, SUN-RGBD의 AbsRel(0.105) 최고, DIODE 2위를 기록한다. UniDepth 계열이 NYUv2·KITTI에서 최고인 반면, DA3-metric은 모든 벤치마크에 걸쳐 강한 일반화와 경쟁력을 보이며 특히 ETH3D 같은 다양한 실외 장면에서 뛰어나다.

teacher supervision의 ablation도 Table 11 하단에 있다. teacher supervision을 빼면 NYUv2·KITTI 지표가 약간 좋아지고 다른 데이터셋은 비슷한 흥미로운 trade-off가 나타난다. 그러나 Figure 10이 보여주듯 teacher supervision은 선명도와 미세 디테일 품질을 크게 개선해, 표준 지표 너머의 상보적 지식을 제공한다.

| Methods | NYUv2 $$\delta_1$$↑ | NYUv2 AbsRel↓ | KITTI $$\delta_1$$↑ | KITTI AbsRel↓ | ETH3D $$\delta_1$$↑ | ETH3D AbsRel↓ | SUN-RGBD $$\delta_1$$↑ | SUN-RGBD AbsRel↓ | DIODE $$\delta_1$$↑ | DIODE AbsRel↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DepthPro | 0.932 | 0.093 | 0.843 | 0.121 | 0.386 | 0.349 | 0.950 | 0.126 | 0.734 | 0.173 |
| Metric3D v2 | 0.971 | 0.067 | 0.976 | **0.051** | 0.830 | 0.138 | 0.954 | 0.132 | 0.018 | 0.154 |
| UniDepthv1 | **0.980** | **0.061** | **0.978** | **0.051** | 0.234 | 0.464 | 0.971 | 0.113 | 0.570 | 0.266 |
| UniDepthv2 | 0.968 | 0.064 | 0.968 | 0.076 | 0.863 | 0.152 | **0.977** | 0.111 | **0.856** | **0.123** |
| DA3-metric | 0.963 | 0.070 | 0.953 | 0.086 | **0.917** | **0.104** | 0.973 | **0.105** | 0.838 | 0.128 |
| w/ teacher | 0.966 | 0.073 | 0.947 | 0.086 | 0.906 | 0.105 | 0.973 | 0.104 | 0.824 | 0.132 |
| w/o teacher | 0.969 | 0.066 | 0.965 | 0.067 | 0.907 | 0.105 | 0.975 | 0.099 | 0.816 | 0.134 |

*[Table 11] Metric depth estimation의 SOTA 비교(굵게 최고). 하단 두 행은 teacher supervision 유무의 ablation*

### 7.5 Analysis for Feed-forward 3DGS

비교하는 모든 feed-forward 3DGS 모델을 재학습하되, farthest point sampling으로 고른 입력 context view 12장을 쓰도록 학습 구성을 test 설정과 일치시킨다. flash attention과 fully sharded data parallelism 같은 최적화로 모든 모델이 12개 view를 효율적으로 처리하게 하고, 안정된 수렴을 위해 모든 baseline에 depth 학습 loss를 넣는다. 모든 모델은 8장의 A100으로 배치 1, 200K step 학습한다(epipolar attention이 느린 pixelSplat만 100K step).

**시각 품질 분석.** Figure 11의 novel view synthesis 비교처럼, DA3에 3D Gaussian DPT head만 얹어도 기존 SOTA보다 rendering 품질이 크게 좋아진다. 특히 얇은 구조(기둥)나 wide-baseline 입력의 대규모 실외 환경 같은 어려운 영역에서 강하다. 이 결과는 고품질 visual rendering에 robust한 geometry backbone이 중요하다는 Table 5의 정량 결과와 일치한다.

## 8. Conclusion and Discussion

Depth Anything 3는 depth-ray target과 teacher-student supervision으로 학습된 plain transformer가 화려한 architecture 없이도 any-view geometry를 통일할 수 있음을 보인다. scale-aware depth, 픽셀별 ray, adaptive cross-view attention 덕에 모델은 강력한 사전학습 feature를 물려받으면서 가볍고 확장하기 쉽다. 제안한 visual geometry benchmark에서 pose·재구성의 새 기록을 세웠고, giant와 compact 변형 모두 기존 모델을 앞서며, 같은 backbone이 효율적인 feed-forward novel view synthesis 모델도 구동한다.

논문은 DA3를 다목적 3D foundation model로 가는 한 걸음으로 본다. 향후 dynamic 장면으로 추론을 확장하고, 언어·상호작용 신호를 통합하고, geometry 이해와 실행 가능한 world model 사이의 고리를 닫는 더 큰 규모의 사전학습을 탐구할 수 있다. 모델·데이터셋 공개, benchmark, 그리고 여기서 제시한 단순한 모델링 원칙이 범용 3D 인지 연구를 촉진하길 기대한다.
