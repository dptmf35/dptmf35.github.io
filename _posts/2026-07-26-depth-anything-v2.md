---
title: "[논문리뷰] Depth Anything V2 (NeurIPS 2024)"
excerpt: "실측 label을 전부 synthetic 이미지로 바꾸고 62M pseudo-labeled 실사 이미지로 증류해 정밀도와 robustness를 함께 잡은 Depth Anything V2 논문 리뷰"
date: 2026-07-26
categories:
  - 논문리뷰
  - CV
tags:
  - Depth Anything
  - monocular depth estimation
  - synthetic data
  - knowledge distillation
  - pseudo label
  - zero-shot
toc: true
toc_sticky: true
---

> **V1의 robustness는 유지하면서 Marigold급 디테일까지 얻으려면 무엇을 바꿔야 할까?**
> Depth Anything V2의 답은 모델이 아니라 또다시 데이터다. 실측 labeled 이미지를 전부 버리고 **정밀한 synthetic 이미지로 teacher를 키운 뒤, 62M 장의 pseudo-labeled 실사 이미지로 student를 가르친다.**

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [Depth Anything V2](https://arxiv.org/abs/2406.09414) |
| 저자 | Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao |
| 소속 / 연도 | HKU · TikTok, NeurIPS 2024 (arXiv:2406.09414) |
| 분야 | CV / monocular depth estimation |
| 코드 | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |

## 1. Introduction

Monocular Depth Estimation(MDE)은 3D reconstruction·내비게이션·자율주행 같은 고전적 응용은 물론 이미지·비디오·3D 장면 생성 같은 최신 AI 생성 시나리오에서도 기본 역할을 하며 주목받고 있고, open-world 이미지를 다룰 수 있는 MDE 모델이 다수 등장했다. 구조 관점에서 이들은 두 갈래다. 하나는 BEiT·DINOv2 같은 **discriminative 모델** 기반, 다른 하나는 Stable Diffusion(SD) 같은 **generative 모델** 기반이다. 각 진영의 대표인 Depth Anything(discriminative)과 Marigold(generative)를 비교하면, Marigold는 디테일 모델링이 뛰어나고 Depth Anything은 복잡한 장면에서 더 robust하다. 또한 Depth Anything이 더 효율적이고 가벼우며 모델 크기 선택지도 있지만, 투명 물체와 반사에는 취약하다(이건 Marigold의 강점이다).

| Preferable Properties | Fine Detail | Transparent Objects | Reflections | Complex Scenes | Efficiency | Transferability |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| Marigold | **✓** | **✓** | **✓** | ✗ | ✗ | ✗ |
| Depth Anything V1 | ✗ | ✗ | ✗ | **✓** | **✓** | **✓** |
| Depth Anything V2 (Ours) | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

*[Table 1] 강력한 MDE 모델이 갖춰야 할 성질들*

이 논문의 목표는 Table 1의 강점을 **전부** 갖춘 더 유능한 MDE foundation model이다. 즉 복잡한 layout·투명 물체(유리)·반사면(거울, 스크린)에 robust하고, 얇은 물체(의자 다리)나 작은 구멍까지 Marigold에 견줄 디테일을 담고, 다양한 모델 크기와 추론 효율을 제공하며, downstream 과제로 fine-tuning될 만큼 일반화 가능해야 한다(V1은 3rd MDEC의 모든 상위 팀이 사전학습 모델로 썼다).

MDE의 본질이 discriminative 과제이므로 Depth Anything V1에서 출발해 강점은 유지하고 약점을 고친다. 흥미롭게도 이 어려운 목표에 화려한 신기술이 필요 없다는 것을 보인다. **가장 결정적인 것은 여전히 데이터다.** V1이 대규모 unlabeled 데이터로 데이터 스케일업을 이뤘던 것과 같은 동기인데, 이번에는 먼저 **labeled 데이터 설계를 재검토**하고 그 다음 unlabeled 데이터의 핵심 역할을 부각한다. 세 가지 핵심 발견을 문답으로 요약하면:

- **Q1 (§2)**: MiDaS·Depth Anything의 거친(coarse) depth는 discriminative 모델링 자체의 한계인가? 디테일에는 무거운 diffusion 방식이 필수인가?
  **A1**: 아니다. 효율적인 discriminative 모델도 극도로 정밀한 디테일을 낼 수 있다. 가장 결정적인 수정은 **모든 labeled 실사 이미지를 정밀한 synthetic 이미지로 교체**하는 것이다.
- **Q2 (§3)**: synthetic이 그렇게 우월하다면 왜 기존 연구 대부분은 실사 이미지를 고수하나?
  **A2**: synthetic 이미지에는 기존 패러다임으로는 해결이 쉽지 않은 고유한 단점들이 있다.
- **Q3 (§4)**: synthetic의 단점을 피하면서 장점을 증폭하려면?
  **A3**: synthetic만으로 학습한 **teacher를 크게 키우고**, **대규모 pseudo-labeled 실사 이미지를 다리 삼아** 더 작은 student들을 가르친다.

![다리, 실내, 군중, 스케치, 나비 등 다양한 이미지에 대한 V2의 depth 예측과 Marigold·V1과의 비교, 아래는 latency·파라미터·정확도 막대 비교](/assets/images/depth-anything-v2-fig1.jpg)
*[Figure 1] robustness와 fine-grained 디테일에서 V1을 크게 앞서고, SD 기반 모델보다 빠르고(10배 이상) 가볍고 정확한 Depth Anything V2*

이 탐구 끝에 더 유능한 MDE foundation model을 완성하지만, 기존 test set이 너무 노이즈가 많아 모델의 진짜 강점을 반영하지 못한다는 것도 발견한다. 그래서 정밀한 annotation과 다양한 장면을 갖춘 새 평가 벤치마크 **DA-2K**를 함께 구축한다(§6).

## 2. Revisiting the Labeled Data Design of Depth Anything V1

zero-shot MDE의 선구자 MiDaS 이후, 최근 연구들은 더 큰 labeled 학습 데이터셋을 모으는 경향이다. Depth Anything V1은 1.5M, Metric3D V1·V2는 8M·16M, ZeroDepth는 15M 장의 labeled 이미지를 모았다. 그런데 이 경향을 비판적으로 검토한 연구는 드물다. **그 많은 labeled 이미지가 정말 유리한가?**

**실사 labeled 데이터의 두 가지 단점.** ① **Label 노이즈**: 수집 절차의 한계 때문에 depth map에 부정확한 label이 불가피하게 섞인다. depth 센서는 투명 물체의 depth를 못 잡고(Figure 3(a)), stereo matching은 textureless·반복 패턴에 취약하며(3(b)), SfM은 동적 물체·outlier에 흔들린다(3(c)). ② **디테일 무시**: 나무나 의자의 depth가 눈에 띄게 거칠게 표현되는 등, 물체 경계나 얇은 구멍의 세밀한 supervision을 주지 못해 over-smooth한 depth 예측을 낳는다. 그 결과 학습된 모델도 같은 실수를 반복한다(3(d)). 예컨대 Transparent Surface Challenge에서 MiDaS는 25.9%, V1은 53.5%의 낮은 점수를 받았다(V2는 zero-shot으로 83.6%).

이를 극복하려고 학습 데이터 자체를 바꿔 훨씬 나은 annotation을 찾는다. 완전한 depth 정보를 가진 synthetic 이미지만으로 학습한 최근 SD 기반 연구들에서 영감을 받아 synthetic 이미지의 label 품질을 광범위하게 검토했고, 위 단점들을 해소할 잠재력을 확인했다.

![위: 실사 데이터셋(HRWSI, DIML)의 거친 depth label, 아래: synthetic 데이터셋(Hypersim, vKITTI)의 정밀한 depth label](/assets/images/depth-anything-v2-fig4.jpg)
*[Figure 4 (a)(b)] 실사 데이터의 coarse depth와 synthetic 데이터의 정밀한 depth 비교*

**Synthetic 이미지의 장점.** depth label이 두 측면에서 매우 정밀하다. ① 경계·얇은 구멍·작은 물체 같은 **모든 fine detail이 올바르게 label**되어 있다. 가는 메시 구조나 나뭇잎까지 실제 depth로 annotation된다. ② depth 센서가 못 잡는 **투명 물체·반사면의 실제 depth**를 얻을 수 있다. 한마디로 synthetic 이미지의 depth는 진짜 "GT"다. 게다가 그래픽 엔진에서 빠르게 늘릴 수 있고 실사 이미지와 달리 프라이버시·윤리 문제도 없다.

## 3. Challenges in Using Synthetic Data

synthetic 데이터가 그렇게 유리하다면 왜 실사 데이터가 여전히 MDE를 지배하나? 현실 사용을 가로막는 두 한계가 있다.

**한계 1: 분포 차이(distribution shift).** 그래픽 엔진이 photorealistic을 지향해도 스타일·색 분포가 실사와 뚜렷이 다르다. synthetic 이미지는 색이 너무 "깨끗하고" layout이 너무 "정돈"된 반면 실사는 무작위성이 크다. 이 차이 때문에 layout이 비슷해도 synthetic에서 real로의 전이가 어렵다.

**한계 2: 제한된 장면 커버리지.** synthetic 이미지는 "거실", "거리" 같은 미리 정의된 장면 타입에서 반복 샘플링된다. Hypersim·Virtual KITTI가 아무리 정밀해도, 거기서 학습한 모델이 "붐비는 군중" 같은 실세계 장면에 일반화되길 기대할 수 없다. 반면 웹 stereo 이미지(HRWSI)나 단안 비디오(MegaDepth)로 만든 일부 실사 데이터셋은 광범위한 실세계 장면을 덮는다.

그래서 MDE에서 **synthetic-to-real 전이는 간단치 않다.** 검증을 위해 BEiT, SAM, SynCLR, DINOv2 네 가지 사전학습 encoder로 synthetic 이미지만으로 MDE를 학습하는 pilot study를 했더니, **DINOv2-G만 만족스러운 결과**를 냈고 다른 모든 계열(더 작은 DINOv2 포함)은 심각한 일반화 문제를 보였다. 그렇다면 가장 큰 DINOv2 encoder에 기대면 되는가? 이 나이브한 해법에는 두 문제가 있다. 첫째, synthetic 학습 이미지에 드문 패턴(하늘·구름, 사람)이 실사 test에 나오면 DINOv2-G조차 자주 실패한다.

![왼쪽: 하늘(구름)이 초원거리로 예측되지 않는 실패, 오른쪽: 사람 머리의 depth가 몸과 일관되지 않는 실패](/assets/images/depth-anything-v2-fig6.jpg)
*[Figure 6] synthetic 이미지만으로 학습한 최강 DINOv2-G 모델의 실패 사례*

둘째, 대부분의 응용은 1.3B짜리 DINOv2-G의 저장·추론 비용을 감당할 수 없다. 실제로 V1에서 가장 널리 쓰인 것은 실시간 속도의 최소 모델이다. 실사·synthetic을 섞어 학습하는 대안은 실사의 거친 depth가 fine-grained 예측을 파괴해 버리고(부록 B.9), synthetic 이미지를 더 모으는 것은 모든 실세계 시나리오를 흉내내는 그래픽 엔진을 만들 수 없어 지속 불가능하다. 따라서 synthetic 데이터로 MDE를 만드는 신뢰할 만한 해법이 필요하다.

## 4. Key Role of Large-Scale Unlabeled Real Images

해법은 단순하다. **unlabeled 실사 이미지를 투입한다.** DINOv2-G 기반의 최강 MDE 모델을 고품질 synthetic 이미지만으로 먼저 학습하고, 이 모델로 unlabeled 실사 이미지에 pseudo depth label을 단 뒤, 새 모델들은 **대규모의 정밀하게 pseudo-label된 이미지로만** 학습한다. V1도 대규모 unlabeled 실사 데이터의 중요성을 강조했지만, labeled가 synthetic뿐인 이 특수한 맥락에서는 그 역할이 세 관점에서 필수불가결하다.

- **도메인 격차의 다리.** 분포 차이 때문에 synthetic 학습에서 실사 test로의 직접 전이는 어렵지만, 실사 이미지를 **중간 학습 대상**으로 끼우면 과정이 훨씬 신뢰할 만해진다. pseudo-labeled 실사 이미지로 명시적으로 학습하고 나면 모델이 실세계 데이터 분포에 익숙해진다. 자동 생성된 pseudo label은 수동 annotation보다 훨씬 fine-grained하고 완전하다
- **장면 커버리지 확대.** 공개 데이터셋의 대규모 unlabeled 이미지로 수많은 별개 장면을 쉽게 덮는다. synthetic 이미지는 미리 정의된 비디오에서 반복 샘플링되어 사실 매우 중복적인 반면, unlabeled 실사 이미지는 뚜렷이 구별되고 정보량이 크다. 충분한 이미지·장면으로 학습한 모델은 zero-shot MDE가 강해질 뿐 아니라 downstream 과제의 더 나은 사전학습 소스가 된다
- **최강 모델에서 작은 모델로의 지식 전이.** 작은 모델은 스스로는 synthetic-to-real 전이를 해내지 못하지만, 대규모 unlabeled 실사 이미지로 무장하면 최강 모델의 고품질 예측을 흉내내며 배울 수 있다. knowledge distillation과 비슷하지만, feature/logit 수준이 아니라 **추가 unlabeled 실사 데이터를 통한 label 수준의 증류**라는 점이 다르다. teacher-student의 규모 차가 클 때 feature 증류가 항상 이롭지는 않다는 증거가 있어, 이 방식이 더 안전하다

## 5. Depth Anything V2

### 5.1 Overall Framework

위 분석에 따라 최종 학습 pipeline은 세 단계다.

1. **DINOv2-G 기반의 신뢰할 만한 teacher를 고품질 synthetic 이미지만으로 학습**
2. 대규모 unlabeled 실사 이미지에 **정밀한 pseudo depth 생성**
3. **pseudo-labeled 실사 이미지로 최종 student들을 학습**해 robust한 일반화 확보 (이 단계에서 synthetic 이미지는 필요 없음을 보인다)

![맨 왼쪽: 정밀하지만 분포 차이와 다양성 한계가 있는 synthetic 이미지로 최대 teacher 학습, 가운데: teacher가 unlabeled 실사 이미지에 pseudo label 생성, 오른쪽: 다양하고 정밀한 pseudo-labeled 실사 이미지로 student 학습](/assets/images/depth-anything-v2-fig7.png)
*[Figure 7] Depth Anything V2의 3단계 학습 pipeline*

student 모델은 DINOv2 small·base·large·giant 기반의 네 가지를 공개한다.

### 5.2 Details

학습에는 5개의 정밀한 synthetic 데이터셋(595K 장)과 8개의 대규모 pseudo-labeled 실사 데이터셋(62M 장)을 쓴다.

| Dataset | Indoor | Outdoor | # Images |
| --- | :-: | :-: | --- |
| **Precise *Synthetic* Images (595K)** | | | |
| BlendedMVS | ✓ | ✓ | 115K |
| Hypersim | ✓ | | 60K |
| IRS | ✓ | | 103K |
| TartanAir | ✓ | ✓ | 306K |
| VKITTI 2 | | ✓ | 20K |
| **Pseudo-labeled *Real* Images (62M)** | | | |
| BDD100K | | ✓ | 8.2M |
| Google Landmarks | | ✓ | 4.1M |
| ImageNet-21K | ✓ | ✓ | 13.1M |
| LSUN | ✓ | | 9.8M |
| Objects365 | ✓ | ✓ | 1.7M |
| Open Images V7 | ✓ | ✓ | 7.8M |
| Places365 | ✓ | ✓ | 6.5M |
| SA-1B | ✓ | ✓ | 11.1M |

*[Table 7] 학습 데이터 소스 구성*

V1과 같이 pseudo-labeled 샘플마다 **loss 상위 $$n$$% 영역을 무시**한다($$n = 10$$). 잠재적으로 노이즈인 pseudo label로 간주하는 것이다. 모델의 출력도 V1과 같은 affine-invariant inverse depth다. labeled(synthetic) 이미지 최적화에는 두 loss를 쓴다. **scale-shift-invariant loss $$\mathcal{L}_{ssi}$$** 와 **gradient matching loss $$\mathcal{L}_{gm}$$** 인데, 둘 다 MiDaS가 제안한 것으로 새롭지 않지만, synthetic 이미지를 쓸 때 $$\mathcal{L}_{gm}$$ 이 **depth 선명도(sharpness)에 매우 유익**하다는 것을 발견한다(부록 B.7). pseudo-labeled 이미지에는 V1을 따라 사전학습 DINOv2 encoder의 유용한 semantic을 보존하는 **feature alignment loss**를 추가한다.

## 6. A New Evaluation Benchmark: DA-2K

### 6.1 Limitations in Existing Benchmarks

§2에서 실사 학습 데이터의 label 노이즈를 보였는데, **널리 쓰이는 test 벤치마크 역시 노이즈가 많다.** 전용 depth 센서를 썼는데도 NYU-D에는 거울·얇은 구조물의 annotation이 틀려 있고, 이런 잦은 label 노이즈는 강력한 MDE 모델의 보고 지표를 더 이상 신뢰할 수 없게 만든다. 또 다른 문제는 **다양성 부족**이다. NYU-D는 실내 방 몇 개, KITTI는 거리 장면 몇 개처럼 대부분 단일 장면용으로 만들어져, 이들 벤치마크의 성능이 실세계 신뢰도를 반영하지 못할 수 있다. 마지막 문제는 **저해상도**다. 대부분 500×500 정도의 이미지를 제공하는데 현대 카메라는 1000×2000 같은 고해상도 depth 추정을 요구하므로, 저해상도 벤치마크의 결론이 고해상도로 안전하게 옮겨지는지 불분명하다.

### 6.2 DA-2K

세 한계를 고려해 ① 정밀한 depth 관계를 제공하고 ② 광범위한 장면을 덮고 ③ 대부분 고해상도인 relative MDE 평가 벤치마크를 구축한다. in-the-wild 이미지의 픽셀별 depth를 사람이 다는 것은 비현실적이므로, DIW를 따라 이미지마다 **sparse depth 쌍**을 annotation한다. 이미지에서 픽셀 두 개를 골라 **어느 쪽이 더 가까운지**만 판정하는 것이다.

픽셀 쌍 선정은 두 pipeline을 쓴다. 첫째 pipeline에서는 SAM으로 물체 mask를 자동 예측하되 mask 대신 그것을 유도한 key point(픽셀)를 활용한다. 무작위로 key 픽셀 두 개를 뽑아 4개의 전문 모델에 상대 depth를 투표시키고, 의견이 갈리면 사람 annotator에게 보내 참 상대 depth를 정한다(모호하면 건너뛸 수 있다). 다만 모든 모델이 똑같이 틀려서 걸러지지 않는 어려운 쌍이 있을 수 있으므로, 이미지를 신중히 분석해 **어려운 쌍을 수동 발굴**하는 둘째 pipeline을 더한다.

![왼쪽: 이미지에서 SAM key point로 픽셀 쌍을 샘플링, 가운데: V1·V2·Marigold·Geowizard 네 모델이 상대 depth를 투표, 오른쪽: 불일치 시 사람이 판정하고 전원 일치면 재샘플링하는 흐름](/assets/images/depth-anything-v2-fig9.jpg)
*[Figure 9 (a)] DA-2K annotation pipeline*

정밀성을 위해 모든 annotation은 다른 두 annotator가 3중 확인한다. 다양성을 위해 MDE의 8대 응용 시나리오를 정리하고 GPT-4로 시나리오별 키워드를 생성해 Flickr에서 해당 이미지를 내려받았다. 최종적으로 **1K 이미지에 총 2K 픽셀 쌍**을 annotation했다.

**DA-2K의 위치.** DA-2K가 기존 벤치마크를 대체하리라 기대하지는 않는다. 정확한 sparse depth는 장면 재구성에 필요한 정밀한 dense depth와는 거리가 멀다. 그러나 dense depth의 **전제 조건**으로서, 광범위한 장면 커버리지와 정밀성 덕에 기존 벤치마크의 귀중한 보완이 될 수 있다. 특정 시나리오용 커뮤니티 모델을 고르는 사용자의 빠른 사전 검증으로도, 미래 멀티모달 LLM의 3D 인지 testbed로도 쓰일 수 있다.

## 7. Experiment

### 7.1 Implementation details

V1과 같이 DINOv2 encoder 위에 DPT decoder를 쓴다. 모든 이미지는 짧은 변을 518로 resize한 뒤 random crop해 518×518로 학습한다. teacher(synthetic) 학습은 배치 64로 160K iteration, 3단계 pseudo-labeled 실사 학습은 배치 192로 480K iteration이다. Adam optimizer로 encoder lr 5e-6, decoder lr 5e-5. 두 단계 모두 데이터셋 균형 없이 단순 연결(concatenate)한다. $$\mathcal{L}_{ssi}$$ 와 $$\mathcal{L}_{gm}$$ 의 가중 비율은 1:2다.

### 7.2 Zero-Shot Relative Depth Estimation

**기존 벤치마크 성능.** affine-invariant inverse depth를 예측하므로 공정성을 위해 V1·MiDaS V3.1과 5개 미학습 test 데이터셋에서 비교한다.

| Method | Encoder | KITTI AbsRel↓ | KITTI $$\delta_1$$↑ | NYU-D AbsRel↓ | NYU-D $$\delta_1$$↑ | Sintel AbsRel↓ | Sintel $$\delta_1$$↑ | ETH3D AbsRel↓ | ETH3D $$\delta_1$$↑ | DIODE AbsRel↓ | DIODE $$\delta_1$$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiDaS V3.1 | ViT-L | 0.127 | 0.850 | 0.048 | 0.980 | 0.587 | 0.699 | 0.139 | 0.867 | 0.075 | 0.942 |
| Depth Anything V1 | ViT-S | 0.080 | 0.936 | 0.053 | 0.972 | 0.464 | 0.739 | 0.127 | **0.885** | 0.076 | 0.939 |
| | ViT-B | 0.080 | 0.939 | 0.046 | 0.979 | **0.432** | 0.756 | **0.126** | 0.884 | 0.069 | 0.946 |
| | ViT-L | 0.076 | 0.947 | **0.043** | **0.981** | 0.458 | 0.760 | 0.127 | 0.882 | 0.066 | 0.952 |
| **Depth Anything V2** | ViT-S | 0.078 | 0.936 | 0.053 | 0.973 | 0.500 | 0.718 | 0.142 | 0.851 | 0.073 | 0.942 |
| | ViT-B | 0.078 | 0.939 | 0.049 | 0.976 | 0.495 | 0.734 | 0.137 | 0.858 | 0.068 | 0.950 |
| | ViT-L | **0.074** | 0.946 | 0.045 | 0.979 | 0.487 | 0.752 | 0.131 | 0.865 | 0.066 | 0.952 |
| | ViT-G | 0.075 | **0.948** | 0.044 | 0.979 | 0.506 | **0.772** | 0.132 | 0.862 | **0.065** | **0.954** |

*[Table 2] Zero-shot relative depth estimation(굵게 최고). 지표만 보면 V2는 MiDaS보다 낫고 V1과 대등한 수준*

MiDaS보다 우수하고 V1과는 대등하다(두 데이터셋의 지표는 V1보다 약간 낮다). 그러나 이 데이터셋들의 단순 지표는 이 논문의 초점이 아니다. V2의 목표인 얇은 구조의 fine-grained 예측, 복잡한 장면·투명 물체에서의 robust한 예측 향상은 **현재 벤치마크에 올바르게 반영될 수 없다.**

**DA-2K 성능.** 다양한 장면의 자체 벤치마크에서는 **가장 작은 모델조차 무거운 SD 기반 모델들을 크게 앞선다.**

| Method | Marigold | Geowizard | DepthFM | Depth Anything V1 | V2 ViT-S | V2 ViT-B | V2 ViT-L | V2 ViT-G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy (%) | 86.8 | 88.1 | 85.8 | 88.5 | 95.3 | 97.0 | 97.1 | **97.4** |

*[Table 3] 8개 대표 시나리오를 아우르는 DA-2K 벤치마크 성능*

최강 모델은 상대 depth 판별 정확도에서 Marigold보다 10.6% 높다.

### 7.3 Fine-tuned to Metric Depth Estimation

일반화 검증을 위해 encoder를 downstream metric depth 추정으로 전이한다. V1과 같이 ZoeDepth pipeline의 MiDaS encoder를 사전학습된 V2 encoder로 교체한다.

| Method | $$\delta_1$$↑ | $$\delta_2$$↑ | $$\delta_3$$↑ | AbsRel↓ | RMSE↓ | log10↓ |
| --- | --- | --- | --- | --- | --- | --- |
| AdaBins | 0.903 | 0.984 | 0.997 | 0.103 | 0.364 | 0.044 |
| DPT | 0.904 | 0.988 | 0.998 | 0.110 | 0.357 | 0.045 |
| P3Depth | 0.898 | 0.981 | 0.996 | 0.104 | 0.356 | 0.043 |
| SwinV2 | 0.949 | 0.994 | 0.999 | 0.083 | 0.287 | 0.035 |
| AiT | 0.954 | 0.994 | 0.999 | 0.076 | 0.275 | 0.033 |
| VPD | 0.964 | 0.995 | 0.999 | 0.069 | 0.254 | 0.030 |
| IEBins | 0.936 | 0.992 | 0.998 | 0.087 | 0.314 | 0.038 |
| ZoeDepth | 0.951 | 0.994 | 0.999 | 0.077 | 0.282 | 0.033 |
| Ours (ViT-S) | 0.961 | 0.996 | 0.999 | 0.073 | 0.261 | 0.032 |
| Ours (ViT-B) | 0.977 | 0.997 | 1.000 | 0.063 | 0.228 | 0.027 |
| Ours (ViT-L) | **0.984** | **0.998** | **1.000** | **0.056** | **0.206** | **0.024** |

*[Table 4 (a)] NYU-D에서의 in-domain metric depth 추정(비교 방법은 ViT-L급 encoder 사용)*

| Method | $$\delta_1$$↑ | $$\delta_2$$↑ | $$\delta_3$$↑ | AbsRel↓ | RMSE↓ | RMSE log↓ |
| --- | --- | --- | --- | --- | --- | --- |
| AdaBins | 0.964 | 0.995 | 0.999 | 0.058 | 2.360 | 0.088 |
| P3Depth | 0.953 | 0.993 | 0.998 | 0.071 | 2.842 | 0.103 |
| NeWCRFs | 0.974 | 0.997 | 0.999 | 0.052 | 2.129 | 0.079 |
| SwinV2 | 0.977 | 0.998 | 1.000 | 0.050 | 1.966 | 0.075 |
| NDDepth | 0.978 | 0.998 | 0.999 | 0.050 | 2.025 | 0.075 |
| GEDepth | 0.976 | 0.997 | 0.999 | 0.048 | 2.044 | 0.076 |
| IEBins | 0.978 | 0.998 | 0.999 | 0.050 | 2.011 | 0.075 |
| ZoeDepth | 0.971 | 0.996 | 0.999 | 0.054 | 2.281 | 0.082 |
| Ours (ViT-S) | 0.973 | 0.997 | 0.999 | 0.053 | 2.235 | 0.081 |
| Ours (ViT-B) | 0.979 | 0.998 | 1.000 | 0.048 | 1.999 | 0.072 |
| Ours (ViT-L) | **0.983** | **0.998** | **1.000** | **0.045** | **1.861** | **0.067** |

*[Table 4 (b)] KITTI에서의 in-domain metric depth 추정*

NYU-D·KITTI 모두에서 기존 방법 대비 큰 개선을 이루고, **ViT-S 기반의 가장 가벼운 모델조차 ViT-L 기반의 다른 모델들을 앞선다.** 다만 지표는 인상적이어도 NYUv2·KITTI로 학습한 모델은 학습 셋 고유의 노이즈 때문에 fine-grained 예측이 안 되고 투명 물체에 robust하지 않다. 그래서 multi-view 합성 같은 실세계 응용을 위해, 실내는 **Hypersim**, 실외는 **Virtual KITTI**의 synthetic 데이터셋으로 encoder를 fine-tuning한 metric depth 모델 둘도 공개한다.

### 7.4 Ablation Study

지면 관계로 대부분의 ablation은 부록으로 미루고 pseudo label 관련 둘만 본다.

**대규모 pseudo-labeled 실사 이미지의 중요성.**

| Encoder | $$\mathcal{D}^l$$ | $$\mathcal{D}^u$$ | KITTI AbsRel↓ | KITTI $$\delta_1$$↑ | NYU-D AbsRel↓ | NYU-D $$\delta_1$$↑ | Sintel AbsRel↓ | Sintel $$\delta_1$$↑ | ETH3D AbsRel↓ | ETH3D $$\delta_1$$↑ | DIODE AbsRel↓ | DIODE $$\delta_1$$↑ | DA-2K Acc(%) |
| --- | :-: | :-: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S | ✓ | | 0.104 | 0.889 | 0.084 | 0.928 | 0.518 | 0.702 | 0.155 | 0.827 | 0.087 | 0.926 | 89.8 |
| | ✓ | ✓ | 0.085 | 0.928 | 0.054 | 0.971 | **0.491** | **0.723** | 0.143 | 0.849 | 0.074 | 0.941 | 94.1 |
| | | ✓ | **0.078** | **0.936** | **0.053** | **0.973** | 0.500 | 0.718 | **0.142** | **0.851** | **0.073** | **0.942** | **95.3** |
| ViT-B | ✓ | | 0.094 | 0.912 | 0.062 | 0.963 | 0.618 | 0.715 | 0.148 | 0.842 | 0.076 | 0.940 | 92.9 |
| | ✓ | ✓ | 0.080 | 0.938 | **0.049** | **0.976** | 0.515 | 0.732 | **0.137** | **0.859** | **0.068** | **0.950** | 96.7 |
| | | ✓ | **0.078** | **0.939** | **0.049** | **0.976** | **0.495** | **0.734** | **0.137** | 0.858 | **0.068** | **0.950** | **97.0** |
| ViT-L | ✓ | | 0.081 | 0.937 | 0.048 | 0.976 | 0.516 | 0.731 | 0.133 | 0.864 | 0.071 | 0.949 | 96.0 |
| | ✓ | ✓ | 0.075 | **0.947** | **0.045** | **0.979** | 0.542 | 0.741 | **0.130** | **0.866** | **0.066** | **0.953** | **97.3** |
| | | ✓ | **0.074** | 0.946 | **0.045** | **0.979** | **0.487** | **0.752** | 0.131 | 0.865 | **0.066** | 0.952 | 97.1 |
| Teacher (ViT-G) | | | 0.075 | 0.947 | 0.044 | 0.979 | 0.530 | 0.767 | 0.131 | 0.865 | 0.066 | 0.954 | 97.4 |

*[Table 5] pseudo-labeled(unlabeled) 실사 이미지 $$\mathcal{D}^u$$ 의 중요성($$\mathcal{D}^l$$: 정밀 labeled synthetic 이미지)*

synthetic만으로 학습했을 때에 비해 pseudo-labeled 실사 이미지를 넣으면 모델이 크게 강해진다. V1과 달리 **student 학습에서 synthetic 이미지를 아예 빼는 것**도 시도했는데, 작은 모델(ViT-S·ViT-B)에서는 오히려 약간 더 좋았다. 그래서 최종 student는 **순수하게 pseudo-labeled 이미지로만** 학습한다. pseudo-label mask만 공개한 SAM과 비슷한 관찰이다.

**Pseudo label vs. 수동 label.** 실사 labeled 데이터셋의 노이즈를 정량 비교한다. DIML 데이터셋의 실사 이미지를 원래의 수동 label과 자체 생성 pseudo label로 각각 학습해 전이 성능을 비교하면:

| Label Source | KITTI AbsRel↓ | KITTI $$\delta_1$$↑ | NYU-D AbsRel↓ | NYU-D $$\delta_1$$↑ | Sintel AbsRel↓ | Sintel $$\delta_1$$↑ | ETH3D AbsRel↓ | ETH3D $$\delta_1$$↑ | DIODE AbsRel↓ | DIODE $$\delta_1$$↑ | DA-2K Acc(%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Manual Label | 0.122 | 0.882 | 0.074 | 0.952 | 0.581 | 0.693 | 0.159 | 0.832 | 0.126 | 0.890 | 80.2 |
| Pseudo Label | **0.099** | **0.901** | **0.062** | **0.963** | **0.514** | **0.701** | **0.147** | **0.843** | **0.084** | **0.929** | **89.7** |

*[Table 6] DIML 데이터셋에서 원래 수동 label과 자체 pseudo label의 비교*

pseudo label로 학습한 모델이 수동 label보다 현저히 낫다. 이 큰 격차가 pseudo label의 높은 품질과 현재 실사 labeled 데이터셋의 풍부한 노이즈를 동시에 보여준다.

## 8. Related Work

**Monocular depth estimation.** 초기 연구는 학습·test 이미지가 같은 도메인이어야 하는 in-domain metric depth에 집중했고, 응용이 제한적이라 최근에는 zero-shot relative MDE로 관심이 옮겨 왔다. 일부는 Stable Diffusion을 depth denoiser로 쓰는 등 **모델링 방식**으로 접근하고, 다른 연구들은 **데이터 관점**으로 접근한다(MiDaS 2M, Metric3D 8M labeled 이미지, V1은 62M unlabeled 이미지). 이 논문은 이와 달리 널리 쓰이는 labeled 실사 이미지의 여러 한계를 지적하고 depth 정밀성을 위해 synthetic 이미지에 의지할 필요성을 부각하며, synthetic이 유발하는 일반화 문제는 데이터 기반(대규모 pseudo-labeled 실사 이미지)과 모델 기반(teacher 스케일업) 전략을 함께 써서 해결한다.

**Learning from unlabeled real images.** semi-supervised learning에서 널리 연구됐지만 소규모 labeled·unlabeled만 허용하는 학술 벤치마크에 집중해 왔다. 여기서는 0.6M labeled 이미지의 baseline을 62M unlabeled 이미지로 끌어올리는 실전 시나리오를 다루고, 특히 labeled 실사 이미지를 모두 synthetic으로 교체한 상황에서 unlabeled 실사 이미지의 필수 역할을 보인다. **"정밀한 synthetic 데이터 + pseudo-labeled 실사 데이터"가 labeled 실사 데이터보다 유망한 로드맵**임을 입증한다.

**Knowledge distillation.** 최강 teacher에서 작은 모델로 전이 가능한 지식을 증류한다는 점에서 KD의 핵심 정신과 비슷하지만, KD가 보통 labeled 이미지에서 feature/logit 수준의 증류 전략을 연구하는 것과 달리 **추가 unlabeled 실사 이미지를 통한 예측(label) 수준의 증류**라는 점이 근본적으로 다르다. 정교한 loss나 증류 pipeline 설계가 아니라 대규모 unlabeled 데이터와 더 큰 teacher의 중요성을 드러내는 것이 목적이다. 규모 차가 큰 두 모델 간 feature 증류는 사실 위험한데, 이 pseudo-label 증류는 1.3B 모델에서 25M 모델로도 쉽고 안전하게 동작한다.

## 9. Conclusion

이 논문은 monocular depth estimation을 위한 더 강력한 foundation model **Depth Anything V2**를 제시한다. ① robust하면서 fine-grained한 depth 예측을 제공하고, ② 25M부터 1.3B 파라미터까지 다양한 모델 크기로 광범위한 응용을 지원하며, ③ 유망한 모델 초기화로서 downstream 과제에 쉽게 fine-tuning된다. 강한 MDE 모델로 가는 길을 닦는 핵심 발견들을 밝혔고, 나아가 기존 test set의 빈약한 다양성과 풍부한 노이즈를 직시해 정밀하고 도전적인 sparse depth label을 가진 다양한 고해상도 이미지의 다목적 평가 벤치마크 **DA-2K**를 구축했다.
