---
title: "[공부] point cloud 정합 기반 6DoF 추정: ICP, FPFH, TEASER++"
excerpt: "ICP·FPFH+RANSAC·TEASER++로 관측 point cloud를 모델에 정합하는 registration 기반 6DoF 추정"
date: 2026-07-03
categories:
  - 공부
  - CV
tags:
  - 6DoF pose estimation
  - point cloud registration
  - ICP
  - FPFH
  - TEASER++
toc: true
toc_sticky: true
---

> [시리즈 개요]({% post_url 2026-07-01-6dof-pose-estimation-overview %}) 의 두 번째 방법론. 무늬가 없어도 **기하(깊이) 정보**만으로 포즈를 구하는 접근이다.

## 1. 문제 정의: Registration = 6DoF

깊이 센서로 얻은 관측 point cloud $$\mathcal{P} = \{p_i\}$$ 을, 물체 모델 point cloud $$\mathcal{Q} = \{q_i\}$$ 에 정렬하는 **강체 변환** $$(R, t)$$ 를 찾는 문제다.

$$
\min_{R \in SO(3),\, t \in \mathbb{R}^3} \sum_{i} \big\| R\,p_i + t - q_i \big\|^2
$$

대응 $$p_i \leftrightarrow q_i$$ 가 정확히 주어지면, 이 문제는 **닫힌 해**([SVD 기반 Procrustes/Kabsch]({% post_url 2026-07-08-kabsch-algorithm-robot-calibration %}))로 바로 풀린다. 현실의 어려움은 (1) 대응을 모른다, (2) 대응에 outlier가 많다는 점이다.

## 2. 지역 정합: ICP

**ICP(Iterative Closest Point)** (Besl & McKay, 1992)는 가장 고전적인 방법이다.

1. 현재 변환으로 가장 가까운 점을 대응으로 가정
2. 그 대응으로 최적 $$(R,t)$$ 갱신
3. 수렴까지 반복

- 변형: **point-to-plane**(표면 법선 활용, 수렴 빠름), **Generalized-ICP(GICP)**
- **한계**: 좋은 **초기값**이 필요하고, 지역 최소값에 빠지기 쉬움. 초기 정렬이 나쁘면 실패

즉 ICP는 보통 **정밀화(refinement)** 단계로 쓰고, 대략적 초기 정렬은 전역 정합이 담당한다.

## 3. 전역 정합: 특징 매칭 + RANSAC

초기값 없이 정렬하려면 **기하 특징 descriptor**로 대응 후보를 만든다.

- **FPFH(Fast Point Feature Histograms)** (Rusu, 2009): 각 점 주변의 법선 각도 분포를 histogram으로 요약한 descriptor
- 두 point cloud에서 FPFH를 뽑아 매칭 → 대응 후보 생성 → **RANSAC** 으로 강체 변환 추정 → **ICP로 정밀화**
- Open3D의 표준 파이프라인이 이 구조(전역 RANSAC → 지역 ICP)

문제는 특징 매칭 대응에 **outlier가 매우 많다**는 점이다. RANSAC은 outlier 비율이 높아질수록 필요한 반복 수가 급격히 늘어 느려지고 불안정해진다.

## 4. TEASER++: outlier에 강인하고 인증 가능

**TEASER++** (Yang, Shi, Carlone, *IEEE T-RO 2021*)는 대응의 **99% 이상이 outlier여도** 견고하게 동작하는 정합 알고리즘이다.

### 핵심 아이디어
- **Truncated Least Squares(TLS)** 비용을 사용 → 큰 오차(outlier)의 영향을 잘라냄
- **회전·평행이동·스케일을 분리(decoupling)** 하여 각각 추정 → 문제를 다루기 쉽게 만듦
  - 두 점의 **쌍(pairwise) 거리**는 회전·평행이동에 불변 → 이를 이용해 스케일/outlier를 먼저 걸러냄(**max clique** 로 상호 일관된 대응만 선별)
  - 회전은 **GNC(Graduated Non-Convexity)** 로 비볼록 TLS를 점진적으로 풀어 강인하게 추정
- 볼록 완화(convex relaxation) 기반으로 해의 최적성을 **인증(certifiable)** 할 수 있음

### 전형적 파이프라인
```
point cloud → keypoint → FPFH descriptor → 대응 후보(대량, outlier 많음)
     → TEASER++ (강인 정합) → (선택) ICP 정밀화 → 6DoF pose
```

FPFH+RANSAC이 흔들리는 **높은 outlier 상황에서 특히 강점**이라, 어질러진 실환경 로봇 인식에 잘 맞는다.

![ICP는 초기값에 민감하고, TEASER++는 outlier에 강인한 전역 정합](/assets/images/6dof-icp-vs-teaser.svg)

## 5. 장단점 정리

| 방법 | 초기값 | outlier 강인성 | 속도 | 역할 |
| --- | --- | --- | --- | --- |
| ICP | 필요(민감) | 약함 | 빠름 | 정밀화 |
| FPFH + RANSAC | 불필요 | 중간 | outlier↑ 시 느림 | 전역 초기정렬 |
| TEASER++ | 불필요 | 매우 강함 | 빠름/견고 | 강인 전역정합 |

> **매니퓰레이션 관점**: 무늬 없는 금속 부품처럼 RGB 특징이 빈약한 대상에 강력하다. 다만 **깊이 품질**과 **대응(descriptor) 품질**에 성능이 좌우되고, 대칭 물체는 여전히 모호성이 남는다.

## 정리 & 다음 글

- 정합이 곧 6DoF다. **ICP(정밀) + 전역정합(초기)** 조합이 기본기다.
- 대응 outlier가 심하면 **TEASER++** 가 판을 바꾼다.
- 하지만 대응/segmentation을 사람이 설계해야 하므로, **학습 기반**이 이를 자동화·일반화한다.
- 다음 글: [Model-based Zero-shot 6DoF 추정 (FoundationPose, SAM-6D)]({% post_url 2026-07-04-6dof-model-based-zero-shot %})
