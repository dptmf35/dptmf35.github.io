---
title: "[논문리뷰] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (NeurIPS 2022)"
excerpt: "attention의 병목을 FLOPs가 아닌 HBM 접근으로 규정하고 tiling·recomputation으로 O(N) 메모리의 exact attention을 구현한 FlashAttention 논문 리뷰"
date: 2026-07-13
categories:
  - 논문리뷰
  - AI
tags:
  - FlashAttention
  - attention
  - Transformer
  - GPU
  - tiling
toc: true
toc_sticky: true
---

> **attention이 느린 이유는 계산량이 아니라 메모리 왕복 때문이다.**
> FlashAttention은 근사 없이 수학적으로 동일한(exact) attention을 계산하면서, GPU 메모리 계층 사이의 읽기/쓰기(IO)를 줄이는 것만으로 속도와 메모리를 동시에 잡는다.

## 논문 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) |
| 저자 | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré |
| 소속 / 연도 | Stanford University · University at Buffalo (SUNY), NeurIPS 2022 (arXiv:2205.14135) |
| 분야 | AI / efficient attention, ML systems |
| 코드 | [GitHub](https://github.com/Dao-AILab/flash-attention) |

## 들어가기 전에: GPU는 어떻게 일하는가

이 논문은 "GPU 메모리 계층"을 이해해야 절반이 읽힌다. 본문에 들어가기 전에 CPU와 GPU의 구성요소와 동작 방식을 정리한다.

### CPU와 GPU: 셰프와 뷔페 주방

- **CPU**는 고급 레스토랑의 셰프다. 소수의 매우 숙련된 코어가 복잡한 요리(분기·순차 로직)를 하나씩 깊게 처리한다
- **GPU**는 대형 뷔페 주방이다. 수천 명의 요리사(CUDA core)가 썰기·볶기 같은 **단순한 작업을 동시에** 수행한다. 개별 요리사는 셰프보다 못하지만, 행렬 곱처럼 같은 연산을 수만 번 반복하는 일에는 압도적이다

### 데이터의 여정: CPU에서 CUDA core까지

![CPU와 DRAM에서 PCIe를 거쳐 GPU의 HBM으로 데이터가 오르고, scheduler가 SM들에 작업을 할당하면 각 SM이 SRAM에 블록을 올려 연산하는 구조](/assets/images/gpu-cpu-architecture.svg)

1. **CPU + DRAM**: 작업이 시작되면 CPU가 데이터를 준비하고, system RAM(DRAM)에 로드한다
2. **PCIe bus**: 데이터가 PCIe를 통해 DRAM에서 GPU의 VRAM(**HBM**)으로 이동한다
3. **GPU 내부**: HBM에 데이터가 오르면 **scheduler가 여러 SM(Streaming Multiprocessor)에 작업을 할당**한다. 각 SM은 HBM에서 필요한 데이터 블록을 가져와 자기 **SRAM**에 올리고, **CUDA core / Tensor core**가 그 데이터로 연산(FLOPs)을 수행한 뒤 결과를 HBM에 되돌린다

SM은 CUDA core·SRAM·scheduler를 갖추고 독립적으로 일하는 **실제 작업 단위**다. A100 GPU에는 SM이 108개 있다.

### 메모리 계층: 창고 비유

| 메모리 | 비유 | 크기 | 속도 (A100 기준) |
| --- | --- | --- | --- |
| HBM (VRAM) | 물류 센터의 **중앙 창고**. 크지만 꺼내는 데 오래 걸린다 | 40~80 GB | 1.5~2.0 TB/s |
| SRAM (on-chip) | 손 뻗으면 닿는 **소형 창고**. 자주 쓰는 것만 보관 | SM당 192 KB | ~19 TB/s (HBM의 약 10배) |
| register | 당장 쓰는 것만 두는 **작업대** | 극히 작음 | 가장 빠름 |

핵심은 이것이다: **연산은 빠른데 창고(HBM) 왕복이 느리다.** 연산 속도가 메모리 속도보다 훨씬 빠르게 발전해 왔기 때문에, 딥러닝의 많은 연산은 계산이 아니라 **HBM 접근이 병목**이다. 기존 attention은 재료 하나 썰 때마다 중앙 창고를 왕복하는 방식이었고, FlashAttention은 재료를 바구니(블록)에 담아 와서 소형 창고(SRAM) 위에서 다 끝내고 결과만 돌려놓는 방식이다. 이제 논문으로 들어간다.

## 1. Introduction

Transformer의 심장인 self-attention은 시퀀스 길이 $$N$$ 에 대해 **시간·메모리 복잡도가 $$O(N^2)$$** 이라, 긴 context를 다루기 어렵다. 이를 해결하려는 **approximate attention**(sparse 근사, low-rank 근사 등) 연구가 많았지만, 널리 쓰이지 못했다. 이유는:

- 근사 방법들은 **FLOPs 감소에 집중**하는데, FLOPs는 실제 시간(wall-clock time)과 비례하지 않는다
- **메모리 접근(IO)의 오버헤드를 무시**해서, 계산량을 줄여도 실제로는 빨라지지 않는 경우가 많다

논문은 빠진 원칙이 **IO-awareness**(GPU의 빠른 메모리 SRAM과 느린 메모리 HBM 사이의 읽기/쓰기를 정밀하게 관리하는 것)라고 주장한다. 목표는 attention 행렬을 HBM에 읽고 쓰는 것 자체를 피하는 것이고, 이를 위해 두 가지 과제를 푼다.

1. 전체 입력을 보지 않고 **softmax를 블록 단위로 계산**하기 → **tiling**
2. backward pass를 위해 거대한 중간 attention 행렬을 **저장하지 않기** → softmax 정규화 통계만 저장하고 on-chip에서 **recomputation**

![GPU 메모리 계층(SRAM 19TB/s, HBM 1.5TB/s, DRAM 12.8GB/s)과, K·V 블록의 outer loop와 Q 블록의 inner loop로 attention을 SRAM 안에서 계산하는 tiling 구조, 그리고 GPT-2 attention에서 PyTorch 대비 속도 비교](/assets/images/flashattention-fig1.png)
*[Figure 1] GPU 메모리 계층과 FlashAttention의 tiling 구조(왼쪽·가운데), GPT-2 attention에서 PyTorch 구현 대비 7.6× 속도 향상(오른쪽)*

CUDA로 구현해 모든 attention 연산을 **하나의 GPU kernel로 융합**했고, recomputation으로 FLOPs가 오히려 늘어나는데도 HBM 접근이 크게 줄어 **더 빠르게 실행되며**(GPT-2에서 최대 7.6×), 메모리는 시퀀스 길이에 **선형**이 된다.

IO 복잡도 분석으로는, FlashAttention의 HBM 접근이 $$O(N^2 d^2 M^{-1})$$ ($$d$$: head 차원, $$M$$: SRAM 크기)로 표준 attention의 $$\Omega(N d + N^2)$$ 보다 훨씬 적음을 증명하고, 모든 SRAM 크기에 대해 이를 점근적으로 개선할 수 없다는 **하한(lower bound)** 도 제시한다. 나아가 block-sparse attention으로 확장한 **block-sparse FlashAttention**은 기존의 어떤 approximate attention보다 빠르다.

실험으로 검증한 기여는 세 가지다.

- **더 빠른 학습**: BERT-large(길이 512)를 MLPerf 1.1 기록보다 15% 빠르게, GPT-2(길이 1K)를 HuggingFace·Megatron-LM 대비 최대 3×, long-range arena(1K~4K)를 2.4× 빠르게 학습
- **더 좋은 모델**: 긴 시퀀스가 가능해져 GPT-2 perplexity 0.7 개선, 장문서 분류 6.4점 향상, 그리고 **Path-X(길이 16K)·Path-256(길이 64K)에서 우연 수준을 넘긴 최초의 Transformer**
- **attention 벤치마크**: 일반적 길이(최대 2K)에서 표준 attention 대비 최대 3× 빠르고 메모리는 선형

## 2. Background

### 2.1 Hardware Performance

**GPU 메모리 계층.** 위 사전 지식에서 본 그대로다. A100은 HBM 40~80GB(1.5~2.0TB/s), SM 108개 각각에 on-chip SRAM 192KB(약 19TB/s). SRAM은 HBM보다 한 자릿수 빠르지만 크기는 여러 자릿수 작다. **연산이 메모리 속도보다 빨라지면서, 연산의 병목은 점점 HBM 접근이 되고 있다.**

**실행 모델.** GPU는 연산(kernel)을 수많은 thread로 실행한다. 각 kernel은 입력을 HBM에서 register·SRAM으로 **로드**하고, **계산**한 뒤, 출력을 HBM에 **쓴다**.

**성능 특성.** 연산은 계산과 메모리 접근의 균형에 따라 둘로 나뉜다. 기준은 **arithmetic intensity**(메모리 접근 1바이트당 산술 연산 수)다.

- **compute-bound**: 시간이 산술 연산량으로 결정된다. 내부 차원이 큰 행렬 곱, 채널 많은 convolution
- **memory-bound**: 시간이 메모리 접근 횟수로 결정된다. elementwise 연산(activation, dropout), reduction(sum, **softmax**, batch norm, layer norm) 등 대부분의 나머지

**Kernel fusion.** memory-bound 연산의 표준 가속법이다. 같은 입력에 여러 연산을 적용할 때 입력을 HBM에서 **한 번만** 읽는다. 다만 학습에서는 backward pass를 위해 중간값을 HBM에 저장해야 해서, 단순 kernel fusion의 효과가 제한된다. 이것이 recomputation이 필요한 이유다.

### 2.2 Standard Attention Implementation

입력 시퀀스 $$Q, K, V \in \mathbb{R}^{N \times d}$$ ($$N$$: 시퀀스 길이, $$d$$: head 차원)에 대해 attention 출력 $$O \in \mathbb{R}^{N \times d}$$ 를 계산한다.

$$
S = QK^\top \in \mathbb{R}^{N \times N}, \qquad
P = \mathrm{softmax}(S) \in \mathbb{R}^{N \times N}, \qquad
O = PV \in \mathbb{R}^{N \times d}
$$

- $$S$$: attention score 행렬, softmax는 **행 단위**로 적용된다
- $$P$$: softmax를 거친 attention 가중치 행렬
- 보통 $$N \gg d$$ 다 (GPT-2: $$N = 1024$$, $$d = 64$$)

표준 구현은 $$N \times N$$ 짜리 $$S$$ 와 $$P$$ 를 **HBM에 통째로 저장(materialize)** 한다. $$O(N^2)$$ 메모리다.

```text
Algorithm 0 Standard Attention Implementation

Require: Matrices Q, K, V ∈ ℝ^(N×d) in HBM.
1: Load Q, K by blocks from HBM, compute S = QKᵀ, write S to HBM.
2: Read S from HBM, compute P = softmax(S), write P to HBM.
3: Load P and V by blocks from HBM, compute O = PV, write O to HBM.
4: Return O.
```

*[Algorithm 0] Standard Attention Implementation*

softmax 등 상당수 연산이 memory-bound라, 이 많은 HBM 왕복이 그대로 느린 wall-clock time이 된다. $$S$$ 에 masking, $$P$$ 에 dropout 같은 elementwise 연산이 더해지면 문제는 더 심해진다. 3.2절에서 이 구현의 HBM 접근이 $$N$$ 에 대해 **제곱**임을 보인다.

## 3. FlashAttention: Algorithm, Analysis, and Extensions

큰 중간 행렬을 저장하지 않고, 더 적은 HBM 읽기/쓰기로 **exact attention**을 계산하는 방법이다. 설명은 forward pass 중심이고, backward는 논문 Appendix B에 있다.

### 3.1 An Efficient Attention Algorithm with Tiling and Recomputation

핵심 아이디어: $$Q, K, V$$ 를 블록으로 쪼개 느린 HBM에서 빠른 SRAM으로 올리고, **블록 단위로 attention을 계산하되 올바른 정규화 상수로 스케일을 보정해 가며 더해서** 마지막에 정확한 결과를 얻는다.

**Tiling: softmax의 블록 분해.** softmax는 $$K$$ 의 열들을 결합(couple)하므로 그냥 쪼갤 수 없다. 수치 안정성을 위한 safe softmax부터 보자. 벡터 $$x \in \mathbb{R}^B$$ 에 대해:

$$
m(x) := \max_i x_i, \qquad
f(x) := \left[ e^{x_1 - m(x)}, \dots, e^{x_B - m(x)} \right], \qquad
\ell(x) := \sum_i f(x)_i, \qquad
\mathrm{softmax}(x) := \frac{f(x)}{\ell(x)}
$$

- $$m(x)$$: 벡터의 **최댓값**. overflow 방지를 위해 지수에서 빼 준다
- $$f(x)$$: 최댓값을 뺀 뒤의 지수 벡터(분자), $$\ell(x)$$: 그 합(분모, 정규화 상수)

두 조각 $$x^{(1)}, x^{(2)} \in \mathbb{R}^B$$ 를 이어 붙인 $$x = [x^{(1)}\; x^{(2)}]$$ 의 softmax는 **조각별 통계로 재구성**할 수 있다.

$$
m(x) = \max\!\left( m(x^{(1)}),\, m(x^{(2)}) \right), \qquad
\ell(x) = e^{m(x^{(1)}) - m(x)}\,\ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)}\,\ell(x^{(2)})
$$

- 각 조각의 $$f, \ell$$ 은 자기 조각의 최댓값 기준으로 계산됐으므로, 전체 최댓값 $$m(x)$$ 기준으로 **$$e^{m(x^{(i)}) - m(x)}$$ 를 곱해 보정**한 뒤 합치면 전체와 정확히 같다

즉 **여분의 통계 $$(m(x), \ell(x))$$ 만 유지하면 softmax를 한 번에 한 블록씩** 계산할 수 있다. 이것이 tiling의 수학적 기반이다.

```text
Algorithm 1 FlashAttention

Require: Matrices Q, K, V ∈ ℝ^(N×d) in HBM, on-chip SRAM of size M.
 1: Set block sizes B_c = ⌈M/4d⌉, B_r = min(⌈M/4d⌉, d).
 2: Initialize O = (0)_{N×d} ∈ ℝ^(N×d), ℓ = (0)_N ∈ ℝ^N, m = (−∞)_N ∈ ℝ^N in HBM.
 3: Divide Q into T_r = ⌈N/B_r⌉ blocks Q_1, ..., Q_{T_r} of size B_r × d each,
    and divide K, V into T_c = ⌈N/B_c⌉ blocks K_1, ..., K_{T_c} and V_1, ..., V_{T_c},
    of size B_c × d each.
 4: Divide O into T_r blocks O_1, ..., O_{T_r} of size B_r × d each, divide ℓ into T_r
    blocks ℓ_1, ..., ℓ_{T_r} of size B_r each, divide m into T_r blocks m_1, ..., m_{T_r}
    of size B_r each.
 5: for 1 ≤ j ≤ T_c do
 6:     Load K_j, V_j from HBM to on-chip SRAM.
 7:     for 1 ≤ i ≤ T_r do
 8:         Load Q_i, O_i, ℓ_i, m_i from HBM to on-chip SRAM.
 9:         On chip, compute S_ij = Q_i K_jᵀ ∈ ℝ^(B_r×B_c).
10:         On chip, compute m̃_ij = rowmax(S_ij) ∈ ℝ^(B_r),
            P̃_ij = exp(S_ij − m̃_ij) ∈ ℝ^(B_r×B_c) (pointwise),
            ℓ̃_ij = rowsum(P̃_ij) ∈ ℝ^(B_r).
11:         On chip, compute m_i^new = max(m_i, m̃_ij) ∈ ℝ^(B_r),
            ℓ_i^new = e^(m_i − m_i^new) ℓ_i + e^(m̃_ij − m_i^new) ℓ̃_ij ∈ ℝ^(B_r).
12:         Write O_i ← diag(ℓ_i^new)^(−1) (diag(ℓ_i) e^(m_i − m_i^new) O_i
            + e^(m̃_ij − m_i^new) P̃_ij V_j) to HBM.
13:         Write ℓ_i ← ℓ_i^new, m_i ← m_i^new to HBM.
14:     end for
15: end for
16: Return O.
```

*[Algorithm 1] FlashAttention*

기호를 풀면: SRAM 크기 $$M$$ 에 맞춰 블록 크기 $$B_c$$(열)·$$B_r$$(행)을 정하고(1행), outer loop가 $$K_j, V_j$$ 블록을 SRAM에 올린 채(6행) inner loop가 $$Q_i$$ 블록을 순회한다(8행). 블록 score $$S_{ij}$$ 와 블록 통계 $$\tilde{m}_{ij}, \tilde{P}_{ij}, \tilde{\ell}_{ij}$$ 를 **전부 SRAM 안에서** 계산하고(9~10행), 지금까지의 통계 $$m_i, \ell_i$$ 와 위 분해 공식으로 합치며(11행), 출력 $$O_i$$ 를 새 정규화 상수로 **재스케일**해 갱신한다(12행). $$N \times N$$ 행렬은 어디에도 저장되지 않고, HBM에는 $$O$$ 와 크기 $$N$$ 짜리 통계 $$\ell, m$$ 만 오간다.

**Recomputation: backward를 위한 선택적 재계산.** backward pass는 원래 $$S, P \in \mathbb{R}^{N \times N}$$ 이 필요하다. FlashAttention은 이를 저장하는 대신 **출력 $$O$$ 와 softmax 정규화 통계 $$(m, \ell)$$ 만 저장**하고, backward에서 SRAM 위의 $$Q, K, V$$ 블록으로부터 $$S, P$$ 를 다시 계산한다. gradient checkpointing의 선택적 형태로 볼 수 있는데, 보통의 checkpointing이 속도를 희생해 메모리를 얻는 것과 달리, 여기서는 **FLOPs가 늘어나도 HBM 접근 감소 덕에 backward까지 더 빨라진다**.

**Kernel fusion.** tiling 덕분에 전체를 **하나의 CUDA kernel**로 구현한다. HBM에서 입력 로드 → 행렬 곱, softmax, (선택) masking·dropout, 행렬 곱까지 전부 수행 → 결과만 HBM에 쓰기 순서다. 입출력의 반복적인 HBM 왕복이 사라진다.

정확성과 비용은 다음과 같이 보장된다.

> **Theorem 1.** Algorithm 1은 $$O = \mathrm{softmax}(QK^\top)V$$ 를 $$O(N^2 d)$$ FLOPs로 반환하며, 입력과 출력 외에 $$O(N)$$ 의 추가 메모리만 필요하다.

### 3.2 Analysis: IO Complexity of FlashAttention

![표준 attention과 FlashAttention의 GFLOPs·HBM 접근·runtime 비교표, block size에 따른 HBM 접근과 forward 시간, sparsity에 따른 block-sparse 속도 향상](/assets/images/flashattention-fig2.png)
*[Figure 2] GPT-2 attention의 FLOPs·HBM 접근·runtime 비교(왼쪽), block size의 효과(가운데), sparsity에 비례하는 block-sparse FlashAttention의 속도(오른쪽)*

> **Theorem 2.** $$N$$ 을 시퀀스 길이, $$d$$ 를 head 차원, $$M$$ 을 SRAM 크기($$d \le M \le Nd$$)라 하자. 표준 attention(Algorithm 0)은 $$\Theta(Nd + N^2)$$ 번의 HBM 접근이 필요하고, FlashAttention(Algorithm 1)은 $$\Theta(N^2 d^2 M^{-1})$$ 번이면 된다.

전형적인 값($$d$$ 는 64~128, $$M$$ 은 약 100KB)에서는 $$d^2$$ 이 $$M$$ 보다 몇 배나 작으므로, FlashAttention의 HBM 접근은 표준 구현보다 **몇 배나 적다**. 증명의 핵심은 이렇다. 크기 $$\Theta(M)$$ 의 $$K, V$$ 블록마다 $$Q$$ 전체를 한 번씩 훑으므로 $$\Theta(NdM^{-1})$$ 번의 패스가 생기고, 패스당 $$\Theta(Nd)$$ 개 원소를 읽어 총 $$\Theta(N^2 d^2 M^{-1})$$ 이 된다.

> **Proposition 3.** $$d \le M \le Nd$$ 인 모든 $$M$$ 에 대해, exact attention을 $$o(N^2 d^2 M^{-1})$$ 번의 HBM 접근으로 계산하는 알고리즘은 존재하지 않는다.

즉 FlashAttention의 IO 복잡도는 **점근적으로 최적**이다. Figure 2가 이 분석을 실증한다. 왼쪽을 보면 FLOPs는 오히려 많은데(75.2 vs 66.6 GFLOPs, recomputation 때문) HBM 접근이 9배 적어(4.4 vs 40.3 GB) runtime이 5.7배 빠르다(7.3 vs 41.7 ms). 가운데를 보면 block size를 키우면 HBM 접근이 줄어 forward가 빨라지다가, 256을 넘으면 다른 요인(산술 연산)이 병목이 된다.

### 3.3 Extension: Block-Sparse FlashAttention

FlashAttention을 approximate attention으로 확장한다. 입력 $$Q, K, V$$ 와 마스크 $$\tilde{M} \in \{0,1\}^{N \times N}$$ 에 대해

$$
S = QK^\top, \qquad
P = \mathrm{softmax}\!\left(S \odot \mathbb{1}_{\tilde{M}}\right), \qquad
O = PV
$$

- $$\mathbb{1}_{\tilde{M}}$$: $$\tilde{M}_{kl} = 1$$ 이면 $$S_{kl}$$ 을 그대로, 0이면 $$-\infty$$ 로 바꾸는 마스킹. softmax에서 해당 위치가 0이 된다
- $$\tilde{M}$$ 은 블록 구조를 가진다고 가정한다. 즉 블록 단위 마스크 $$M \in \{0,1\}^{N/B_r \times N/B_c}$$ 로 표현된다

알고리즘은 Algorithm 1과 동일하되 **0인 블록을 건너뛰기만** 하면 된다.

> **Proposition 4.** block-sparse FlashAttention의 HBM 접근은 $$\Theta(Nd + N^2 d^2 M^{-1} s)$$ 다. $$s$$ 는 블록 마스크에서 0이 아닌 블록의 비율이다.

sparsity가 IO 복잡도의 큰 항을 **정비례로** 깎는다. 큰 $$N$$ 에서 $$s$$ 를 $$N^{-1/2}$$ 또는 $$N^{-1}\log N$$ 으로 두면 IO 복잡도가 $$\Theta(N\sqrt{N})$$, $$\Theta(N \log N)$$ 이 된다. 실험에서는 임의의 sparsity를 근사할 수 있다고 알려진 고정 butterfly sparsity 패턴을 쓴다.

## 4. Experiments

학습 속도, 모델 품질, attention 벤치마크 세 가지를 검증한다.

### 4.1 Faster Models with FlashAttention

**BERT.** MLPerf 1.1 학습 속도 기록(Nvidia)과 같은 초기화에서 목표 정확도(masked LM 72.0%)까지의 시간을 비교한다. **15% 더 빠른, 당시 가장 빠른 single-node BERT 학습**이다.

| BERT Implementation | Training time (minutes) |
| --- | --- |
| Nvidia MLPerf 1.1 | 20.0 ± 1.5 |
| FlashAttention (ours) | 17.4 ± 1.4 |

*[Table 1] MLPerf benchmark 초기화에서 목표 정확도 72.0%까지의 BERT-large 학습 시간 (8×A100, 10회 평균)*

**GPT-2.** OpenWebText에서 HuggingFace·Megatron-LM 구현과 비교한다. 모델 정의를 바꾸지 않으므로 **perplexity는 동일**하고, 학습/검증 곡선도 baseline과 같다. 수치적으로 안정적이라는 확인이다.

| Model implementations | OpenWebText (ppl) | Training time (speedup) |
| --- | --- | --- |
| GPT-2 small - Huggingface | 18.2 | 9.5 days (1.0×) |
| GPT-2 small - Megatron-LM | 18.2 | 4.7 days (2.0×) |
| GPT-2 small - FlashAttention | 18.2 | 2.7 days (3.5×) |
| GPT-2 medium - Huggingface | 14.2 | 21.0 days (1.0×) |
| GPT-2 medium - Megatron-LM | 14.3 | 11.5 days (1.8×) |
| GPT-2 medium - FlashAttention | 14.3 | 6.9 days (3.0×) |

*[Table 2] GPT-2 small·medium의 학습 시간(HuggingFace 대비 최대 3.5×, Megatron-LM 대비 최대 1.7×, 8×A100)*

**Long-range Arena.** 시퀀스 길이 1024~4096의 LRA benchmark에서 표준 attention 대비 **2.4× 가속**, block-sparse FlashAttention은 테스트한 **모든 approximate attention보다 빠르다**.

| Models | ListOps | Text | Retrieval | Image | Pathfinder | Avg | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 36.0 | 63.6 | 81.6 | 42.3 | 72.7 | 59.3 | - |
| FlashAttention | 37.6 | 63.9 | 81.4 | 43.5 | 72.7 | 59.8 | 2.4× |
| Block-sparse FlashAttention | 37.0 | 63.0 | 81.3 | 43.6 | 73.3 | 59.6 | 2.8× |
| Linformer | 35.6 | 55.9 | 77.7 | 37.8 | 67.6 | 54.9 | 2.5× |
| Linear Attention | 38.8 | 63.2 | 80.7 | 42.6 | 72.5 | 59.6 | 2.3× |
| Performer | 36.8 | 63.6 | 82.2 | 42.1 | 69.9 | 58.9 | 1.8× |
| Local Attention | 36.1 | 60.2 | 76.7 | 40.6 | 66.6 | 56.0 | 1.7× |
| Reformer | 36.5 | 63.8 | 78.5 | 39.6 | 69.4 | 57.6 | 1.3× |
| Smyrf | 36.1 | 64.1 | 79.0 | 39.6 | 70.5 | 57.9 | 1.7× |

*[Table 3] Long-Range-Arena benchmark에서의 정확도와 속도 비교*

### 4.2 Better Models with Longer Sequences

**긴 context 언어 모델링.** GPT-2의 context를 4배 늘려도 Megatron-LM(1K)보다 빠르다. **4K context로 30% 더 빠르면서 perplexity 0.7 개선**이다.

| Model implementations | Context length | OpenWebText (ppl) | Training time (speedup) |
| --- | --- | --- | --- |
| GPT-2 small - Megatron-LM | 1k | 18.2 | 4.7 days (1.0×) |
| GPT-2 small - FlashAttention | 1k | 18.2 | 2.7 days (1.7×) |
| GPT-2 small - FlashAttention | 2k | 17.6 | 3.0 days (1.6×) |
| GPT-2 small - FlashAttention | 4k | 17.5 | 3.6 days (1.3×) |

*[Table 4] context 길이를 4배 키운 GPT-2 small의 perplexity와 학습 시간 (8×A100)*

**장문서 분류.** 중환자실 퇴원 요약(MIMIC-III)과 유럽인권재판소 판례(ECtHR)처럼 평균 2천 토큰이 넘는 문서에서, 사전학습 RoBERTa의 시퀀스 길이를 늘리며 micro $$F_1$$ 을 잰다. MIMIC은 16K가 512보다 4.3점, ECtHR은 8K가 512보다 8.5점 높다.

| | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-III | 52.8 | 50.7 | 51.7 | 54.6 | 56.4 | **57.1** |
| ECtHR | 72.2 | 74.3 | 77.1 | 78.6 | **80.7** | 79.2 |

*[Table 5] 시퀀스 길이에 따른 장문서 분류 성능 (micro F1)*

**Path-X와 Path-256.** 128×128(또는 256×256) 흑백 이미지의 두 점이 경로로 이어져 있는지를 픽셀 시퀀스(16K/64K)로 분류하는 초장문 과제다. 이전의 모든 Transformer는 메모리 부족이거나 우연 수준이었다.

| Model | Path-X | Path-256 |
| --- | --- | --- |
| Transformer | ✗ | ✗ |
| Linformer | ✗ | ✗ |
| Linear Attention | ✗ | ✗ |
| Performer | ✗ | ✗ |
| Local Attention | ✗ | ✗ |
| Reformer | ✗ | ✗ |
| SMYRF | ✗ | ✗ |
| FlashAttention | **61.4** | ✗ |
| Block-sparse FlashAttention | 56.0 | **63.1** |

*[Table 6] Path-X(길이 16K)·Path-256(길이 64K)에서 우연 수준을 넘긴 최초의 Transformer 결과*

### 4.3 Benchmarking Attention

A100(40GB HBM) 한 장에서 시퀀스 길이를 바꿔 가며 runtime과 메모리를 잰다.

![시퀀스 길이 128~64K에 걸친 attention runtime과 메모리 사용량. FlashAttention은 표준 대비 최대 3배 빠르고, 메모리는 선형으로 최대 20배 절약](/assets/images/flashattention-fig3.png)
*[Figure 3] 시퀀스 길이에 따른 attention runtime(왼쪽)과 메모리 사용량(오른쪽)*

- **Runtime**: FlashAttention은 PyTorch 구현 대비 **최대 3× 빠르다**. 짧은 시퀀스에서는 메모리 접근이 적어 approximate attention들보다도 빠르고, 512~1024 사이에서 교차한다. **block-sparse FlashAttention은 모든 길이에서, 알려진 모든 exact·sparse·approximate 구현보다 빠르다**
- **Memory**: 메모리 사용량이 시퀀스 길이에 **선형**으로, exact baseline 대비 **최대 20× 절약**된다. 64K에서 Linformer 외의 모든 baseline이 OOM이 나고, FlashAttention은 Linformer보다도 2× 효율적이다

## 5. Limitations and Future Directions

- **CUDA 컴파일의 필요.** 지금은 새로운 attention 변형마다 CUDA kernel을 새로 짜야 한다. PyTorch보다 훨씬 저수준의 언어라 상당한 엔지니어링이 들고 GPU 아키텍처 간 이식도 안 될 수 있다. 고수준 언어로 쓴 attention을 IO-aware CUDA 구현으로 컴파일하는 방법(이미지 처리의 Halide 같은)이 필요하다
- **IO-aware 딥러닝.** attention이 Transformer에서 가장 메모리 집약적인 연산이지만, 깊은 네트워크의 **모든 layer가 HBM을 만진다**. IO-aware 접근이 다른 모듈로 확장되기를 기대한다
- **Multi-GPU IO-aware 방법.** 단일 GPU에서는 상수 배 내에서 최적이지만, attention 계산은 여러 GPU에 걸쳐 병렬화될 수 있다. GPU 간 데이터 전송까지 고려하는 한 층 더 높은 IO 분석이 미래 과제다
