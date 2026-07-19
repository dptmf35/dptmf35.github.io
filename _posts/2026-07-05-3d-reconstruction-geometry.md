---
title: "[공부] 3D Reconstruction ① 기초 기하학: camera calibration과 다중 시점 기하학"
excerpt: "카메라 투영 모델·calibration·epipolar geometry·RANSAC·triangulation까지, 두 시점 3D 복원의 기하학"
date: 2026-07-05 09:00:00 +0900
categories:
  - 공부
  - CV
tags:
  - 3D Reconstruction
  - camera calibration
  - epipolar geometry
  - feature matching
  - RANSAC
  - triangulation
toc: true
toc_sticky: true
---

> [학습 로드맵]({% post_url 2026-07-05-3d-reconstruction-roadmap %})의 첫 축. 3D를 복원하려면 먼저 **카메라가 세계를 2D로 어떻게 투영하는지**를 수식으로 알아야 한다.
> 이 글은 Hartley & Zisserman, [*Multiple View Geometry in Computer Vision* (2nd ed.)](https://www.robots.ox.ac.uk/~vgg/hzbook/)의 흐름을 따라 정리한다.

## 1. 카메라 투영 모델

### 1.1 핀홀 모델과 동차 좌표

핀홀 카메라는 3D 점을 이미지의 픽셀로 투영한다.

$$
\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= K\,[R \mid t]
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

기호를 하나씩 보면:

- $$(X, Y, Z)$$: **세계 좌표계**의 3D 점. 끝에 1을 붙인 4-vector가 동차 좌표(homogeneous coordinates). 투영·회전·평행이동을 전부 행렬곱 하나로 쓰기 위한 표현
- $$(u, v)$$: 이미지의 **픽셀 좌표**. 역시 끝에 1을 붙인 동차 3-vector
- $$\lambda$$: 점의 **깊이에 비례하는 스케일**. 동차 좌표는 스케일 배를 같은 점으로 보므로 투영 과정에서 자연히 나온다
- $$K$$: 내부 파라미터(intrinsics), $$3 \times 3$$
- $$[R \mid t]$$: 외부 파라미터(extrinsics), $$3 \times 4$$

한 장의 이미지에서는 $$\lambda$$(깊이)를 알 수 없다. 그래서 **픽셀 하나는 3D 점 하나가 아니라 광선(ray) 하나**에 대응한다. 이것이 단일 시점의 근본 한계이고, 두 시점이 필요한 이유다.

전체 곱 $$P = K[R \mid t]$$ 를 **투영 행렬(projection matrix)** 이라 한다. $$3 \times 4$$ 행렬로 원소는 12개지만 전체 스케일이 무의미하므로 자유도는 **11**이다.

### 1.2 내부 파라미터 $$K$$

$$
K = \begin{bmatrix}
f_x & \gamma & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

- $$f_x = f\,m_x$$, $$f_y = f\,m_y$$: **픽셀 단위 초점거리**. 물리 초점거리 $$f$$(mm)에 센서의 픽셀 밀도 $$m_x, m_y$$(pixel/mm)를 곱한 값. 픽셀이 정사각형이면 $$f_x \approx f_y$$
- $$(c_x, c_y)$$: **주점(principal point)**. 광축이 이미지 평면을 뚫는 픽셀 위치. 대략 이미지 중심이지만 정확히는 아니다
- $$\gamma$$: **skew**. 픽셀 축이 직각에서 벗어난 정도. 현대 센서에서는 0으로 둔다

총 5 DoF. 렌즈·센서 고유값이라 **카메라마다 한 번만 구하면 된다.**

### 1.3 외부 파라미터 $$[R \mid t]$$

- $$R \in SO(3)$$: 세계 좌표계 → 카메라 좌표계의 **회전** ($$3\times3$$ 회전행렬, 3 DoF)
- $$t \in \mathbb{R}^3$$: 같은 변환의 **평행이동** (3 DoF)
- 카메라 중심(투영의 원점) $$C$$ 는 $$RC + t = 0$$ 을 만족하는 점, 즉 $$C = -R^\top t$$

세계 좌표계를 어디에 두느냐에 따라 달라지므로 **장면·설치마다 다시 정해진다.**

### 1.4 렌즈 왜곡 (distortion)

핀홀은 이상적 모델이고, 실제 렌즈는 직선을 휘게 한다. 대표적으로 **radial distortion**:

$$
x_d = x_u \left( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \right)
$$

- $$x_u$$: 왜곡이 없다고 가정한 **정규화 좌표**($$K$$ 를 적용하기 전, 주점 기준)
- $$x_d$$: 실제 관측되는 **왜곡된 좌표**
- $$r$$: 주점에서의 거리, $$r^2 = x_u^2 + y_u^2$$. 왜곡이 **중심에서 멀수록 커지는** 것을 담는다
- $$k_1, k_2, k_3$$: radial 계수. 부호에 따라 barrel(볼록)/pincushion(오목) 왜곡

렌즈와 센서가 완전히 평행하지 않아 생기는 **tangential distortion**은 계수 $$p_1, p_2$$ 로 모델링한다. 왜곡 계수는 calibration에서 $$K$$ 와 함께 추정하고, 이미지를 **undistort** 한 뒤에는 순수 핀홀로 취급할 수 있다.

## 2. camera calibration

calibration은 $$K$$, 왜곡 계수, $$[R \mid t]$$ 를 **추정**하는 과정이다.

### 2.1 DLT: 3D 좌표를 아는 대상으로 $$P$$ 구하기

3D 위치를 정확히 아는 점 $$X_i$$ 와 그 이미지 관측 $$x_i$$ 의 쌍이 있으면 $$P$$ 를 직접 풀 수 있다. $$\lambda x_i = P X_i$$ 에서 미지의 $$\lambda$$ 를 없애기 위해 양변에 cross product를 취한다.

$$
x_i \times (P X_i) = 0
$$

- cross product가 0 = 두 벡터가 **평행** = 관측 방향과 투영 방향이 일치한다는 뜻. $$\lambda$$ 가 소거된다
- 한 쌍이 방정식 3개를 주지만 독립은 **2개**(cross product의 세 성분 중 하나는 종속)
- $$P$$ 는 11 DoF → 최소 **5.5쌍**, 실무에선 6쌍 이상
- $$P$$ 의 12개 원소를 벡터 $$p$$ 로 펴면 $$A p = 0$$ 꼴의 선형 문제가 되고, $$A$$ 의 SVD에서 **최소 특이값에 대응하는 특이벡터**가 해다. 이 방식을 **DLT**(Direct Linear Transform)라 한다
- 얻은 $$P$$ 는 RQ decomposition으로 $$K[R \mid t]$$ 로 분해한다

한계는 명확하다. **3D 좌표를 정밀하게 아는 입체 지그**가 필요하다.

### 2.2 Zhang's method: 평면 패턴 여러 장

실무 표준은 **체커보드 사진 여러 장**으로 끝내는 Zhang's method다. 핵심 아이디어: 패턴이 평면이라 $$Z = 0$$ 으로 두면 투영이 $$3\times3$$ **homography** $$H$$ 로 단순해진다.

$$
\lambda x = H \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix},
\qquad H = K \, [\, r_1 \;\; r_2 \;\; t \,]
$$

- $$(X, Y)$$: 체커보드 평면 위의 코너 좌표(칸 크기로 이미 안다)
- $$r_1, r_2$$: 회전행렬 $$R$$ 의 첫 두 열. $$Z=0$$ 이라 셋째 열 $$r_3$$ 항이 사라진 것
- $$H$$: 사진마다 코너 검출 → DLT로 추정(4점 이상)

여기서 $$R$$ 이 회전행렬이라는 사실이 제약을 준다: 열끼리 직교($$r_1^\top r_2 = 0$$)하고 길이가 같다($$\lVert r_1 \rVert = \lVert r_2 \rVert$$). 이 두 조건을 $$H$$ 와 $$K$$ 로 다시 쓰면, **사진 한 장당** $$\omega = K^{-\top} K^{-1}$$ 에 대한 **선형 제약 2개**가 나온다.

- $$\omega$$: 대칭 $$3\times3$$ 행렬(image of the absolute conic). 스케일 제외 미지수 5개
- 사진 **3장 이상**이면 제약이 6개 이상 → $$\omega$$ 를 선형으로 풀고, Cholesky 분해로 $$K$$ 를 복원한다($$\gamma = 0$$ 을 가정하면 2장도 가능)
- 이후 각 사진의 $$[R \mid t]$$ 를 계산하고, 왜곡 계수까지 포함해 전체를 비선형으로 정련한다

### 2.3 재투영 오차: calibration 품질의 척도

정련 단계가 최소화하는 것은 **재투영 오차(reprojection error)** 다.

$$
E = \sum_{i,j} \left\lVert \, x_{ij} - \hat{x}\!\left(K, d, R_i, t_i, X_j\right) \right\rVert^2
$$

- $$x_{ij}$$: $$i$$ 번째 사진에서 **관측된** $$j$$ 번째 코너의 픽셀 위치
- $$\hat{x}(\cdot)$$: 현재 추정한 파라미터로 그 코너를 **투영해 본 예측** 위치. $$d$$ 는 왜곡 계수 묶음
- 즉 "지금 파라미터로 세계를 다시 찍어 보면 관측과 몇 픽셀 어긋나는가"

Levenberg–Marquardt로 최소화하며, 최종 **RMS 재투영 오차(픽셀)** 가 calibration 품질 지표다. 잘 된 calibration은 sub-pixel(예: 0.2~0.5 px)이 나온다.

## 3. 다중 시점 기하학: epipolar geometry

한 장으로는 깊이가 모호하다(1.1의 $$\lambda$$). **두 시점**을 쓰면 깊이를 복원할 수 있고, 이때 두 이미지의 대응을 제약하는 것이 **epipolar geometry**다.

![두 카메라 중심과 3D 점이 이루는 평면이 각 이미지에 epipolar 선을 만들고, 대응점은 그 선 위에 놓인다](/assets/images/3drecon-epipolar.svg)

### 3.1 구성 요소

- **baseline**: 두 카메라 중심 $$C, C'$$ 을 잇는 선분
- **epipole** $$e, e'$$: baseline이 각 이미지 평면을 뚫는 점. 상대 카메라가 내 이미지에 맺히는 위치
- **epipolar plane**: $$C, C'$$ 과 3D 점 $$X$$ 가 만드는 평면
- **epipolar line** $$l'$$: epipolar plane과 이미지 평면의 교선. 모든 epipolar line은 epipole을 지난다

핵심 결론: 좌 이미지의 점 $$x$$ 에 대응하는 우 이미지의 점 $$x'$$ 는 **epipolar line 위에만 존재**한다. 대응 탐색이 2D 전체에서 **1D 선 위**로 줄어든다.

### 3.2 Essential matrix $$E$$

$$K$$ 를 아는 경우, 픽셀 좌표에서 $$K$$ 를 벗겨 낸 **정규화 좌표** $$\hat{x} = K^{-1} x$$ (카메라 중심에서 나가는 ray의 방향)를 쓴다. 이때 epipolar 제약은:

$$
\hat{x}'^\top E \,\hat{x} = 0,
\qquad E = [t]_\times R
$$

- $$R, t$$: 첫째 카메라 좌표계에서 둘째 카메라로의 **상대 회전·평행이동**
- $$[t]_\times$$: "$$t$$ 와의 cross product"를 행렬곱으로 쓴 skew-symmetric 행렬:

$$
[t]_\times = \begin{bmatrix}
0 & -t_z & t_y \\
t_z & 0 & -t_x \\
-t_y & t_x & 0
\end{bmatrix}
$$

유도는 한 줄이다. 첫 카메라의 ray $$R\hat{x}$$(둘째 좌표계로 회전해 놓은 것), baseline $$t$$, 둘째 카메라의 ray $$\hat{x}'$$ 세 벡터는 모두 **한 epipolar plane 위**에 있다. 세 벡터가 공면이면 scalar triple product가 0이다:

$$
\hat{x}' \cdot \left( t \times R\hat{x} \right) = 0
\;\;\Longrightarrow\;\;
\hat{x}'^\top [t]_\times R \,\hat{x} = 0
$$

$$E$$ 의 자유도는 **5**(회전 3 + 평행이동 방향 2)다. $$t$$ 의 **크기는 알 수 없다**(scale ambiguity): 장면 전체를 2배 키우고 카메라를 2배 멀리 놓아도 같은 이미지가 나온다.

### 3.3 Fundamental matrix $$F$$

$$K$$ 를 모르면 픽셀 좌표를 그대로 쓰는 **fundamental matrix** $$F$$ 로 같은 제약을 표현한다.

$$
x'^\top F x = 0,
\qquad F = K'^{-\top} E \, K^{-1}
$$

- $$x, x'$$: 두 이미지의 **픽셀 동차 좌표**. $$K', K$$ 는 각 카메라의 intrinsics
- $$l' = Fx$$: 좌측 점 $$x$$ 가 정하는 **우측의 epipolar line** (직선 $$l'$$ 은 $$l'^\top x' = 0$$ 을 만족하는 3-vector로 표현)
- $$F e = 0$$: epipole $$e$$ 는 $$F$$ 의 null vector이고, 그래서 $$F$$ 는 **rank 2**
- 자유도 **7**: 원소 9 − 스케일 1 − $$\det F = 0$$ 제약 1

| | Essential $$E$$ | Fundamental $$F$$ |
| --- | --- | --- |
| 좌표 | 정규화 좌표($$K$$ 필요) | 픽셀 좌표($$K$$ 불필요) |
| 자유도 | 5 | 7 |
| 최소 대응 쌍 | 5 (five-point) | 7~8 (eight-point) |
| 분해하면 | 상대 자세 $$(R, t)$$ | (calibration 없이는 자세 불가) |

### 3.4 8-point algorithm과 정규화

$$F$$ 추정의 기본은 **8-point algorithm**이다.

1. 대응 쌍 8개 이상에서 $$x'^\top F x = 0$$ 을 $$F$$ 의 9개 원소에 대한 **선형식**으로 편다 → $$A f = 0$$ ($$f$$ 는 $$F$$ 를 편 9-vector)
2. SVD의 최소 특이값 벡터로 $$f$$ 를 푼다
3. **rank 2 강제**: 얻은 $$F$$ 를 다시 SVD 해 가장 작은 특이값을 0으로 만든다

여기서 실무적으로 결정적인 것이 **정규화(Hartley normalization)** 다. 픽셀 좌표(수백~수천)를 그대로 쓰면 $$A$$ 의 조건수가 나빠 해가 크게 흔들린다. 각 이미지에서 점들의 **centroid를 원점으로 평행이동**하고 **평균 거리가 $$\sqrt{2}$$** 가 되게 스케일한 뒤 계산하고, 마지막에 변환을 되돌린다. 이 한 단계로 정확도가 극적으로 좋아진다.

$$K$$ 를 아는 경우에는 $$E$$ 를 최소 5쌍으로 푸는 **five-point algorithm**(Nistér)이 표준이다. 표본이 작을수록 RANSAC(5장)에서 압도적으로 유리하다.

### 3.5 $$E$$ 분해 → 상대 자세 $$(R, t)$$

$$E$$ 를 SVD 하면 $$E = U \operatorname{diag}(1,1,0)\, V^\top$$ 꼴이 되고, 여기서 상대 자세 후보가 나온다.

$$
R \in \{\, U W V^\top,\; U W^\top V^\top \,\},
\qquad t = \pm u_3,
\qquad W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

- $$u_3$$: $$U$$ 의 셋째 열($$E$$ 의 left null vector 방향 = baseline 방향)
- $$W$$: 90° 회전을 담는 고정 행렬

조합이 $$2 \times 2 = 4$$ 개 나오는데, 물리적으로 맞는 것은 하나다. 대응점 하나를 각 후보로 triangulation(6장)해 보고 **두 카메라 모두의 앞($$Z > 0$$)에 놓이는 조합**을 고른다. 이를 cheirality check라 한다. $$t$$ 는 **방향만** 정해지므로, 실제 거리 스케일은 알려진 물체 크기·IMU 같은 외부 정보로 정한다.

## 4. 특징점과 descriptor

epipolar geometry를 쓰려면 실제 대응 $$x \leftrightarrow x'$$ 가 필요하다. 대응은 **검출 → descriptor → 매칭**의 세 단계로 찾는다.

### 4.1 검출 (detector): 어디를 볼 것인가

아무 픽셀이나 매칭할 수는 없다. **어느 방향으로 움직여도 밝기가 변하는 점(corner)** 이 구별하기 좋다. Harris detector는 이를 structure tensor로 잰다.

$$
M = \sum_{(x,y) \in W} w(x,y)
\begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}
$$

- $$I_x, I_y$$: 이미지의 $$x, y$$ 방향 **gradient**(밝기 변화율)
- $$W$$: 점 주변의 작은 윈도, $$w(x,y)$$: 가우시안 가중치
- $$M$$ 의 고유값 $$\lambda_1, \lambda_2$$ 가 **둘 다 크면** corner(모든 방향으로 변화), 하나만 크면 edge, 둘 다 작으면 평탄한 영역
- 고유값을 직접 구하지 않고 response $$R = \det M - k(\operatorname{tr} M)^2$$ ($$k \approx 0.04\!\sim\!0.06$$)로 판정한다

다른 대표 검출기:

- **FAST**: 중심 픽셀 주위 원(16픽셀) 위에 충분히 밝거나 어두운 픽셀이 $$n$$ 개 연속이면 corner. 비교 연산뿐이라 매우 빠르다. 실시간 SLAM의 표준
- **DoG (SIFT의 검출기)**: 서로 다른 크기의 Gaussian blur 차(Difference of Gaussians)를 스케일 축으로 쌓아 3D 극값을 찾는다 → 멀리서 찍든 가까이서 찍든 같은 점이 잡히는 **scale invariance**

### 4.2 descriptor: 그 점을 어떻게 요약할 것인가

검출한 점 주변 패치를 **비교 가능한 벡터**로 요약한 것이 descriptor다.

| | SIFT | ORB |
| --- | --- | --- |
| 형식 | 128차원 float | 256-bit binary |
| 구성 | $$4\times4$$ 셀 × 8방향 gradient histogram | 픽셀 쌍의 밝기 대소 비교(BRIEF) |
| 거리 | L2 | Hamming (XOR, 매우 빠름) |
| 불변성 | scale + rotation + 조명 일부 | rotation(oriented FAST), scale은 피라미드로 근사 |
| 속도 | 느림 | 매우 빠름 (실시간) |

- SIFT가 rotation에 불변인 이유: 패치의 **주 gradient 방향을 먼저 구해 그 방향으로 정렬**한 뒤 histogram을 만들기 때문
- 최근에는 학습 기반 descriptor(SuperPoint 등)가 어려운 조명·시점 변화에서 더 강하다

### 4.3 매칭

1. **최근접 탐색**: 한 이미지의 descriptor마다 다른 이미지에서 거리가 가장 가까운 것을 찾는다(brute-force, 또는 kd-tree/LSH)
2. **Lowe's ratio test**: 최근접 거리 $$d_1$$ 과 두 번째로 가까운 거리 $$d_2$$ 의 비가 $$d_1 / d_2 < 0.7\!\sim\!0.8$$ 일 때만 채택한다. 1등과 2등이 비슷하다는 것은 **애매한 매칭**(반복 패턴, 텍스처 없는 영역)이라는 뜻이라 버린다
3. **cross-check**: A→B 최근접과 B→A 최근접이 서로 일치할 때만 채택

이렇게 걸러도 잘못된 매칭(outlier)이 수십 %씩 남는다. 그대로 $$F$$ 를 풀면 오염된다. RANSAC이 필요한 이유다.

## 5. RANSAC: outlier 속에서 모델 찾기

최소제곱은 **outlier 하나에도 전체가 끌려간다**. RANSAC(RANdom SAmple Consensus)은 발상을 뒤집는다: 전부 쓰지 말고, **최소한의 표본으로 모델을 세운 뒤 얼마나 많은 점이 동의하는지** 세자.

![무작위 최소 표본으로 모델을 세우고, 임계값 안의 inlier를 세고, 가장 많은 inlier를 얻은 모델을 채택하는 RANSAC 3단계](/assets/images/3drecon-ransac.svg)

$$F$$ 추정 기준 절차:

1. 대응 쌍에서 무작위로 **최소 표본** $$s$$ 쌍을 뽑는다 ($$F$$: 8, $$E$$: 5, homography: 4)
2. 그 표본만으로 모델을 계산한다
3. **모든** 대응에 대해 오차 $$< \tau$$ 인 것(inlier)의 수를 센다
4. $$N$$ 번 반복해 inlier가 가장 많은 모델을 채택하고, 그 **inlier 전체로 모델을 재추정**한다

반복 횟수 $$N$$ 은 확률로 정한다.

$$
N = \frac{\log(1 - p)}{\log(1 - w^s)}
$$

- $$p$$: 적어도 한 번은 **outlier가 하나도 없는 표본**을 뽑을 확률(보통 0.99로 둔다)
- $$w$$: 전체 대응 중 inlier의 비율
- $$w^s$$: 표본 $$s$$ 개가 **모두** inlier일 확률. 그래서 $$1 - w^s$$ 는 한 번의 시도가 실패할 확률, $$(1-w^s)^N$$ 은 $$N$$ 번 모두 실패할 확률이고, 이것이 $$1-p$$ 이하가 되도록 푼 것
- 예: $$w = 0.5$$ 일 때 $$s = 8$$ 이면 $$N \approx 1177$$, $$s = 5$$ 면 $$N \approx 145$$. **표본이 작을수록 기하급수적으로 유리**하다. five-point algorithm이 선호되는 실질적 이유

inlier 판정 오차로는 점과 epipolar line 사이 거리, 또는 그 1차 보정판인 **Sampson distance**를 쓴다.

$$
d_{\text{Sampson}}
= \frac{\left( x'^\top F x \right)^2}
{(Fx)_1^2 + (Fx)_2^2 + (F^\top x')_1^2 + (F^\top x')_2^2}
$$

- 분자 $$x'^\top F x$$: epipolar 제약의 **위반량**(0이면 완벽한 대응)
- $$(Fx)_i$$: 벡터 $$Fx$$ 의 $$i$$ 번째 성분. 분모는 위반량을 **픽셀 단위 기하 거리로 정규화**하는 항. 재투영 오차의 1차 근사가 된다

## 6. Triangulation: 대응에서 3D 점으로

상대 자세 $$(R, t)$$ 와 inlier 대응이 준비되면 드디어 3D 점을 복원한다.

![두 카메라에서 대응점을 지나는 ray를 쏘면 노이즈 때문에 정확히 만나지 않고, 재투영 오차를 최소화하는 점을 찾는다](/assets/images/3drecon-triangulation.svg)

### 6.1 왜 "교점"이 아닌가

이상적으로는 두 카메라에서 대응점을 지나 쏜 ray가 3D의 한 점에서 만난다. 그러나 검출·매칭에 픽셀 노이즈가 있으므로 실제 두 ray는 **비껴간다(skew)**. 교점이 존재하지 않는다. 그래서 "가장 그럴듯한" 점 $$\hat{X}$$ 를 정의해야 한다.

### 6.2 선형 방법 (DLT triangulation)

각 시점에서 $$\lambda x = P X$$ 의 $$\lambda$$ 를 2.1과 같은 cross product 트릭으로 소거한다.

$$
x \times (P X) = 0
$$

- 시점당 독립 방정식 **2개** → 두 시점이면 $$4 \times 4$$ 행렬 $$A$$ 로 $$A X = 0$$
- $$X$$: 구하려는 3D 점의 동차 4-vector. SVD의 최소 특이값 벡터가 해
- 빠르고 시점 수 제한이 없지만, 최소화하는 양이 대수적 오차(algebraic error)라 **기하적 의미가 없다**

### 6.3 최적 방법: 재투영 오차 최소화

기하적으로 옳은 기준은 **양쪽 이미지의 재투영 오차**다.

$$
\min_{\hat{X}} \;\; d\!\left(x,\, P\hat{X}\right)^2 + d\!\left(x',\, P'\hat{X}\right)^2
$$

- $$d(\cdot,\cdot)$$: 이미지 평면에서의 유클리드 거리(픽셀)
- $$P\hat{X}$$: 추정한 3D 점 $$\hat{X}$$ 를 각 카메라로 **다시 투영한 위치**. 관측 $$x, x'$$ 와의 어긋남을 잰다

H&Z 12장은 이 문제를 epipolar 제약 아래에서 **6차 다항식의 근**으로 정확히 푸는 optimal triangulation을 제시한다. 실무에서는 DLT로 초기값을 잡고 비선형 정련하며, 점·시점이 많아지면 전체를 한꺼번에 최적화하는 **bundle adjustment**로 넘어간다.

### 6.4 정확도: baseline과 depth의 트레이드오프

스테레오에서 깊이는 시차(disparity) $$d = \frac{b f}{Z}$$ 로 정해진다 ($$b$$: baseline 길이, $$f$$: 픽셀 단위 초점거리, $$Z$$: 깊이). 이를 뒤집으면 깊이 오차가 보인다.

$$
\Delta Z \approx \frac{Z^2}{b f}\, \Delta d
$$

- $$\Delta d$$: 매칭의 픽셀 오차. 같은 픽셀 오차라도 깊이 오차는 **$$Z^2$$ 로 증폭**된다. 먼 물체일수록 깊이가 급격히 부정확해진다
- **baseline이 좁으면**: 두 시점이 비슷해 매칭은 쉽지만 깊이가 부정확하다
- **baseline이 넓으면**: 깊이는 정확하지만 시점 차가 커서 매칭이 어렵고 가림(occlusion)이 는다

## 7. two-view 복원 파이프라인: 전체 조립

지금까지의 조각을 순서대로 이으면 두 장의 사진에서 3D 점을 복원하는 완결된 파이프라인이 된다.

1. **calibration** → $$K$$, 왜곡 계수 (2장). 오프라인에서 한 번
2. **특징점 검출 + descriptor** (4.1, 4.2)
3. **매칭 + ratio test** (4.3)
4. **RANSAC + five-point** → $$E$$ 와 inlier 대응 (3.4, 5장)
5. **$$E$$ 분해 + cheirality check** → 상대 자세 $$(R, t)$$ (3.5)
6. **triangulation** → 3D point cloud (6장)
7. (확장) 시점을 계속 추가하며 **bundle adjustment**로 전체 정련 → SfM(Structure from Motion). COLMAP이 이 파이프라인의 표준 구현이다

## 정리 & 다음 글

| 단계 | 얻는 것 | 핵심 도구 |
| --- | --- | --- |
| 투영 모델 | 픽셀 ↔ ray 변환 | $$P = K[R \mid t]$$, 동차 좌표 |
| calibration | $$K$$, 왜곡, $$[R \mid t]$$ | Zhang's method, 재투영 오차 |
| epipolar geometry | 대응 제약, 상대 자세 | $$E = [t]_\times R$$, $$F$$, 8/5-point |
| 매칭 | 시점 간 대응 후보 | detector + descriptor + ratio test |
| RANSAC | outlier 없는 대응 | 최소 표본 + inlier 투표 |
| triangulation | 3D 점 | DLT, 재투영 오차 최소화 |

- 기하학은 **픽셀을 3D ray로 바꾸고, 시점 간 대응을 제약하고, 대응에서 3D 점을 복원**하는 도구다.
- 다음 글은 이 two-view 파이프라인을 **수백 장의 순서 없는 사진**으로 확장한다. 전통 3D reconstruction의 본체인 SfM·MVS다.
- 다음 글: [3D Reconstruction ② SfM·MVS: 이미지 뭉치에서 dense 3D까지, 전통 파이프라인]({% post_url 2026-07-05-3d-reconstruction-sfm-mvs %})
