---
title: "[공지] 논문 리뷰 아카이브를 시작하며"
excerpt: "이 블로그의 목적과 논문 리뷰 작성 방식에 대한 소개입니다."
categories:
  - Notice
tags:
  - blog
  - intro
toc: true
toc_sticky: true
last_modified_at: 2026-07-05
---

## 이 블로그는

로보틱스와 인공지능 논문을 읽고 정리하는 개인 공부 아카이브입니다.
읽은 논문마다 **문제 정의 → 핵심 아이디어 → 방법 → 실험 → 한계**의 흐름으로 정리합니다.

## 새 논문 리뷰 작성 방법

`_posts/` 폴더에 `YYYY-MM-DD-제목.md` 형식으로 파일을 만들면 됩니다.
예: `2026-07-10-nerf.md`

파일 상단(front matter)에 아래처럼 카테고리와 태그를 지정하세요.

```yaml
---
title: "[논문리뷰] NeRF: Representing Scenes as Neural Radiance Fields"
excerpt: "한 줄 요약을 여기에 적습니다."
categories:
  - AI
  - Computer Vision
tags:
  - NeRF
  - 3D Reconstruction
  - novel view synthesis
toc: true
toc_sticky: true
---
```

이후 마크다운으로 본문을 작성하면 됩니다. 끝!
