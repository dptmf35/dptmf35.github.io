# Research Archive

로보틱스와 인공지능 논문을 읽고 정리하는 개인 공부 아카이브입니다.
[Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) Jekyll 테마 기반이며,
폰트로 [Pretendard](https://github.com/orioncactus/pretendard)를 사용합니다.

🔗 **배포 주소**: https://dptmf35.github.io

## 새 논문 리뷰 작성하기

`_posts/` 폴더에 `YYYY-MM-DD-제목.md` 형식의 파일을 만듭니다.

```markdown
---
title: "[논문리뷰] 논문 제목"
excerpt: "한 줄 요약"
categories:
  - Robotics        # 또는 AI
  - Manipulation    # 세부 카테고리
tags:
  - 키워드1
  - 키워드2
toc: true
toc_sticky: true
---

본문...
```

## 카테고리 구조

- **Robotics** — SLAM, Manipulation, Locomotion, Robot Learning, Perception
- **AI** — Computer Vision, Reinforcement Learning, Multimodal, Generative Models, NLP

## 로컬에서 미리보기 (선택)

Ruby가 설치되어 있어야 합니다.

```bash
bundle install
bundle exec jekyll serve
# http://localhost:4000 접속
```

## 배포

`main` 브랜치에 push하면 GitHub Actions가 자동으로 빌드 & 배포합니다.
(GitHub 저장소 Settings → Pages → Source를 **GitHub Actions**로 설정)
