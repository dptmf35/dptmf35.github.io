---
title: "전체 게시글"
layout: archive
permalink: /posts/
author_profile: true
---

<p class="page__meta">총 {{ site.posts | size }}개의 글</p>

{% for post in site.posts %}
  {% include archive-single.html type="list" %}
{% endfor %}
