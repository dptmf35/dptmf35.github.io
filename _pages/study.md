---
title: "공부"
layout: archive
permalink: /study/
author_profile: true
---

{% assign notes = site.categories['공부'] %}
<p class="page__meta">총 {{ notes | size }}개의 글</p>

{% for post in notes %}
  {% include archive-single.html type="list" %}
{% endfor %}
