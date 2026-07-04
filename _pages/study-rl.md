---
title: "공부 · RL"
layout: archive
permalink: /study/rl/
author_profile: true
---

{% assign posts = site.categories['RL'] | where_exp: "item", "item.categories contains '공부'" %}
<p class="page__meta">총 {{ posts | size }}개의 글</p>

{% for post in posts %}
  {% include archive-single.html type="list" %}
{% endfor %}
