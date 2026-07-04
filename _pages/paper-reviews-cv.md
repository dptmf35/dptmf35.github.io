---
title: "논문리뷰 · CV"
layout: archive
permalink: /paper-reviews/cv/
author_profile: true
---

{% assign posts = site.categories['CV'] | where_exp: "item", "item.categories contains '논문리뷰'" %}
<p class="page__meta">총 {{ posts | size }}개의 글</p>

{% for post in posts %}
  {% include archive-single.html type="list" %}
{% endfor %}
