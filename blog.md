---
layout: page
title: "Blog"
permalink: /blog
comments: false
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">({{ post.date | date: "%Y-%m-%d" }}) {{ post.title }}</a>
	  <p style="font-size:85%">{{ post.tldr }}</p>
    </li>
  {% endfor %}
</ul>
