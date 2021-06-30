---
layout: page
title: "Blog posts"
permalink: /blog
comments: false
---

Things I have written:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
