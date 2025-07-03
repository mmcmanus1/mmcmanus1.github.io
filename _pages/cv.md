---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

{% raw %}
{% if site.resume_print_social_links %}
<ul class="list-unstyled">
{% for site_link in site.data.cv.social_links %}
  {% if site_link.present %}
  <li><strong>{{ site_link.label }}:</strong> <a href="{{ site_link.url }}">{{ site_link.url }}</a></li>
  {% endif %}
{% endfor %}
</ul>
{% endif %}

{% if site.resume_section_skills %}
<section class="main-content">
{% if site.resume_section_skills.enable %}
<h2><a id="skills"></a>Skills</h2>
{% for skill in site.data.cv.skills %}
<div class="container">
<div class="skillTitle">{{skill.name}}</div>
<div class="skillBar">
<div class="skillBarFill" style="width: {{skill.level}}%"></div>
</div>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_experience %}
<section class="main-content">
{% if site.resume_section_experience.enable %}
<h2><a id="experience"></a>Experience</h2>
{% for job in site.data.cv.experience %}
<div class="container">
<h3>{{job.company}}</h3>
<p class="resumeItemHeading">{{job.title}}</p>
<p class="resumeItemHeading">{{job.duration}} | {{job.location}}</p>
<p> {{job.description}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_education %}
<section class="main-content">
{% if site.resume_section_education.enable %}
<h2><a id="education"></a>Education</h2>
{% for degree in site.data.cv.education %}
<div class="container">
<h3>{{degree.institution}}</h3>
<p class="resumeItemHeading">{{degree.degree}} | {{degree.year}}</p>
<p><em>{{degree.thesis}}</em></p>
<p>Advisor: {{degree.advisor}}</p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_publications %}
<section class="main-content">
{% if site.resume_section_publications.enable %}
<h2><a id="publications"></a>Publications</h2>
{% for pub in site.data.cv.publications %}
<div class="container">
<h3>{{pub.title}}</h3>
<p class="resumeItemHeading">{{pub.authors}}</p>
<p><em>{{pub.venue}}</em></p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_projects %}
<section class="main-content">
{% if site.resume_section_projects.enable %}
<h2><a id="projects"></a>Projects</h2>
{% for project in site.data.cv.projects %}
<div class="container">
<h3>{{project.name}}</h3>
<p class="resumeItemHeading">{{project.abstract}}</p>
<p> {{project.description}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_volunteer %}
<section class="main-content">
{% if site.resume_section_volunteer.enable %}
<h2><a id="volunteer"></a>Service and Leadership</h2>
{% for volunteer in site.data.cv.volunteer %}
<div class="container">
<h3>{{volunteer.organization}}</h3>
<p class="resumeItemHeading">{{volunteer.position}}</p>
<p class="resumeItemHeading">{{volunteer.duration}} | {{volunteer.location}}</p>
<p> {{volunteer.description}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_recognition %}
<section class="main-content">
{% if site.resume_section_recognition.enable %}
<h2><a id="awards"></a>Awards</h2>
{% for award in site.data.cv.awards %}
<div class="container">
<h3>{{award.title}}</h3>
<p class="resumeItemHeading">{{award.date}}</p>
<p> {{award.description}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_associations %}
<section class="main-content">
{% if site.resume_section_associations.enable %}
<h2><a id="associations"></a>Associations</h2>
{% for association in site.data.cv.associations %}
<div class="container">
<h3>{{association.organization}}</h3>
<p class="resumeItemHeading">{{association.website}}</p>
<p> {{association.description}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_interests %}
<section class="main-content">
{% if site.resume_section_interests.enable %}
<h2><a id="interests"></a>Interests</h2>
{% for interest in site.data.cv.interests %}
<div class="container">
<h3>{{interest.name}}</h3>
<p> {{interest.keywords}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}

{% if site.resume_section_references %}
<section class="main-content">
{% if site.resume_section_references.enable %}
<h2><a id="references"></a>References</h2>
{% for reference in site.data.cv.references %}
<div class="container">
<h3>{{reference.name}}</h3>
<p class="resumeItemHeading">{{reference.position}}</p>
<p class="resumeItemHeading">{{reference.company}}</p>
<p> {{reference.reference}} </p>
</div>
{% endfor %}
{% endif %}
</section>
{% endif %}
{% endraw %}
