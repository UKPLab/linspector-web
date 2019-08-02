from django.contrib.sitemaps import Sitemap
from django.urls import reverse

class StaticViewSitemap(Sitemap):

    priority = 0.5
    changefreq = 'weekly'

    def items(self):
        return ['index', 'about', 'paper', 'select_language']

    def location(self, item):
        return reverse(item)
