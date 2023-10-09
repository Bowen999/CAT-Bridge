from django.urls import re_path

from . import views

urlpatterns = [
        re_path("CAT/",views.cat),
	re_path("CAT_RESULT/",views.cat_result),
        re_path("about/", views.about),  # new url pattern for about page
        re_path("tutorial/", views.tutorial),  # new url pattern for tutorial page
        re_path("contact/", views.contact),  # new url pattern for hello page
        re_path("home/",views.home)
        ]