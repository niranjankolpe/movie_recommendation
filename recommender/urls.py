from django.urls import path
from recommender import views

urlpatterns = [
    path('', views.home, name="index"),
    path('home', views.home, name="home"),
    path('train_model', views.train_model, name="train_model"),
    path('recommend', views.recommend, name="recommend"),
]
