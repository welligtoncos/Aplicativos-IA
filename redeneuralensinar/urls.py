from django.urls import path
from . import views

urlpatterns = [
    # Remova ou comente a rota abaixo se não for necessária:
    # path('', views.home, name='home_redeneuralensinar'),
    path('api/feedback/', views.FeedbackAPIView.as_view(), name='feedback_api'),
]
