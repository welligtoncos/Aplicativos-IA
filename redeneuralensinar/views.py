# redeeuralensinar/views.py
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import provide_detailed_feedback
from .models import Feedback

 

class FeedbackAPIView(APIView):
    """
    API que recebe os dados de um problema matemático e a resposta do estudante,
    gera o feedback e o registra no banco de dados.
    """
    def post(self, request, *args, **kwargs):
        try:
            num1 = int(request.data.get("num1"))
            num2 = int(request.data.get("num2"))
            operation = int(request.data.get("operation"))
            student_answer = int(request.data.get("student_answer"))
        except (ValueError, TypeError):
            return Response(
                {"error": "Dados de entrada inválidos. Certifique-se de enviar valores inteiros."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Gera o feedback utilizando a função definida em utils.py
        feedback_data = provide_detailed_feedback(num1, num2, operation, student_answer)

        # Registra o feedback no banco de dados
        Feedback.objects.create(
            num1=num1,
            num2=num2,
            operation=operation,
            student_answer=student_answer,
            correct_answer=feedback_data.get("correct_answer"),
            is_correct=feedback_data.get("is_correct"),
            feedback_message=feedback_data.get("message"),
            explanation=feedback_data.get("explanation"),
            step_by_step=feedback_data.get("step_by_step"),
            error_type=feedback_data.get("error_type")
        )

        return Response(feedback_data, status=status.HTTP_200_OK)
