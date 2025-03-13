from django.db import models

OPERATION_CHOICES = (
    (0, 'Adição'),
    (1, 'Subtração'),
    (2, 'Multiplicação'),
    (3, 'Divisão'),
)

class Feedback(models.Model):
    num1 = models.IntegerField(verbose_name="Número 1")
    num2 = models.IntegerField(verbose_name="Número 2")
    operation = models.IntegerField(choices=OPERATION_CHOICES, verbose_name="Operação")
    student_answer = models.IntegerField(verbose_name="Resposta do Aluno")
    correct_answer = models.IntegerField(verbose_name="Resposta Correta")
    is_correct = models.BooleanField(verbose_name="Está Correta?")
    feedback_message = models.TextField(verbose_name="Mensagem de Feedback")
    explanation = models.TextField(verbose_name="Explicação")
    step_by_step = models.TextField(blank=True, null=True, verbose_name="Passo a Passo")
    error_type = models.CharField(max_length=50, blank=True, null=True, verbose_name="Tipo de Erro")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Criado em")

    def __str__(self):
        return f"{self.num1} {self.get_operation_display()} {self.num2} - {self.student_answer}"

    class Meta:
        verbose_name = "Feedback"
        verbose_name_plural = "Feedbacks"
        ordering = ['-created_at']
