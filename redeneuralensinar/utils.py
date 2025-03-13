import random
import numpy as np

# Suponha que as operações e os dicionários já estejam definidos
operations = {
    0: {"name": "adição", "symbol": "+", "function": lambda x, y: x + y},
    1: {"name": "subtração", "symbol": "-", "function": lambda x, y: x - y},
    2: {"name": "multiplicação", "symbol": "×", "function": lambda x, y: x * y},
    3: {"name": "divisão", "symbol": "÷", "function": lambda x, y: x // y if y != 0 else None}
}

error_patterns = {
    "addition": {
        "wrong_operation_sub": lambda x, y, r: abs(r - (x - y)) < 0.5,
        "wrong_operation_mul": lambda x, y, r: abs(r - (x * y)) < 0.5,
        "digit_error": lambda x, y, r: abs(r - (x + y)) <= 10 and abs(r - (x + y)) > 0,
        "reversed_digits": lambda x, y, r: abs(r - int(str(x + y)[::-1])) < 0.5 if len(str(x + y)) > 1 else False
    },
    "subtraction": {
        "wrong_operation_add": lambda x, y, r: abs(r - (x + y)) < 0.5,
        "wrong_operation_mul": lambda x, y, r: abs(r - (x * y)) < 0.5,
        "wrong_order": lambda x, y, r: abs(r - (y - x)) < 0.5,
        "missed_borrow": lambda x, y, r: any(int(dx) < int(dy) for dx, dy in zip(str(x).zfill(len(str(y))), str(y)))
    },
    "multiplication": {
        "wrong_operation_add": lambda x, y, r: abs(r - (x + y)) < 0.5,
        "off_by_factor": lambda x, y, r: abs(r - (x * y)) % min(x, y) == 0 and r != x * y,
        "table_error": lambda x, y, r: abs(r - (x * y)) <= max(x, y) and r != x * y
    },
    "division": {
        "wrong_operation_mul": lambda x, y, r: abs(r - (x * y)) < 0.5,
        "wrong_order": lambda x, y, r: abs(r - (y // x if x != 0 else 0)) < 0.5,
        "remainder_included": lambda x, y, r: abs(r - ((x // y) + (x % y) / 100)) < 0.1 if y != 0 else False
    }
}

def generate_solution_explanation(num1, num2, operation, correct_answer):
    op_name = operations[operation]["name"]
    op_symbol = operations[operation]["symbol"]
    if operation == 0:
        return (
            f"Para resolver {num1} + {num2}:\n"
            f"1. Some {num1} e {num2} dígito por dígito, considerando o 'vai um' quando necessário.\n"
            f"2. O resultado final é {correct_answer}."
        )
    elif operation == 1:
        return (
            f"Para resolver {num1} - {num2}:\n"
            f"1. Subtraia {num2} de {num1} dígito por dígito, realizando o 'empréstimo' se necessário.\n"
            f"2. O resultado final é {correct_answer}."
        )
    elif operation == 2:
        return (
            f"Para resolver {num1} × {num2}:\n"
            f"1. Multiplique os números, considerando os deslocamentos de posição quando necessário.\n"
            f"2. Some os resultados parciais para obter o total final, que é {correct_answer}."
        )
    else:
        if num2 == 0:
            return "Divisão por zero não é definida."
        return (
            f"Para resolver {num1} ÷ {num2}:\n"
            f"1. Determine quantas vezes {num2} cabe em {num1}.\n"
            f"2. Divida passo a passo subtraindo {num2} de {num1} até não ser mais possível.\n"
            f"3. O resultado final (parte inteira) é {correct_answer}."
        )

def provide_detailed_feedback(num1, num2, operation, student_answer):
    op_func = operations[operation]["function"]
    correct_answer = op_func(num1, num2)
    is_correct = student_answer == correct_answer
    op_name = operations[operation]["name"]
    op_symbol = operations[operation]["symbol"]

    feedback = {
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "problem": f"{num1} {op_symbol} {num2}",
        "student_answer": student_answer
    }

    if is_correct:
        messages = [
            f"Correto! {num1} {op_symbol} {num2} = {correct_answer}.",
            f"Muito bem! A resposta {correct_answer} está correta.",
            f"Excelente! Você acertou.",
            f"Perfeito! {correct_answer} é a resposta correta."
        ]
        feedback["message"] = random.choice(messages)
        feedback["explanation"] = generate_solution_explanation(num1, num2, operation, correct_answer)
        feedback["next_steps"] = "Continue praticando com problemas mais desafiadores!"
        return feedback

    error_type = "unknown"
    error_pat = error_patterns.get(operations[operation]["name"], {})
    for err_type, check_func in error_pat.items():
        if check_func(num1, num2, student_answer):
            error_type = err_type
            break
    feedback["error_type"] = error_type

    if error_type == "wrong_operation_add":
        feedback["explanation"] = f"Parece que você somou em vez de fazer {op_name}."
    elif error_type == "wrong_operation_sub":
        feedback["explanation"] = f"Parece que você subtraiu em vez de fazer {op_name}."
    elif error_type == "wrong_operation_mul":
        feedback["explanation"] = f"Parece que você multiplicou em vez de fazer {op_name}."
    elif error_type == "wrong_order":
        feedback["explanation"] = f"A ordem dos números importa em {op_name}. Verifique qual número deve ser o minuendo/dividendo."
    elif error_type == "digit_error":
        feedback["explanation"] = "Você cometeu um pequeno erro de cálculo. Verifique a soma/subtração dos dígitos."
    elif error_type == "reversed_digits":
        feedback["explanation"] = f"Parece que você inverteu os dígitos no resultado. O correto é {correct_answer}."
    elif error_type == "off_by_factor":
        feedback["explanation"] = "Seu resultado está incorreto por um fator. Verifique a multiplicação cuidadosamente."
    elif error_type == "table_error":
        feedback["explanation"] = "Verifique a tabuada. Pode ter havido um erro de cálculo básico."
    elif error_type == "missed_borrow":
        feedback["explanation"] = "Você pode ter esquecido de fazer o 'empréstimo' ao subtrair dígitos maiores dos menores."
    elif error_type == "remainder_included":
        feedback["explanation"] = "Na divisão inteira, não incluímos o resto no resultado."
    else:
        feedback["explanation"] = f"A resposta correta é {correct_answer}. Tente refazer o cálculo."

    feedback["step_by_step"] = generate_solution_explanation(num1, num2, operation, correct_answer)
    feedback["next_steps"] = "Experimente praticar com problemas semelhantes para reforçar este conceito."
    return feedback
