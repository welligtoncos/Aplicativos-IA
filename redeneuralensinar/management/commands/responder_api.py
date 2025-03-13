import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, Concatenate, Lambda
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import random

# Configurações globais
np.random.seed(42)
tf.random.set_seed(42)

# Definição de operações e dificuldades
operations = {
    0: {"name": "adição", "symbol": "+", "function": lambda x, y: x + y},
    1: {"name": "subtração", "symbol": "-", "function": lambda x, y: x - y},
    2: {"name": "multiplicação", "symbol": "×", "function": lambda x, y: x * y},
    3: {"name": "divisão", "symbol": "÷", "function": lambda x, y: x // y if y != 0 else None}
}

difficulties = {
    0: {"name": "Muito Fácil", "range": (1, 10)},
    1: {"name": "Fácil", "range": (10, 50)},
    2: {"name": "Médio", "range": (50, 100)},
    3: {"name": "Difícil", "range": (100, 500)},
    4: {"name": "Muito Difícil", "range": (500, 1000)}
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

# Função para carregar problemas reais (simulados)
def load_real_problems():
    problems = [
        {"question": "Se eu tenho 8 maçãs e dou 3, com quantas fico?", "numbers": [8, 3], "operation": 1, "result": 5},
        {"question": "Maria tem 15 lápis e João tem 8. Quantos lápis eles têm juntos?", "numbers": [15, 8], "operation": 0, "result": 23},
        {"question": "Um pacote tem 24 biscoitos. Se cada criança receber 6, para quantas crianças dará?", "numbers": [24, 6], "operation": 3, "result": 4},
        {"question": "Uma caixa tem 7 fileiras com 6 chocolates cada. Quantos chocolates há no total?", "numbers": [7, 6], "operation": 2, "result": 42},
        {"question": "Pedro tinha 54 figurinhas e perdeu 17. Com quantas ficou?", "numbers": [54, 17], "operation": 1, "result": 37},
        {"question": "Uma pizza foi dividida em 8 pedaços iguais. Se comermos 3 pedaços, quantos sobram?", "numbers": [8, 3], "operation": 1, "result": 5},
        {"question": "Se cada livro custa R$ 12 e compro 5, quanto gastarei?", "numbers": [12, 5], "operation": 2, "result": 60},
        {"question": "Uma jarra tem 750ml de suco. Se servirmos 6 copos iguais, quanto terá em cada copo?", "numbers": [750, 6], "operation": 3, "result": 125},
        {"question": "Um ônibus tem 42 assentos. Se 36 estão ocupados, quantos estão vazios?", "numbers": [42, 36], "operation": 1, "result": 6},
        {"question": "Se cada caixa tem 25 laranjas e tenho 6 caixas, quantas laranjas tenho?", "numbers": [25, 6], "operation": 2, "result": 150}
    ]
    return problems

real_problems = load_real_problems()
# Geração de dados expandido
def generate_expanded_dataset(num_examples=5000, include_real_problems=True):
    X = []
    y = []
    op_dist = [0.3, 0.25, 0.25, 0.2]
    diff_dist = [0.15, 0.25, 0.3, 0.2, 0.1]

    for _ in range(num_examples):
        operation = np.random.choice(len(operations), p=op_dist)
        difficulty = np.random.choice(len(difficulties), p=diff_dist)
        min_val, max_val = difficulties[difficulty]["range"]

        if operation == 3:  # Divisão
            if difficulty <= 2:
                divisor = np.random.randint(1, min(10, max_val // 2))
            else:
                divisor = np.random.randint(2, min(20, max_val // 5))
            max_multiplier = max(1, max_val // divisor)
            multiplier = np.random.randint(1, max_multiplier + 1)
            num1 = divisor * multiplier
            num2 = divisor
            if difficulty >= 3 and random.random() > 0.7:
                possible_divisors = [d for d in range(2, min(50, num1)) if num1 % d == 0]
                if possible_divisors:
                    num2 = random.choice(possible_divisors)

        elif operation == 2:  # Multiplicação
            if difficulty <= 1:
                num1 = np.random.randint(1, 11)
                num2 = np.random.randint(1, 11)
            elif difficulty == 2:
                if random.choice([True, False]):
                    num1 = np.random.randint(10, min(max_val, 100))
                    num2 = np.random.randint(1, 10)
                else:
                    num1 = np.random.randint(1, 10)
                    num2 = np.random.randint(10, min(max_val, 100))
            else:
                num1 = np.random.randint(min_val // 2, max_val // 2)
                num2 = np.random.randint(min_val // 2, max_val // 2)
                special_case = random.random()
                if special_case > 0.8:
                    power = random.randint(1, 3)
                    num2 = 10 ** power
                elif special_case > 0.6:
                    num2 = random.choice([5, 25, 50])

        elif operation == 1:  # Subtração
            if difficulty <= 2:
                # Garantir que num1 > min_val para ter um intervalo válido
                num1 = np.random.randint(min_val + 1, max_val)
                num2 = np.random.randint(min_val, num1)
            else:
                num1 = np.random.randint(min_val, max_val)
                num2 = np.random.randint(min_val, max_val)
                if num1 < num2:
                    num1, num2 = num2, num1
                if difficulty >= 2 and random.random() > 0.7:
                    digits1 = [int(d) for d in str(num1)]
                    digits2 = [int(d) for d in str(num2)]
                    if len(digits2) <= len(digits1):
                        for i in range(min(len(digits1) - 1, len(digits2))):
                            if random.random() > 0.5:
                                digits2[i] = min(9, digits1[i] + random.randint(1, 3))
                        num2 = int(''.join(map(str, digits2)))
                        if num2 >= num1:
                            num2 = num1 - random.randint(1, min(10, num1))

        else:  # Adição (operation == 0)
            num1 = np.random.randint(min_val, max_val)
            max_second = min(max_val - num1, max_val)
            if max_second < 1:
                max_second = max_val
            num2 = np.random.randint(1, max_second + 1)
            if difficulty >= 2 and random.random() > 0.7:
                digits1 = [int(d) for d in str(num1)]
                digits2 = [int(d) for d in str(num2)]
                if len(digits2) <= len(digits1):
                    for i in range(min(len(digits1), len(digits2))):
                        if random.random() > 0.5:
                            digits2[i] = max(1, 10 - digits1[i] + random.randint(0, 3))
                    num2 = int(''.join(map(str, digits2)))

        result = operations[operation]["function"](num1, num2)
        if result is not None:
            X.append([num1, num2, operation, difficulty])
            y.append(result)

    if include_real_problems and real_problems:
        for problem in real_problems:
            X.append([problem["numbers"][0], problem["numbers"][1], problem["operation"], 1])
            y.append(problem["result"])

    return np.array(X), np.array(y)


def compute_attention(args):
    attention_query, attention_key, attention_value = args
    # Expandir as dimensões para a multiplicação
    attention_scores = tf.matmul(tf.expand_dims(attention_query, axis=1),
                                 tf.expand_dims(attention_key, axis=2))
    # Squeeze para remover as dimensões 1 e normalizar
    attention_scores = tf.squeeze(attention_scores, axis=[1, 2]) / tf.sqrt(tf.cast(128, tf.float32))
    attention_weights = tf.nn.softmax(attention_scores)
    # Expande a última dimensão para que a multiplicação seja compatível
    attention_weights = tf.expand_dims(attention_weights, axis=-1)
    context_vector = attention_weights * attention_value
    return context_vector

# Construção do modelo avançado com mecanismo de atenção
def build_advanced_model():
    input_layer = Input(shape=(4,))
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_layer)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l2=1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    # Definindo as projeções para atenção
    attention_query = Dense(128, activation='tanh')(x)
    attention_key = Dense(128, activation='tanh')(x)
    attention_value = Dense(128)(x)

    # Aplicando o mecanismo de atenção via Lambda
    context_vector = Lambda(compute_attention)([attention_query, attention_key, attention_value])

    # Concatenando o contexto com a entrada original
    combined = Concatenate()([context_vector, input_layer])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.1)(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

# Treinamento do modelo
def train_model(model, X, y, epochs=200, batch_size=64, validation_split=0.2, patience=15):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1, X_std)  # Evita divisão por zero
    X_norm = (X - X_mean) / X_std
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 3, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(X_norm, y, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, callbacks=callbacks, verbose=1)
    val_idx = int((1 - validation_split) * len(X))
    X_val_norm = X_norm[val_idx:]
    y_val = y[val_idx:]
    val_pred = model.predict(X_val_norm)
    val_pred_rounded = np.round(val_pred).flatten()
    mae = np.mean(np.abs(val_pred.flatten() - y_val))
    mse = np.mean(np.square(val_pred.flatten() - y_val))
    accuracy = np.mean(val_pred_rounded == y_val)
    print(f"\nAvaliação do modelo:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Acurácia (previsão arredondada): {accuracy:.4f}")
    metrics = {"mae": mae, "mse": mse, "accuracy": accuracy}
    return history, X_mean, X_std, metrics

# Função para fazer previsão
def predict_result(model, X_mean, X_std, num1, num2, operation, difficulty=1):
    X_input = np.array([[num1, num2, operation, difficulty]])
    X_input_norm = (X_input - X_mean) / X_std
    prediction = model.predict(X_input_norm, verbose=0)[0][0]
    if operation != 3 or (num2 != 0 and num1 % num2 == 0):
        prediction = round(prediction)
    return prediction


# Avaliação do modelo em dados de teste
def evaluate_model(model, X_test, y_test, X_mean, X_std):
    X_test_norm = (X_test - X_mean) / X_std
    predictions = model.predict(X_test_norm, verbose=0).flatten()
    predictions_rounded = np.round(predictions)
    mae = np.mean(np.abs(predictions - y_test))
    mse = np.mean(np.square(predictions - y_test))
    accuracy = np.mean(predictions_rounded == y_test)
    report = {"mae": float(mae), "mse": float(mse), "accuracy": float(accuracy), "total_samples": len(y_test)}
    error_by_operation = {}
    for op in range(4):
        op_mask = X_test[:, 2] == op
        if np.any(op_mask):
            op_pred = predictions[op_mask]
            op_true = y_test[op_mask]
            op_mae = np.mean(np.abs(op_pred - op_true))
            op_acc = np.mean(np.round(op_pred) == op_true)
            error_by_operation[operations[op]["name"]] = {"mae": float(op_mae), "accuracy": float(op_acc), "samples": int(np.sum(op_mask))}
    report["error_by_operation"] = error_by_operation
    return report


# Plot do histórico de treinamento
def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo (MSE)')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Treino')
    plt.plot(history.history['val_mae'], label='Validação')
    plt.title('Erro Absoluto Médio (MAE)')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mse'], label='Treino')
    plt.plot(history.history['val_mse'], label='Validação')
    plt.title('Erro Quadrático Médio (MSE)')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Geração da explicação passo a passo para a solução correta
def generate_solution_explanation(num1, num2, operation, correct_answer):
    op_name = operations[operation]["name"]
    op_symbol = operations[operation]["symbol"]
    if operation == 0:  # Adição
        return (
            f"Para resolver {num1} + {num2}:\n"
            f"1. Some {num1} e {num2} dígito por dígito, considerando o 'vai um' quando necessário.\n"
            f"2. O resultado final é {correct_answer}."
        )
    elif operation == 1:  # Subtração
        return (
            f"Para resolver {num1} - {num2}:\n"
            f"1. Subtraia {num2} de {num1} dígito por dígito, realizando o 'empréstimo' se necessário.\n"
            f"2. O resultado final é {correct_answer}."
        )
    elif operation == 2:  # Multiplicação
        return (
            f"Para resolver {num1} × {num2}:\n"
            f"1. Multiplique os números, considerando os deslocamentos de posição quando necessário.\n"
            f"2. Some os resultados parciais para obter o total final, que é {correct_answer}."
        )
    else:  # Divisão
        if num2 == 0:
            return "Divisão por zero não é definida."
        return (
            f"Para resolver {num1} ÷ {num2}:\n"
            f"1. Determine quantas vezes {num2} cabe em {num1}.\n"
            f"2. Divida passo a passo subtraindo {num2} de {num1} até não ser mais possível.\n"
            f"3. O resultado final (parte inteira) é {correct_answer}."
        )
    

    # Fornece feedback detalhado baseado na resposta do estudante
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


def run_interactive_test(num_questions=1):
    """
    Executa um teste interativo onde são gerados problemas aleatórios,
    o usuário informa sua resposta e o sistema fornece feedback detalhado.
    """
    print("Iniciando teste interativo. Responda as seguintes questões:\n")
    score = 0
    for i in range(num_questions):
        # Gere um problema aleatório (sem incluir problemas reais para manter o teste dinâmico)
        X_sample, y_sample = generate_expanded_dataset(num_examples=1, include_real_problems=False)
        num1, num2, op, diff = X_sample[0]
        # Formate a questão
        problem_text = f"Qual é o resultado de {num1} {operations[op]['symbol']} {num2}?"
        print(f"Questão {i+1}: {problem_text}")

        # Recebe a resposta do usuário
        user_input = input("Sua resposta: ")
        try:
            user_ans = int(user_input)
        except:
            print("Resposta inválida! Considerando como resposta errada.\n")
            user_ans = None

        # Obtenha o feedback detalhado
        feedback = provide_detailed_feedback(num1, num2, op, user_ans)
        if feedback["is_correct"]:
            score += 1

        print("\nFeedback:")
        print(feedback["explanation"])
        print("Detalhamento:")
        print(feedback.get("step_by_step", ""))
        print("-" * 40, "\n")

    print(f"Você acertou {score} de {num_questions} questões.")


# Exemplo de uso no Colab
if __name__ == "__main__":
    run_interactive_test(1)
    
    # # Gerar dados e construir o modelo
    # X, y = generate_expanded_dataset(num_examples=1000)
    # model = build_advanced_model()
    # history, X_mean, X_std, train_metrics = train_model(model, X, y, epochs=5, batch_size=32, validation_split=0.2, patience=3)

    # # Exemplo de previsão e feedback
    # num1, num2, op = 12, 7, 1  # Exemplo: 12 - 7
    # student_ans = 4
    # feedback = provide_detailed_feedback(num1, num2, op, student_ans)
    # print("\nFeedback ao estudante:")
    # print(json.dumps(feedback, indent=4, ensure_ascii=False))

    # # Relatório de avaliação
    # report = evaluate_model(model, X, y, X_mean, X_std)
    # print("\nRelatório de avaliação:")
    # print(json.dumps(report, indent=4, ensure_ascii=False))

    # # Exibir histórico de treinamento
    # plot_training_history(history)


    