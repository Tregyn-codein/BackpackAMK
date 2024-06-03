from datetime import datetime
import flet as ft
import random
import matplotlib.pyplot as plt
import io
import base64
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

class DataHandler:
    def __init__(self, filename="data.json"):
        self.filename = filename
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {"items": [], "tests": []}

    def save_data(self):
        with open(self.filename, "w") as file:
            json.dump(self.data, file, indent=4)

    def add_item(self, value, weight):
        self.data["items"].append({"value": value, "weight": weight})
        self.save_data()

    def clear_items(self):
        self.data["items"] = []
        self.save_data()

    def add_test(self, test):
        self.data["tests"].append(test)
        self.save_data()

def initialize_pheromones(num_items):
    return [1.0] * num_items

def calculate_probabilities(pheromones, items, alpha, beta):
    probabilities = []
    for i in range(len(items)):
        heuristic = items[i].value / items[i].weight
        probabilities.append((pheromones[i] ** alpha) * (heuristic ** beta))
    total = sum(probabilities)
    return [p / total for p in probabilities]

def select_item(probabilities):
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if r < cumulative:
            return i
    return len(probabilities) - 1

def update_pheromones(pheromones, solutions, decay, Q):
    for i in range(len(pheromones)):
        pheromones[i] *= (1 - decay)
    for solution in solutions:
        for item in solution['items']:
            pheromones[item] += Q / solution['value']

def ant_colony_optimization(items, max_weight, num_ants, num_iterations, alpha, beta, decay, Q, progress_callback):
    num_items = len(items)
    pheromones = initialize_pheromones(num_items)
    best_solution = None
    best_values = []
    average_values = []

    for iteration in range(num_iterations):
        solutions = []
        total_value = 0
        for ant in range(num_ants):
            solution = {'items': [], 'value': 0, 'weight': 0}
            probabilities = calculate_probabilities(pheromones, items, alpha, beta)
            while solution['weight'] < max_weight:
                item_index = select_item(probabilities)
                item = items[item_index]
                if solution['weight'] + item.weight <= max_weight:
                    solution['items'].append(item_index)
                    solution['value'] += item.value
                    solution['weight'] += item.weight
                else:
                    break
            solutions.append(solution)
            total_value += solution['value']
            if best_solution is None or solution['value'] > best_solution['value']:
                best_solution = solution
        update_pheromones(pheromones, solutions, decay, Q)
        best_values.append(best_solution['value'])
        average_values.append(total_value / num_ants)

        # Вывод промежуточных результатов
        progress_callback(f"Итерация {iteration + 1}/{num_iterations}, Лучшая стоимость: {best_solution['value']}")

    return best_solution, best_values, average_values

def plot_results(test_results):
    fig, ax = plt.subplots()
    for test_name, results in test_results.items():
        best_values, average_values = results
        ax.plot(best_values, label=f"{test_name} - Лучшая стоимость")
        ax.plot(average_values, label=f"{test_name} - Средняя стоимость")
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Стоимость')
    ax.set_title('Муравьиный алгоритм для задачи о рюкзаке')
    ax.legend()

    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    return buf

def generate_random_items(num_items, max_value, max_weight):
    return [Item(random.randint(1, max_value), random.randint(1, max_weight)) for _ in range(num_items)]

def main(page: ft.Page, data_handler):
    page.title = "Муравьиный алгоритм для задачи о рюкзаке"
    page.scroll = "adaptive"

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        # extended=True,
        width=70,
        min_extended_width=400,
        group_alignment=-1,
        destinations=[
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.HOME_OUTLINED),
                selected_icon_content=ft.Icon(ft.icons.HOME),
                label="Главная",

            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.ACCESS_TIME),
                selected_icon_content=ft.Icon(ft.icons.ACCESS_TIME_FILLED),
                label="История",
            )
        ],
        on_change=lambda e: go_to_history(e) if e.control.selected_index == 1 else go_to_main,
    )

    def go_to_history(e):
        page.clean()
        page.go("/history")

    def go_to_main(e):
        page.clean()
        page.go("/")

    def generate_items(e):
        num_items = int(num_items_input.value)
        max_value = int(max_value_input.value)
        max_weight = int(max_weight_input.value)
        data_handler.clear_items()
        items = generate_random_items(num_items, max_value, max_weight)
        for item in items:
            data_handler.add_item(item.value, item.weight)
        update_items_list()

    def update_items_list():
        items_list.controls.clear()
        for index, item in enumerate(data_handler.data["items"]):
            items_list.controls.append(ft.Text(f"{index}: Стоимость: {item['value']}, Вес: {item['weight']}"))
        items_list.update()

    def run_tests(e):
        max_knapsack_weight = int(max_knapsack_weight_input.value)
        num_ants = int(num_ants_input.value)
        alpha = float(alpha_input.value)
        beta = float(beta_input.value)
        decay = float(decay_input.value)
        Q = int(Q_input.value)
        num_iterations = int(num_iterations_input.value)

        items = [Item(item['value'], item['weight']) for item in data_handler.data["items"]]
        test_results = {}
        progress_text.value = ""
        progress_text.update()

        def progress_callback(message):
            progress_text.value += message + "\n"
            progress_text.update()

        best_solution = None
        best_iteration = 0  # Добавляем переменную для хранения номера итерации
        best_values = []
        average_values = []
        pheromones = initialize_pheromones(len(items))  # Инициализируем феромоны

        for iteration in range(num_iterations):
            solutions = []
            total_value = 0
            for ant in range(num_ants):
                solution = {'items': [], 'value': 0, 'weight': 0}
                probabilities = calculate_probabilities(pheromones, items, alpha, beta)
                while solution['weight'] < max_knapsack_weight:  # Используем max_knapsack_weight
                    item_index = select_item(probabilities)
                    item = items[item_index]
                    if solution['weight'] + item.weight <= max_knapsack_weight:  # Используем max_knapsack_weight
                        solution['items'].append(item_index)
                        solution['value'] += item.value
                        solution['weight'] += item.weight
                    else:
                        break
                solutions.append(solution)
                total_value += solution['value']
                if best_solution is None or solution['value'] > best_solution['value']:
                    best_solution = solution
                    best_iteration = iteration + 1  # Обновляем номер итерации

            update_pheromones(pheromones, solutions, decay, Q)
            best_values.append(best_solution['value'])
            average_values.append(total_value / num_ants)

            # Вывод промежуточных результатов
            progress_callback(f"Итерация {iteration + 1}/{num_iterations}, Лучшая стоимость: {best_solution['value']}")

        test_results["Test"] = (best_values, average_values)
        plot_buf = plot_results(test_results)
        plot_image.src_base64 = base64.b64encode(plot_buf.getvalue()).decode('utf-8')
        plot_image.update()

        result_text.value = f"Предметы: {best_solution['items']}\nОбщая стоимость: {best_solution['value']}\nОбщий вес: {best_solution['weight']}\nНайдено на итерации: {best_iteration}"
        result_text.update()

        data_handler.add_test({
            "date": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "items": data_handler.data["items"],
            "settings": {
                "max_knapsack_weight": max_knapsack_weight,
                "num_ants": num_ants,
                "alpha": alpha,
                "beta": beta,
                "decay": decay,
                "Q": Q,
                "num_iterations": num_iterations
            },
            "result": {
                "best_solution": best_solution,
                "best_values": best_values,
                "average_values": average_values,
                "best_iteration": best_iteration,  # Сохраняем номер итерации
                "plot": base64.b64encode(plot_buf.getvalue()).decode('utf-8')
            }
        })

    num_items_input = ft.TextField(label="Количество предметов", value="100")
    max_value_input = ft.TextField(label="Максимальная стоимость", value="100")
    max_weight_input = ft.TextField(label="Максимальный вес", value="10")
    generate_button = ft.ElevatedButton(text="Сгенерировать список", on_click=generate_items)
    items_list = ft.Column(scroll="always", height=400)

    max_knapsack_weight_input = ft.TextField(label="Максимальный вес рюкзака", value="500")
    num_ants_input = ft.TextField(label="Количество муравьев", value="50")
    alpha_input = ft.TextField(label="Альфа (α)", value="1.0")
    beta_input = ft.TextField(label="Бета (β)", value="2.0")
    decay_input = ft.TextField(label="Коэффициент испарения феромонов (ρ)", value="0.5")
    Q_input = ft.TextField(label="Q", value="100")
    num_iterations_input = ft.TextField(label="Количество итераций", value="100")
    run_button = ft.ElevatedButton(text="Запустить тест", on_click=run_tests)
    progress_text = ft.Text()
    result_text = ft.Text()
    plot_image = ft.Image(src="img/fireants-ants.gif",
                          width=640,
                          height=480,
                          fit=ft.ImageFit.CONTAIN)

    input_column = ft.Column([
        num_items_input,
        max_value_input,
        max_weight_input,
        generate_button,
        items_list,
    ], alignment=ft.MainAxisAlignment.START,
    height=700, width=250)

    settings_column = ft.Column([
        max_knapsack_weight_input,
        num_ants_input,
        alpha_input,
        beta_input,
        decay_input,
        Q_input,
        num_iterations_input,
        run_button
    ], alignment=ft.MainAxisAlignment.START,
    height=700, width=250)

    results_column = ft.Column([
        plot_image,
        ft.Row([
            ft.Container(content=ft.Column([progress_text], scroll="always", height=200)),
            ft.Container(content=ft.Column([result_text], scroll="always", height=200, width=300))
        ])
    ], alignment=ft.MainAxisAlignment.START,
    height=700)

    page.add(ft.Row([rail,
                            ft.VerticalDivider(width=1),
                            input_column,
                            settings_column,
                            results_column], expand=False, height=700))

    # Вызовите update_items_list для отображения существующего списка предметов
    update_items_list()

def history_page(page: ft.Page, data_handler):
    page.title = "История тестов"
    page.scroll = "adaptive"

    rail = ft.NavigationRail(
        selected_index=1,
        label_type=ft.NavigationRailLabelType.ALL,
        # extended=True,
        width=70,
        min_extended_width=400,
        group_alignment=-1,
        destinations=[
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.HOME_OUTLINED),
                selected_icon_content=ft.Icon(ft.icons.HOME),
                label="Главная",

            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.ACCESS_TIME),
                selected_icon_content=ft.Icon(ft.icons.ACCESS_TIME_FILLED),
                label="История",
            )
        ],
        on_change=lambda e: go_to_main(e) if e.control.selected_index == 0 else go_to_history,
    )

    def go_to_history(e):
        page.clean()
        page.go("/history")

    def go_to_main(e):
        page.clean()
        page.go("/")

    history_list = ft.Column(scroll="always", height=500)

    def load_history():
        history_list.controls.clear()
        for test in data_handler.data["tests"]:
            date =  test["date"]
            items_str = ", ".join([f"({item['value']}, {item['weight']})" for item in test["items"][0:100]])
            settings_str = f"Вес рюкзака: {test['settings']['max_knapsack_weight']}, Муравьи: {test['settings']['num_ants']}, α: {test['settings']['alpha']}, β: {test['settings']['beta']}, ρ: {test['settings']['decay']}, Q: {test['settings']['Q']}, Итерации: {test['settings']['num_iterations']}"
            result_str = f"Лучшая стоимость: {test['result']['best_solution']['value']}, Общий вес: {test['result']['best_solution']['weight']}, Найдено на итерации: {test['result']['best_iteration']}"
            plot_image = ft.Image(width=500,
                                  src_base64=test['result']["plot"],
                                  fit=ft.ImageFit.CONTAIN)
            history_list.controls.append(ft.Row([
            ft.Container(content=ft.Text(f"◷ {date}\nПредметы: {items_str}...\nНастройки: {settings_str}\nРезультат: {result_str}\n",width=600)),
            ft.Container(content=plot_image)
        ]))
        history_list.update()


    page.add(ft.Row([
        rail,
        ft.VerticalDivider(width=1),
        ft.Column([
            ft.Text("История тестов", size=24),
            history_list])
    ], expand = False, height = 700))

    # Вызов функции load_history после добавления элементов на страницу
    load_history()

def main_app(page: ft.Page):
    data_handler = DataHandler()  # Создаем экземпляр DataHandler

    def route_change(route):
        if page.route == "/history":
            history_page(page, data_handler)  # Передаем data_handler
        else:
            main(page, data_handler)

    page.on_route_change = route_change
    page.clean()
    page.go(page.route)

ft.app(target=main_app)
