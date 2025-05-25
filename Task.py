import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sphere_function(x):
    return sum(xi ** 2 for xi in x)


def random_point(bounds):
    return [random.uniform(low, high) for low, high in bounds]


def neighbor(point, bounds, step_size=0.1):
    return [max(min(xi + random.uniform(-step_size, step_size), bounds[i][1]), bounds[i][0])
            for i, xi in enumerate(point)]


def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    current = random_point(bounds)
    current_value = func(current)

    for _ in range(iterations):
        next_point = neighbor(current, bounds)
        next_value = func(next_point)

        if abs(current_value - next_value) < epsilon:
            break

        if next_value < current_value:
            current, current_value = next_point, next_value

    return current, current_value


def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    best = random_point(bounds)
    best_value = func(best)

    for _ in range(iterations):
        candidate = random_point(bounds)
        candidate_value = func(candidate)

        if abs(best_value - candidate_value) < epsilon:
            break

        if candidate_value < best_value:
            best, best_value = candidate, candidate_value

    return best, best_value


def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    current = random_point(bounds)
    current_value = func(current)
    best = current
    best_value = current_value

    for _ in range(iterations):
        if temp < epsilon:
            break

        next_point = neighbor(current, bounds)
        next_value = func(next_point)

        delta = next_value - current_value

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current, current_value = next_point, next_value

            if current_value < best_value:
                best, best_value = current, current_value

        temp *= cooling_rate

    return best, best_value


def plot_sphere_with_points(bounds, points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Позначимо знайдені точки
    for label, point in points.items():
        x, y = point
        z = sphere_function([x, y])
        ax.scatter(x, y, z, label=label, s=50)

    ax.set_title("Функція Сфери та результати оптимізації")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    bounds = [(-5, 5), (-5, 5)]

    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)

    print("Hill Climbing:", hc_solution, "Значення:", hc_value)
    print("Random Local Search:", rls_solution, "Значення:", rls_value)
    print("Simulated Annealing:", sa_solution, "Значення:", sa_value)

    # Візуалізація
    plot_sphere_with_points(bounds, {
        "Hill Climbing": hc_solution,
        "Random Local Search": rls_solution,
        "Simulated Annealing": sa_solution
    })
