from math import comb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent


# ======================================================================
# Среда
# ======================================================================
# Gridworld 5x5: старт в левом верхнем углу, цель в правом нижнем. За шаг -1,
# за достижение цели +10, то есть агенту выгодно дойти быстро. Переходы шумные:
# с вероятностью slip агент едет не туда, куда собирался. Этот шум вместе со
# случайностью epsilon-жадной стратегии и создаёт разброс между запусками -
# ровно то, ради чего всё затевается.

SIZE = 5
N_STATES = SIZE * SIZE
N_ACTIONS = 4
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # вверх, вниз, влево, вправо
GOAL = N_STATES - 1


def step(state, action, rng, slip=0.1):
    """Один шаг среды: новое состояние, награда, флаг завершения."""
    if rng.random() < slip:
        action = rng.integers(N_ACTIONS)

    row, col = divmod(state, SIZE)
    d_row, d_col = MOVES[action]
    # По краям поля стенки: выйти за границу нельзя, упрёмся и останемся на месте.
    row = min(max(row + d_row, 0), SIZE - 1)
    col = min(max(col + d_col, 0), SIZE - 1)

    new_state = row * SIZE + col
    done = new_state == GOAL
    return new_state, (10.0 if done else -1.0), done


def run_qlearning(seed, n_episodes=300, alpha=0.1, gamma=0.95, eps=0.1,
                  max_steps=100):
    """Обучает одного агента и возвращает одно число - качество этого запуска.

    Качество меряем как среднюю награду за последние 50 эпизодов: к этому
    моменту агент уже чему-то научился, а усреднение по полусотне эпизодов
    снижает случайность отдельной траектории.

    Единственное, чем один запуск отличается от другого, - это seed.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []

    for _ in range(n_episodes):
        state, total, done = 0, 0.0, False

        for _ in range(max_steps):
            # epsilon-жадный выбор: изредка исследуем, обычно жадничаем
            if rng.random() < eps:
                action = rng.integers(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            new_state, reward, done = step(state, action, rng)

            # обычное обновление Q-learning
            td_target = reward + gamma * np.max(Q[new_state]) * (not done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            total += reward
            state = new_state
            if done:
                break

        returns.append(total)

    return float(np.mean(returns[-50:]))


# ======================================================================
# Три способа сравнить две группы запусков
# ======================================================================

def naive_says_better(a, b, rng=None):
    """Так это чаще всего и делают: "среднее +- std, у нас выше - значит лучше".

    Формализуем как непересекающиеся интервалы [mean - std, mean + std].
    Стоит сразу отметить: это вообще не статистический тест, никакого
    заявленного уровня значимости у него нет. Тем интереснее посмотреть,
    как он себя ведёт.
    """
    mean_a, std_a = np.mean(a), np.std(a, ddof=1)
    mean_b, std_b = np.mean(b), np.std(b, ddof=1)
    return (mean_a - std_a > mean_b + std_b) or (mean_b - std_b > mean_a + std_a)


def bootstrap_says_better(a, b, rng, n_boot=400):
    """Аккуратнее: строим 95% бутстрэп-интервалы средних и смотрим, пересекаются ли.
    """
    def ci(x):
        resamples = rng.choice(x, size=(n_boot, len(x)), replace=True)
        means = resamples.mean(axis=1)
        return np.percentile(means, [2.5, 97.5])

    low_a, high_a = ci(a)
    low_b, high_b = ci(b)
    return (low_a > high_b) or (low_b > high_a)


def permutation_says_better(a, b, rng, n_perm=199, alpha=0.05):
    """
    Корректная опорная процедура: двусторонний перестановочный тест.
    """
    pooled = np.concatenate([a, b])
    n = len(a)
    observed = abs(np.mean(a) - np.mean(b))

    # argsort от случайных чисел - быстрый способ получить пачку перестановок
    order = np.argsort(rng.random((n_perm, len(pooled))), axis=1)
    shuffled = pooled[order]
    diffs = np.abs(shuffled[:, :n].mean(axis=1) - shuffled[:, n:].mean(axis=1))

    # +1 сверху и снизу - стандартная поправка, чтобы p-значение не занулялось
    p_value = (1 + np.sum(diffs >= observed)) / (n_perm + 1)
    return p_value < alpha


PROCEDURES = {
    "наивно (±std)": naive_says_better,
    "бутстрэп-ДИ": bootstrap_says_better,
    "перестановочный": permutation_says_better,
}


# ======================================================================
# Часть 1. Ошибка первого рода: находим ли различие там, где его нет
# ======================================================================

def false_positive_rate(scores, group_size, n_trials, rng, procedure):
    """
    Берём две непересекающиеся подгруппы из общего пула и сравниваем.
    """
    hits = 0
    for _ in range(n_trials):
        idx = rng.permutation(len(scores))[: 2 * group_size]
        a = scores[idx[:group_size]]
        b = scores[idx[group_size:]]
        hits += procedure(a, b, rng)
    return hits / n_trials


# ======================================================================
# Часть 2. Мощность: замечаем ли различие, которое есть
# ======================================================================

def power(scores, group_size, effect, n_trials, rng, procedure):
    """
    Доля случаев, когда процедура находит искусственно внесённый эффект.
    """
    hits = 0
    for _ in range(n_trials):
        idx = rng.permutation(len(scores))[: 2 * group_size]
        a = scores[idx[:group_size]]
        b = scores[idx[group_size:]] + effect
        hits += procedure(a, b, rng)
    return hits / n_trials


def min_p_value(group_size):
    """
    Наименьшее p-значение, которое перестановочный тест вообще может выдать.
    """
    return 2 / comb(2 * group_size, group_size)


def min_seeds_for(target_power, powers_by_size, group_sizes):
    """Наименьший размер группы, при котором мощность дотягивает до целевой."""
    for size, value in zip(group_sizes, powers_by_size):
        if value >= target_power:
            return size
    return None


# ======================================================================

def main():
    N_SEEDS = 200          # размер пула запусков
    N_TRIALS_FP = 800      # повторов для оценки ошибки I рода
    N_TRIALS_POWER = 400   # повторов для оценки мощности
    GROUP_SIZES = [3, 5, 10, 20]
    EFFECTS_IN_SIGMA = [0.5, 1.0, 1.5, 2.0]

    # ---- обучение пула --------------------------------------------------
    print(f"Обучаю {N_SEEDS} запусков одной и той же конфигурации...")
    scores = np.array([run_qlearning(seed) for seed in range(N_SEEDS)])
    sigma = scores.std(ddof=1)

    try:
        np.save(OUT_DIR / "scores.npy", scores)
    except OSError as err:
        print(f"  (не удалось сохранить scores.npy: {err} - продолжаю)")

    print(f"Качество по запускам: среднее {scores.mean():.2f}, "
          f"std {sigma:.2f}, размах [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"Размах составляет {(scores.max() - scores.min()) / sigma:.1f} "
          f"стандартных отклонения - вот с каким разбросом мы имеем дело.\n")

    rng = np.random.default_rng(0)

    # ---- часть 1: ошибка первого рода -----------------------------------
    print("=" * 64)
    print("ЧАСТЬ 1. Как часто находим различие, которого нет")
    print("=" * 64)
    print(f"{'в группе':>10}" + "".join(f"{name:>18}" for name in PROCEDURES))

    fp_rates = {name: [] for name in PROCEDURES}
    for size in GROUP_SIZES:
        row = f"{size:>10}"
        for name, procedure in PROCEDURES.items():
            rate = false_positive_rate(scores, size, N_TRIALS_FP, rng, procedure)
            fp_rates[name].append(rate)
            row += f"{rate:>17.1%}"
        print(row)

    # ---- часть 2: мощность ----------------------------------------------
    print("\n" + "=" * 64)
    print("ЧАСТЬ 2. Мощность перестановочного теста:")
    print("вероятность заметить улучшение заданного размера")
    print("=" * 64)
    print(f"{'в группе':>10}" +
          "".join(f"{'+' + str(e) + ' sigma':>13}" for e in EFFECTS_IN_SIGMA))

    power_curves = {e: [] for e in EFFECTS_IN_SIGMA}
    for size in GROUP_SIZES:
        row = f"{size:>10}"
        for effect in EFFECTS_IN_SIGMA:
            value = power(scores, size, effect * sigma, N_TRIALS_POWER, rng,
                          permutation_says_better)
            power_curves[effect].append(value)
            row += f"{value:>12.1%}"
        print(row)

    print("\nПредел разрешения самого теста (он дискретный):")
    for size in GROUP_SIZES:
        p_min = min_p_value(size)
        verdict = "значимость 5% недостижима в принципе" if p_min > 0.05 else "ок"
        print(f"  {size:>2} на группу -> минимальное p = {p_min:.3f}   {verdict}")
    print("  То есть при 3 запусках корректный тест не может объявить различие")
    print("  значимым ни при каком размере эффекта. Если в статье на трёх сидах")
    print("  всё-таки заявлено улучшение, оно получено процедурой без контроля ошибки.")

    print("\nСколько запусков нужно, чтобы заметить эффект с вероятностью 80%:")
    for effect in EFFECTS_IN_SIGMA:
        need = min_seeds_for(0.8, power_curves[effect], GROUP_SIZES)
        verdict = f"{need} на группу" if need else f"больше {GROUP_SIZES[-1]}"
        print(f"  улучшение на {effect} sigma  ->  {verdict}")

    # ---- графики ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].hist(scores, bins=25, color="#3170A6", alpha=0.85)
    axes[0].set_title(f"Разброс качества по {N_SEEDS} запускам\n"
                      "(конфигурация одна и та же)")
    axes[0].set_xlabel("средняя награда за последние 50 эпизодов")
    axes[0].set_ylabel("число запусков")

    colors = ["#C0392B", "#3170A6", "#2E8B57"]
    for (name, values), color in zip(fp_rates.items(), colors):
        axes[1].plot(GROUP_SIZES, values, "o-", color=color, label=name)
    axes[1].axhline(0.05, ls="--", c="gray", lw=1, label="номинальные 5%")
    axes[1].set_xlabel("число запусков в группе")
    axes[1].set_ylabel("доля ложных выводов")
    axes[1].set_title("Ошибка I рода:\nразличия нет по построению")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for effect in EFFECTS_IN_SIGMA:
        axes[2].plot(GROUP_SIZES, power_curves[effect], "o-",
                     label=f"улучшение +{effect}$\\sigma$")
    axes[2].axhline(0.8, ls="--", c="gray", lw=1, label="мощность 80%")
    axes[2].set_xlabel("число запусков в группе")
    axes[2].set_ylabel("вероятность заметить эффект")
    axes[2].set_title("Мощность перестановочного теста")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "result.png"
    plt.savefig(out, dpi=150)
    print(f"\nГрафик сохранён: {out}")


if __name__ == "__main__":
    main()
