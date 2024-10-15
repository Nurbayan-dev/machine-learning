# Учится принимать последовательность решений для получения максимального вознаграждения.


import gym                                   # библиотека для создания различных сред, включая "Frozen Lake"
import numpy as np
import matplotlib.pyplot as plt                                                       # используется для построения графиков

# Создаем среду Frozen Lake
env = gym.make("FrozenLake-v1", is_slippery=True)                # агенттин S тен G ге чейинки жолун. H ден качуу

# Параметры Q-Learning
alpha = 0.1  # скорость обучения
gamma = 0.99  # коэффициент дисконтирования
epsilon = 1.0  # вероятность случайного действия (epsilon-greedy)
epsilon_decay = 0.995  # понижение epsilon после каждого эпизода
min_epsilon = 0.01
episodes = 10000  # количество эпизодов

# Инициализируем Q-таблицу
q_table = np.zeros([env.observation_space.n, env.action_space.n])  # Q-таблица инициализируется нулями.
# Это таблица, строки - состояния агента, столбцы — возможные действия.

# Списки для хранения значений для графика
epsilon_history = []
success_history = []

# Для подсчета успехов
successes = 0

# ε-greedy политика
def choose_action(state):                                         # функция выбора действия
    if np.random.uniform(0, 1) < epsilon:                  # epsilon - выбирается случайное действие
        return env.action_space.sample()  # случайное действие
    else:
        return np.argmax(q_table[state])  # действие с наивысшим значением Q

# Q-Learning алгоритм   - Основной цикл обучения
for episode in range(episodes):        # В каждом эпизоде среда сбрасывается, и агент начинает с позиции старта.

    state, _ = env.reset()
    state = int(state)
    done = False                      # устанавливает начальное условие, пока эпизод не завершен.

    while not done:
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)    # выполняем шаг
        next_state = int(next_state)
        done = done or truncated

        # Обновляем Q-таблицу
        q_table[state][action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )


        # gamma * np.max(q_table[next_state]) - оценка будущих наград
        # alpha - это скорость обучения

        # Переход в следующее состояние
        state = next_state

    # Если агент достиг цели, увеличиваем счетчик успехов
    if reward == 1:
        successes += 1

    # Понижаем epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epsilon_history.append(epsilon)

    # Сохраняем частоту успехов
    success_rate = successes / (episode + 1)
    success_history.append(success_rate)

    # Вывод прогресса каждые 100 эпизодов
    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode + 1}, Epsilon: {epsilon:.3f}, Success Rate: {success_rate:.3f}"
        )

# Проверка обученного агента
state, _ = env.reset()
state = int(state)
done = False
env.render()  # отобразим начальное состояние

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, truncated, info = env.step(action)
    next_state = int(next_state)
    done = done or truncated
    env.render()  # отображаем текущее состояние
    state = next_state

env.close()

# Построение графиков
plt.figure(figsize=(12, 5))

# График значения epsilon
plt.subplot(1, 2, 1)
plt.plot(epsilon_history, label="Epsilon", color="blue")
plt.title("Epsilon Decay Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Epsilon")
plt.legend()

# График частоты успехов
plt.subplot(1, 2, 2)
plt.plot(success_history, label="Total Reward", color="green")
plt.title("total reward over episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()

plt.tight_layout()
plt.show()
