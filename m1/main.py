import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class StoneFlightSimulator:
    def __init__(self):
        # Параметры по умолчанию
        self.g = 9.81  # ускорение свободного падения, м/с²
        self.rho = 1.225  # плотность воздуха, кг/м³
        self.Cd_linear = 0.1  # коэффициент линейного сопротивления
        self.Cd_quadratic = 0.47  # коэффициент квадратичного сопротивления
        self.A = 0.01  # площадь поперечного сечения камня, м²
        self.m = 0.1  # масса камня, кг
        
        # Начальные условия
        self.v0 = 20.0  # начальная скорость, м/с
        self.angle = 45.0  # угол броска, градусы
        self.drag_model = 'quadratic'  # модель сопротивления
        
        # Результаты
        self.solution = None
        
    def drag_force_linear(self, v):
        """Линейная модель сопротивления: F = -k·v"""
        return -self.Cd_linear * v
    
    def drag_force_quadratic(self, v):
        """Квадратичная модель сопротивления: F = -0.5·ρ·Cd·A·v²"""
        return -0.5 * self.rho * self.Cd_quadratic * self.A * v * abs(v)
    
    def equations_of_motion(self, t, y):
        """Уравнения движения камня"""
        x, y_pos, vx, vy = y
        
        # Скорость
        v = np.sqrt(vx**2 + vy**2)
        
        # Выбор модели сопротивления
        if self.drag_model == 'linear':
            F_drag_x = self.drag_force_linear(v) * (vx/v) if v > 0 else 0
            F_drag_y = self.drag_force_linear(v) * (vy/v) if v > 0 else 0
        else:  # quadratic
            F_drag_x = self.drag_force_quadratic(v) * (vx/v) if v > 0 else 0
            F_drag_y = self.drag_force_quadratic(v) * (vy/v) if v > 0 else 0
        
        # Уравнения движения
        dxdt = vx
        dydt = vy
        dvxdt = F_drag_x / self.m
        dvydt = -self.g + F_drag_y / self.m
        
        return [dxdt, dydt, dvxdt, dvydt]
    
    def solve_trajectory(self, t_max=10.0):
        """Решить уравнения движения"""
        # Начальные условия
        angle_rad = np.radians(self.angle)
        vx0 = self.v0 * np.cos(angle_rad)
        vy0 = self.v0 * np.sin(angle_rad)
        y0 = [0, 0, vx0, vy0]
        
        # Событие для определения момента падения
        def hit_ground(t, y):
            return y[1]
        hit_ground.terminal = True
        hit_ground.direction = -1
        
        # Решение ОДУ
        self.solution = solve_ivp(
            self.equations_of_motion,
            [0, t_max],
            y0,
            events=hit_ground,
            max_step=0.01,
            rtol=1e-6,
            atol=1e-8
        )
        
        return self.solution
    
    def analytical_solution_no_drag(self, t):
        """Аналитическое решение без сопротивления воздуха"""
        angle_rad = np.radians(self.angle)
        x = self.v0 * np.cos(angle_rad) * t
        y = self.v0 * np.sin(angle_rad) * t - 0.5 * self.g * t**2
        return x, y
    
    def get_landing_point(self):
        """Получить точку падения"""
        if self.solution is None:
            return None
        
        t_land = self.solution.t[-1]
        x_land = self.solution.y[0][-1]
        y_land = self.solution.y[1][-1]
        
        return x_land, y_land, t_land
    
    def plot_results(self):
        """Построить графики результатов"""
        if self.solution is None:
            print("Сначала выполните solve_trajectory()")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Траектория
        ax1.plot(self.solution.y[0], self.solution.y[1], 'b-', label='Численное решение')
        ax1.set_xlabel('Расстояние, м')
        ax1.set_ylabel('Высота, м')
        ax1.set_title('Траектория полета камня')
        ax1.grid(True)
        
        # Для сравнения: решение без сопротивления воздуха
        t_analytical = np.linspace(0, self.solution.t[-1], 100)
        x_analytical, y_analytical = self.analytical_solution_no_drag(t_analytical)
        ax1.plot(x_analytical, y_analytical, 'r--', label='Без сопротивления воздуха')
        
        # Точка падения
        x_land, y_land, t_land = self.get_landing_point()
        ax1.plot(x_land, y_land, 'ro', markersize=8, label=f'Падение: ({x_land:.2f} м, {y_land:.2f} м)')
        
        ax1.legend()
        ax1.axis('equal')
        
        # Скорости
        v = np.sqrt(self.solution.y[2]**2 + self.solution.y[3]**2)
        ax2.plot(self.solution.t, v, 'g-')
        ax2.set_xlabel('Время, с')
        ax2.set_ylabel('Скорость, м/с')
        ax2.set_title('Зависимость скорости от времени')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Вывод информации о полете
        print(f"Начальная скорость: {self.v0} м/с")
        print(f"Угол броска: {self.angle}°")
        print(f"Модель сопротивления: {self.drag_model}")
        print(f"Время полета: {t_land:.3f} с")
        print(f"Дальность полета: {x_land:.3f} м")
        print(f"Максимальная высота: {np.max(self.solution.y[1]):.3f} м")

# Функция для исследования влияния параметров
def parameter_study():
    simulator = StoneFlightSimulator()
    
    # Исследование влияния угла броска
    angles = np.linspace(15, 75, 7)
    ranges = []
    
    print("Исследование влияния угла броска:")
    for angle in angles:
        simulator.angle = angle
        simulator.v0 = 20.0
        simulator.drag_model = 'quadratic'
        simulator.solve_trajectory()
        x_land, _, _ = simulator.get_landing_point()
        ranges.append(x_land)
        print(f"Угол {angle:.1f}°: дальность = {x_land:.2f} м")
    
    # Исследование влияния начальной скорости
    print("\nИсследование влияния начальной скорости:")
    velocities = np.linspace(10, 50, 5)
    for v in velocities:
        simulator.angle = 45.0
        simulator.v0 = v
        simulator.solve_trajectory()
        x_land, _, _ = simulator.get_landing_point()
        print(f"Скорость {v:.1f} м/с: дальность = {x_land:.2f} м")
    
    # Сравнение моделей сопротивления
    print("\nСравнение моделей сопротивления:")
    simulator.angle = 45.0
    simulator.v0 = 30.0
    
    simulator.drag_model = 'none'
    simulator.solve_trajectory()
    x_no_drag = simulator.get_landing_point()[0]
    
    simulator.drag_model = 'linear'
    simulator.solve_trajectory()
    x_linear = simulator.get_landing_point()[0]
    
    simulator.drag_model = 'quadratic'
    simulator.solve_trajectory()
    x_quadratic = simulator.get_landing_point()[0]
    
    print(f"Без сопротивления: {x_no_drag:.2f} м")
    print(f"Линейное сопротивление: {x_linear:.2f} м")
    print(f"Квадратичное сопротивление: {x_quadratic:.2f} м")

# Пример использования
if __name__ == "__main__":
    # Создаем симулятор
    simulator = StoneFlightSimulator()
    
    for i in range(20, 80, 10):
        # Настраиваем параметры
        simulator.v0 = 25.0  # м/с
        simulator.angle = float(i)  # градусы
        simulator.drag_model = 'linear'  # 'linear' или 'quadratic'
        
        # Рассчитываем траекторию
        simulator.solve_trajectory()
        
        # Отображаем результаты
        simulator.plot_results()
    
    # Проводим исследование параметров
    parameter_study()