import time

def timer_decorator(func):          # принимает функцию
    def wrapper(*args, **kwargs):   # обертка, которая будет вызвана вместо func
        start = time.time()          # 1. делаем что-то ДО
        result = func(*args, **kwargs) # вызываем исходную функцию
        end = time.time()             # 2. делаем что-то ПОСЛЕ
        print(f"Время: {end - start}")
        return result                 # возвращаем результат исходной функции
    return wrapper                    # возвращаем обертку

# Применяем декоратор
@timer_decorator
def slow_function():
    time.sleep(1)
    return print("Готово")

slow_function()