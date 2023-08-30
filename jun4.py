def divide(x, y):
    if y == 0:
        raise ZeroDivisionError("Division by zero is not allowed")
    return x / y
divide(10, 0)

# try:
#     result = divide(10, 0)
# except ZeroDivisionError as e:
#     print("An error occurred:", e)


# try:
#     x = 10 / 0  # ZeroDivisionError 발생
# except ZeroDivisionError as e:
#     print(e)
#     raise ValueError("Custom error occurred") from e

