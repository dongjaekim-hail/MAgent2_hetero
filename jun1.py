# try:
#     x = 10 / 0
# except ZeroDivisionError as e:
#     print("An error occurred:", e)

# #오류가 날 수 도 있는 것을 try 에 넣고, 만약 오류가 발생하지 않는다면 그냥 넘어가고, 오류가 발생한다면 except 문을 실행한다.
# #오류 메세지를 e로 인지하게 해서 원하는 간략한 형태의 출력을 가능하게 한다. (print문에서와 같이 말이다)
#
# #또한 다음과 같은 것도 가능하다.
# try:
#     x = 10 / 2
# except ZeroDivisionError as e:
#     raise ValueError("Custom error occurred") from e
#
# """
# raise는 파이썬에서 예외를 명시적으로 발생시키는데 사용되는 키워드입니다. 프로그래머가 특정한 상황에서 예외를 발생시키고 싶을 때 사용합니다.
# 이를 통해 코드의 특정 조건을 만족하지 않을 때 프로그램의 흐름을 제어하고 오류를 처리할 수 있습니다.
#
# raise의 기본적인 사용법은 다음과 같습니다:
# """
# raise Exception("Error message")

"""
위 코드에서 Exception 대신에 내장 예외 클래스 중 하나를 사용하여 원하는 종류의 예외를 발생시킬 수 있습니다. 또한 "Error message" 대신에 오류 메시지를 적절하게 작성할 수 있습니다. 이 메시지는 예외 정보를 설명하는 데 사용됩니다.
예를 들어, 다음과 같이 예외를 발생시키는 코드를 작성할 수 있습니다:
"""
# def divide(x, y):
#     if y == 0:
#         raise ZeroDivisionError("Division by zero is not allowed")
#     return x / y
#
# try:
#     result = divide(10, 0)
# except ZeroDivisionError as e:
#     print("An error occurred:", e)

try:
    x = int(input('3의 배수를 입력하세요: '))
    if x % 3 != 0:                            # x가 3의 배수가 아니면
        raise Exception('3의 배수가 아닙니다.')    # 예외를 발생시킴
    print(x)
except KeyError as e:                        # 예외가 발생했을 때 실행됨
    print('예외가 발생했습니다.', e)