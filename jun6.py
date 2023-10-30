import numpy as np


def process_array(arr):
    # 3번째, 5번째, 7번째 차원 삭제
    arr = np.delete(arr, [2, 4, 6], axis=2)

    # 4번째와 6번째 차원을 OR 연산하여 하나로 묶기
    combined_dim = np.logical_or(arr[:, :, 2], arr[:, :, 3])

    # 결과 배열 생성 (10, 10, 3)
    result = np.dstack((arr[:, :, :2], combined_dim))

    return result


# 예제 배열 생성 (10, 10, 7)
original_array = np.random.randint(0, 2, size=(10, 10, 7))

# 함수를 사용하여 배열 처리
result_array = process_array(original_array)

# 결과 확인
print("원래 배열 형태:", original_array.shape)
print("처리된 배열 형태:", result_array.shape)
