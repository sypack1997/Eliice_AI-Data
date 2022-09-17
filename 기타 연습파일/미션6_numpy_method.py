import numpy as np

# np.arange(N): [0,1,2, ..., N-1] 의 배열을 반환합니다.
arr1 = np.arange(10) # [0,1,2, ..., 9] 배열을 만들어보세요.
print(arr1)

# np.zeros(N): [0,0,0, ..., 0] 길이가 N이고, 원소가 모두 1인 배열을 반환합니다.
arr2 = np.zeros(10) # 길이가 10이고, 원소가 모두 0인 배열을 만들어보세요.
print(arr2)

# np.ones(N): [1,1,1, ..., 1] 길이가 N이고, 원소가 모두 1인 배열을 반환합니다.
arr3 = np.ones(10) # 길이가 10이고, 원소가 모두 1인 배열을 만들어보세요.
print(arr3)

# np.sum(arr): 배열 arr 속 원소의 합을 반환합니다.
arr1_sum = np.sum(arr1) # arr1 속 모든 원소들의 합을 구해보세요.
print(arr1_sum)

# np.mean(arr): 배열 arr 속 원소의 평균을 반환합니다.
arr1_mean = np.mean(arr1) # arr1 속 모든 원소들의 평균을 구해보세요.
print(arr1_mean)

# np.std(arr): 배열 arr 속 원소의 표준편차를 반환합니다.
arr1_std = np.std(arr1) # arr1 속 모든 원소들의 표준편차를 구해보세요.
print(arr1_std)

# arr.reshape(N1, N2): numpy array인 arr의 shape을 [N1, N2]로 바꿔줍니다.
arr1_reshaped = arr1.reshape(2, 5) # arr1 의 shape을 (2, 5) 로 바꿔보세요.
print(arr1_reshaped.shape)

# 주어진 numpy 배열 given_arr 을 오름차순 정렬해보세요. 
given_arr = np.asarray([1, 8, 3, 4, 6, 2, 5, 7, 9])
sorted_arr = np.sort(given_arr)
print(sorted_arr)