import numpy as np
import matplotlib.pyplot as plt

# 카페인 데이터
coffee = np.array([202,177,121,148,89,121,137,158])

fig, ax = plt.subplots()

"""
1. 히스토그램을 그리는 코드를 작성해 주세요
"""
plt.hist(coffee)



# 히스토그램을 출력합니다.
plt.show()
fig.savefig("hist_plot.png")

