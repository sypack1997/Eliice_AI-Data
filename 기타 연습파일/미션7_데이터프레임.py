import numpy as np
import pandas as pd

population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)

gdp_dict = {
    'korea': 169320000,
    'japan': 516700000,
    'china': 1409250000,
    'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)

country = pd.DataFrame({
    "population" : population,
    "gdp" : gdp
})

# loc 을 활용하여 'korea' 행의 데이터를 출력하세요.
print('=== Korea ===')
print(country.loc['korea'])
print('')
# loc 을 활용하여 'korea', 'japan', 'china' 행의 데이터를 출력하세요.
print('=== Korea, Japan, China ===')
print(country.loc['korea':'china']) # korea ~ china
print('')
# iloc 을 활용하여 'korea', 'japan', 'china' 행의 데이터를 출력하세요.
print('=== Korea, Japan, China ===')
print(country.iloc[ 0 : 3 ]) # 0-th ~ 2-nd -> [0:3]
print('')
# iloc 을 활용하여 모든 나라의 'gdp' 열의 데이터를 출력하세요. 
print('=== GDP ===')
print(country.iloc[:, 1 ]) # population는 0-th 열, gdp는 1-st 열
print('')
# 컬럼명을 활용하여 모든 나라의 'population' 데이터를 출력하세요. 
print('=== population ===')
print(country['population'])
print('')