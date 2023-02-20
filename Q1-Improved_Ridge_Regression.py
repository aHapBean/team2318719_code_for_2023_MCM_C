import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import math
plt.rcParams["font.family"] = "serif"
"""
量化单词模型
单词特性：
1、元音占比 比例作为数值 -> 
(1)0.5
(2)0.8
(3)0.4
(4)0.1
(5)0
2、是否有重复字母 0 / 1
3、单词出现频率 sort所有单词 -> (sum - rank) / sum 回归项
4、单词的开头是否为某些特定字母 -> 10 个 yes -> 1 no -> 0 ((0.2 + _ + _ + _ + _ ) / 5
5、与前一天单词的相似度 (评价
6. 时间项
"""
xlsx = pd.read_excel('Q1_attribute_pre_move.xlsx')
numerator = xlsx['Number in hard mode'].values.copy()   # 分子
denominator = xlsx['Number of reported results'].values.copy() # 分母
words = xlsx['Word'].values.copy()

rate = []
for i in range(len(numerator)):
    rt = numerator[i] / denominator[i]
    rate.append(rt)
# 输出各项值

# 拆分自变量和因变量

X = xlsx[['vowel rate','is repeat','rate grade','specific','similarity','index','index_sq']]
"""
[ 2.22397900e-02  2.58455076e-03  9.02921343e-04 -1.13428384e-03
 -3.59479695e-03 -4.90272059e-05  4.81417189e-04 -7.89760477e-07]
"""
y = rate.copy()

X = np.array(X) # numpy !
# 添加常数项
# X = np.hstack([np.ones((len(X),1)),X])    # 添加项
X = np.hstack([np.ones((len(X), 1)), X])

# 岭回归
alpha = 0.4
XTX = np.dot(X.T,X)
XTy = np.dot(X.T,y)
w = np.dot(np.linalg.inv(XTX + alpha * np.eye(X.shape[1])), XTy)

y_pred = np.dot(X,w)
residuals = y - y_pred

plt.plot(xlsx['index'].values,y_pred,label="predict")
plt.plot(xlsx['index'].values,rate,label="real")

x_ticks = xlsx['index'][::20]
plt.xticks(x_ticks,fontsize=16)
plt.ylabel("the percent of participants in hard mode",fontsize=20)
plt.xlabel("index",fontsize=20)
plt.yticks(fontsize=16)
plt.legend(fontsize=20)
plt.show()
# 输出拟合参数
print(w)


