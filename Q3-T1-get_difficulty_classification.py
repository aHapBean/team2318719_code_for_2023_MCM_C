from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"


"""
attri1: 0.03915504328196925 vowel rate
attri2: 0.11001328716979249 is repeated
attri3: 0.1777851215778241 rate grade(排名打分
attri4: 0.18989862542516342 specific(每个位置在题库中的占比
attri5: 0.017930583744217108 similarity(相似度定义1
attri6: 0.07192112975339136 distance(相似度定义2
我们认为随机森林的系数就是其对正确率影响的程度，所以根据以上数据，我们考虑去除影响因子最低的那个因素,同步扩大100倍，方便显示结果
"""
coef_ = np.array([3.915504328196925,
                  17.77851215778241,
                  18.989862542516342,   #
                  1.7930583744217108,   # similarity
                  7.192112975339136])

xlsx = pd.read_excel('Q1_attribute.xlsx')
_str = {}
_str[1] = '1 try'
_str[2] = '2 tries'
_str[3] = '3 tries'
_str[4] = '4 tries'
_str[5] = '5 tries'
_str[6] = '6 tries'
_str[7] = '7 or more tries (X)'

x_7 = [1, 2, 3, 4, 5, 6, 7]

attri1 = np.array(xlsx['vowel rate'].values)
attri2 = np.array(xlsx['rate grade'].values) # 抛掉了之前的is repeat
attri3 = np.array(xlsx['specific'].values)  # 特征值越大，说明越好猜
attri4 = np.array(xlsx['similarity'].values) #
attri6 = np.array(xlsx['distance'].values)
"""
attri1 = np.array(xlsx['vowel rate'].values)
attri2 = np.array(xlsx['is repeat'].values)
attri3 = np.array(xlsx['rate grade'].values)
attri4 = np.array(xlsx['specific'].values)
attri6 = np.array(xlsx['distance'].values)
"""
attr = np.array([attri1,
                 attri2,
                 attri3,
                 attri4,
                 attri6])

# difficulty 是负值 2.20已加负号
difficulty = -np.dot(coef_, attr)
# print(difficulty)

accuracy = np.array([xlsx[_str[1]].values,
                     xlsx[_str[2]].values,
                     xlsx[_str[3]].values,
                     xlsx[_str[4]].values,
                     xlsx[_str[5]].values,
                     xlsx[_str[6]].values,
                     xlsx[_str[7]].values]).T  # (359,5)
avg_accuracy = []
for i in range(len(accuracy)):
    sum = 0
    for j in range(7):
        sum += accuracy[i][j] * (j + 1)
    sum /= 100
    avg_accuracy.append(sum)
# plt.scatter(difficulty,avg_accuracy)
# plt.ylabel("the times of success")
# plt.show()

# changed tag
ini_point = np.array([difficulty, xlsx['index'].values]).T # (359, 2) tag
interval_nums = 6
length = 25 - (-5)
span = length / interval_nums

interval = []
pre = -5
cur = -5
for i in range(interval_nums):
    cur += span
    interval.append([pre,cur])
    pre = cur

colors = ['blue', 'orange', 'red', 'gray', 'magenta', 'cyan', 'yellow', 'gray', 'green', 'purple']

categories = [[] for _ in range(6)]
for (x,y) in ini_point:
    for (index,[i,j]) in enumerate(interval):
        if i <= x and x < j:
            categories[index].append([x,y])
            break
avg_x = []
avg_y = []
# point = []
gain_factor = [1.2,
               1.1,
               1.0,
               0.9,
               0.8,
               0.7] # 增益因子
gain_factor.reverse()

point = []
for i in range(len(categories)):
    # print(categories[i])
    x = [coord[0] for coord in categories[i]]
    y = [coord[1] for coord in categories[i]]
    avg_x.append(np.array(x).mean())
    avg_y.append(np.array(y).mean())
    for j in range(len(x)):
        point.append(np.array([x[j], y[j]]))
    plt.scatter(x,y,color=colors[i])
# 五类单词可视化
point = np.array(point)
plt.ylim([0, 700])
corr_matrix = np.corrcoef(point[:,0], point[:,1])
plt.ylabel("the index of time",fontsize=15)
plt.xlabel("the words' difficulty",fontsize=15) # 这里得更改78行那里的avg_accuracy
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
print(corr_matrix)
plt.show()

# 首先进行归一化
def nnormalize(tmp,Flag:bool=False):
    mean_y = np.mean(tmp[:, 1])    # 计算 y 坐标的平均值
    std_y = np.std(tmp[:, 1])    # 计算 y 坐标的标准差
    # 归一化 y 坐标
    if Flag:
        std_y += 0.1
    normalized_y = (tmp[:, 1] - mean_y) / std_y
    # 返回归一化后的结果，保持输入数组不变
    return np.column_stack((tmp[:, 0], normalized_y))
tmp1 = np.column_stack((xlsx['index'].values, ini_point[:, 0])) # difficulty
tmp2 = np.column_stack((xlsx['index'].values, ini_point[:, 1]))

tmp1 = nnormalize(tmp1)
tmp2 = nnormalize(tmp2,True)

# 单词难度与准确性关系图
# plt.plot(xlsx['index'].values,tmp1[:,1],label="the words' difficulty")
# plt.plot(xlsx['index'].values,tmp2[:,1],label="the average time needed to success")
# plt.xlabel("the index of time",fontsize=21)
# plt.ylabel("the value after normalizing",fontsize=21)
# plt.legend(fontsize=21)
# plt.show()

# 难度分类图
xx = xlsx['index'].values
plt.scatter(xx,tmp1[:,1],label="the words' difficulty")
plt.ylabel("the index of time",fontsize=15)
plt.xlabel("the words' difficulty",fontsize=15)
plt.legend(fontsize=15)
plt.show()
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# 构造数据
np.random.seed(0)
x = point[:,0]
y = point[:,1]

# 创建一个 LinearRegression 对象
reg = LinearRegression()

# 将 x 作为自变量，y 作为因变量拟合线性回归模型
reg.fit(x.reshape(-1, 1), y)

# 打印出斜率和截距
print("Slope:", reg.coef_[0])
print("Intercept:", reg.intercept_)

# 用线性回归模型对数据进行预测
y_pred = reg.predict(x.reshape(-1, 1))

# 绘制数据点和回归线
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print('R-squared:', r2)
"""
此处是 定义的难度 与 当日正确所需次数的期望 的关系
难度 值 越高代表越容易写出来
x,y的相关性：
[[ 1.         -0.87347784]
 [-0.87347784  1.        ]]
拟合直线的数据：
Slope: -0.10369876561329937
Intercept: 2.955437155189226

R-squared（决定系数）用于评价线性回归模型的拟合程度，其取值范围为0到1，值越接近1表示模型的拟合程度越好。

在你提供的结果中，R-squared的值为0.76，意味着线性回归模型对数据的拟合程度相对较好，但仍有一定的误差存在。需要根据具体问题和应用场景来判断R-squared的值是否满足要求。

当没有加增益因子时
[[ 1.         -0.35531706]
 [-0.35531706  1.        ]]
Slope: -0.021634357630698323
Intercept: 3.958077653610849
R-squared: 0.1262502122058895
"""




