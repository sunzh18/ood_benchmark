import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 生成数据集
np.random.seed(42)
data = np.random.randn(100, 10)

# 将数据集转换为 DataFrame
df = pd.DataFrame(data)

# 计算每个维度的均值和方差
stats = df.describe().loc[['mean', 'std'], :]

# 将数据集转换为长格式
df_long = pd.melt(df, var_name='Dimension', value_name='Value')
print(df_long)
# 绘制统计图
sns.set(style='whitegrid')
# sns.swarmplot(x='Dimension', y='Value', data=df_long, color='blue')
sns.boxplot(x='Dimension', y='Value', data=df_long, color='orange', width=0.2)

plt.savefig('test.png',dpi=600)
    
plt.close()