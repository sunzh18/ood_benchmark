p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

# 以上内容是模拟数据 请忽略
# 画图部分开始
plt.figure(figsize=(10, 6))

plt.plot(p, auc, 'o-', color='red', label='ours')
# plt.plot(p, sota, '--', color='blue', label='LINe')

# sota_mean = np.mean(sota)
sota_mean = 0.95
plt.axhline(y=sota_mean, color='blue', linestyle='--', label='LINe')

# Add the y-coordinate of the intersection with the y-axis for the green dashed line
plt.text(0, sota_mean, f'{sota_mean:.2f}', color='green', va='bottom', ha='right')


plt.xlabel('Pruning percentile')
plt.ylabel('AUROC')

# plt.plot(fpr95, color='red', label='FPR95')
# plt.ylabel('FPR95')

# ax2.plot(p, fpr95, 's-', color='red', label='FPR95')
# ax2.set_ylabel('FPR95')

# plt.xlim(0, 1.0)
plt.xticks(p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


# plt.title('Training and Validation Accuracy over Epochs')
plt.legend(loc='lower right')
plt.grid(True)