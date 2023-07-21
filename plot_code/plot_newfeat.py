dim = 128
in_class_mean = np.random.rand(dim)*3
out_class_mean = np.random.rand(dim)*3
fc = np.random.rand(dim)
# w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # replace with your actual data

out_dataset = "iSUN"

# sort indices by the weights
sorted_indices = np.argsort(fc)
sorted_indices = sorted_indices[::-1]  # reverse the order

# sort v1 and v2 according to the sorted indices
in_sorted = in_class_mean[sorted_indices]
out_sorted = out_class_mean[sorted_indices]

# plot
plt.figure(figsize=(10, 6))
plt.ylim(0, 6)

plt.bar(np.arange(len(in_sorted)), in_sorted, color='red', label='v1', alpha=0.5)
plt.bar(np.arange(len(out_sorted)), out_sorted, color='green', label='v2', alpha=0.5)

plt.xlabel('Channel')
plt.ylabel('Activation')

ax = plt.gca().twinx()

plt.plot(fc[sorted_indices], color='blue', label='w')
plt.ylabel('Classifer weight')

# 同时显示左右两个图例
lines, labels = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2)

import matplotlib.patches as mpatches
id_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black')
ood_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black')

# 添加右上角的label
plt.legend([id_patch, ood_patch], ['ID:CIFAR-100', f'OOD:{out_dataset}'], loc='upper right')

plt.grid(True)

plt.show()