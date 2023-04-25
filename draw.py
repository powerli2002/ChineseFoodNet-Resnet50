import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('drawpic.xls')


# 提取需要绘制的数据
epochs = df['epoch']
train_loss = df['train_loss']
valid_loss = df['valid_loss']
train_acc = df['train_acc']
valid_acc = df['valid_acc']


# 绘制折线图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(epochs, train_loss, 'b', label='Training Loss')
ax1.plot(epochs, valid_loss, 'r', label='Validation Loss')
ax1.legend(loc='best')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.plot(epochs, train_acc, 'b', label='Training Accuracy')
ax2.plot(epochs, valid_acc, 'r', label='Validation Accuracy')
ax2.legend(loc='best')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_ylim([0, 1.0])  # 设置y轴范围为[0, 1.0]
plt.show()
