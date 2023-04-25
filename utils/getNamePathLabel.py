import pandas as pd


def getClassName(xlsx_path=r'../ChineseFoodNet/class_names.xlsx'):
    """
    函数从提供的Excel文件中读取菜名称和索引的对应值（包含中文名和英语名）
    :param xlsx_path: Excel文件的位置
    :return: 一个208*3的list，list中每个元素对应一个种类的菜，包含索引、中文名和英语名
    """
    # 读取excel文件，因为文件中没有表头，所以读取的时候要去掉表头
    xlsx = pd.read_excel(xlsx_path, sheet_name='Sheet1', header=None)

    # 获得文件的行数和列数
    row_num = xlsx.shape[0]
    col_num = xlsx.shape[1]

    name_list = []
    for row in range(row_num):
        temp_list = []
        for col in range(col_num):
            temp_list.append(xlsx.iloc[row, col])
        # name_list.append([xlsx.iloc[row, 0], xlsx.iloc[row, 1], xlsx.iloc[row, 2]])
        name_list.append(temp_list)

    return name_list


def getPathLabel(sets='train_sets'):
    """
    函数从划分好的三个集合中获取图像对应的文件路径和标签
    :param sets: 需要获取的集合名字，枚举型，默认为训练集
    :return:一个字典包含两个字段：Path和label，每个字段都是一个可索引的列表
    """
    # 集合列表目录
    train_path = r'./ChineseFoodNet/release_data/train_list.txt'
    test_path = r'./ChineseFoodNet/release_data/test_truth_list.txt'
    val_path = r'./ChineseFoodNet/release_data/val_list.txt'

    # 按照需求读取文件，获得图像位置和标签，同样去掉表头
    if sets == 'train_sets':
        text = pd.read_csv(train_path, sep=' ', names=['path', 'label'], header=None)
    elif sets == 'test_sets':
        text = pd.read_csv(test_path, sep=' ', names=['path', 'label'], header=None)
    elif sets == 'val_sets':
        text = pd.read_csv(val_path, sep=' ', names=['path', 'label'], header=None)
    else:
        raise ValueError('请输入有效集: 训练集train_sets, 测试集test_sets, 验证集val_sets')

    # trains_path = list(text['path'])
    # trains_label = list(text['label'])
    return text


if __name__ == '__main__':
    x_path = r'../ChineseFoodNet/class_names.xlsx'
    names = getClassName(x_path)
    print(names)
    print(names[105])
    trains = getPathLabel('val_sets')
    print(trains['path'][0])
    print(trains['label'])
