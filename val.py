import pandas
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from utility.initialize import initialize
from model.wide_res_net import WideResNet
from random import random

if __name__ == '__main__':
    # model = WideResNet(8, 8, 0.0, in_channels=3, labels=208)
    # model.load_state_dict(torch.load('model_data/model-26-2.1383.pt'))
    # model.eval()


    classname = pandas.read_excel('class_names.xls')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = torchvision.models.resnet50(pretrained=False).to(device)

    # checkpoint = torch.load('model_data/resmodel-23-4000-0.4873.pt')
    checkpoint = torch.load('model_data/resmodel-32-1.069.pt')
    model.load_state_dict(checkpoint['model'])
    model.eval() # 制定model.eval()固定dropout和BN层。
    img = Image.open('./ChineseFoodNet/release_data/train/000/000001.jpg').convert('RGB')
    # img = Image.open('dataset/000000.jpg').convert('RGB')
    img = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    ])(img)
    img = img.reshape(1, 3, 128, 128).to(device)
    pre = model(img)
    _, predicted = torch.max(pre, 1)
    # print(pre)
    print("predicted"+str(predicted))
    correct = torch.argmax(pre.data, 1)
    _, predicted_top3 = pre.topk(3, dim=1)
    # print(correct)
    # for i in range(pre.shape[1]):
    #     print("Class {}: {:.2f}%".format(i, pre[0][i]*100))
    i = correct.item()
    print("Class {}: {:.2f}%".format(i, pre[0][i]*10))

    print(correct.item())
    pretop3 = predicted_top3.tolist()
    print(classname["chname"][correct.item()])
    print(classname["chname"][pretop3[0][1]])




# 这个脚本加载一个预先训练好的WideResNet模型，并使用它对输入图像进行预测。
#
# WideResNet体系结构在model/wide_res_net.py文件中定义，并使用参数8,8,0.0,in_channels=3和labels=208实例化。这意味着该网络的深度为8，扩大系数为8，并被设计为将图像分为208个类别之一。
#
# 使用模型对象的load_state_dict方法从model.pt文件加载预训练的模型权重。然后利用评价方法将模型置于评价模式。
#
# 使用PIL库加载输入图像，并使用torchvision库的transforms模块对其应用一系列转换。这些转换包括将图像的中心裁剪为500x500像素，将图像大小调整为128x128像素，将图像转换为PyTorch张量，并使用代码中提供的平均值和标准差值规范化像素值。
#
# 得到的张量被重塑为批处理大小为1，并使用模型对象在模型中传递。模型的输出是一个形状张量(1,208)，包含输入图像的预测类概率。然后使用PyTorch的argmax函数计算这个张量中最大值的索引，它给出了预测的类标签。
#
# 最后，预测的类标签被打印到控制台。注意，print语句只显示了预测概率的张量，而不是相应的类标签。
