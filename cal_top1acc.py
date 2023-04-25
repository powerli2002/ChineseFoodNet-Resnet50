import argparse

import torch
from PIL import Image
from torch import device
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model.wide_res_net import WideResNet
from utility.initialize import initialize
from utils.ChineseFoodNetSet import ChineseFoodNetTestSet, ChineseFoodNetTrainSet
from utility.log import Log

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=8, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WideResNet(8, 8, 0.0, in_channels=3, labels=208).to(device)
    # model.load_state_dict(torch.load('model_data/model-26-2.1383.pt'))
    # model.eval()

    checkpoint = torch.load('model_data/resmodel-23-1.085.pt')
    model.load_state_dict(checkpoint['model'])

    dataset_test = ChineseFoodNetTestSet()
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.threads)

    sum_correct = 0
    # with torch.no_grad:
    for batch in dataloader_test :
        inputs, targets = (b.to(device) for b in batch)
        predictions = model(inputs)
        # loss = smooth_crossentropy(predictions, targets)
        correct = torch.argmax(predictions, 1) == targets
        sum_correct += correct.cpu().sum().item()
            # print(correct.cpu().sum().item())

    print(sum_correct / len(dataset_test))

    # img = Image.open('./ChineseFoodNet/release_data/test/000012.jpg').convert('RGB')
    # # img = Image.open('dataset/000000.jpg').convert('RGB')
    # img = transforms.Compose([
    #     transforms.CenterCrop(500),
    #     transforms.Resize(128),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    # ])(img)
    # img = img.reshape(1, 3, 128, 128)
    # pre = model(img)
    # print(pre)
    # correct = torch.argmax(pre, 1)
    # print(correct)
    # print(correct.item())


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
