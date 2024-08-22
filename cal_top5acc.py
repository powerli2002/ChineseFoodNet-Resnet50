import argparse

import torch
import torchvision
from PIL import Image
from torch import device
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model.wide_res_net import WideResNet
from utility.initialize import initialize
from utils.ChineseFoodNetSet import ChineseFoodNetTestSet, ChineseFoodNetTrainSet, ChineseFoodNetValSet
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
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet50(pretrained=True).to(device)
    # model = WideResNet(8, 8, 0.0, in_channels=3, labels=208).to(device)
    # model.load_state_dict(torch.load('model_data/model-26-2.1383.pt'))
    # model.eval()

    checkpoint = torch.load('model_data/resmodel-32-1.069.pt')
    model.load_state_dict(checkpoint['model'])

    dataset_test = ChineseFoodNetTestSet()
    # dataset_test = ChineseFoodNetTrainSet()


    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.threads)

    sum_correct = 0

    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data in dataloader_test:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = outputs.topk(3, dim=1)

            total += targets.size(0)
            correct_top1 += (predicted == targets).sum().item()
            correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    print(f'Top 1 Accuracy: {top1_acc:.2f}%, Top 5 Accuracy: {top5_acc:.2f}%')
