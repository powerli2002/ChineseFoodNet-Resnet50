import argparse

import torch
from PIL import Image
from torch import device
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.resnet_cbam import resnet50_cbam 

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

    # model = WideResNet(8, 8, 0.0, in_channels=3, labels=208).to(device)
    # model.load_state_dict(torch.load('model_data/model-26-2.1383.pt'))
    # model.eval()

    model = resnet50_cbam(pretrained = True).to(device)

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