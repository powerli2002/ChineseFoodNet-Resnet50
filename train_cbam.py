import argparse
import torch
import torchvision

from torch.utils.data import DataLoader
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from utils.ChineseFoodNetSet import ChineseFoodNetTrainSet, ChineseFoodNetTestSet,ChineseFoodNetValSet
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from model.resnet50 import ResNet50
from model.resnet_cbam import resnet50_cbam 
import sys
import wandb
from datetime import datetime

sys.path.append("..")
from sam import SAM

# 是否加载模型
LOAD_MODEL = 0
# 模型路径
model_path = ""

IF_WANDB = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=8, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.9, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=90, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    if IF_WANDB:
        wandb.init(project="Chinesefood_classification", name = datetime.now().strftime('%Y-%m-%d_%H-%M'))

    config = dict (
        # dataset
        threads = args.threads,
        batch_size = args.batch_size,

        # optimizer
        learning_rate = args.learning_rate,        
        rho=args.rho, 
        adaptive=args.adaptive, 
        lr=args.learning_rate,
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )

    if IF_WANDB:
        wandb.config.update(config)

    initialize(args, seed=42)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("running on " + str(device))
    # device = torch.device("cpu")

    dataset_train = ChineseFoodNetTrainSet()
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.threads)
    dataset_val = ChineseFoodNetValSet()
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.threads)
    dataset_test = ChineseFoodNetTestSet()
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.threads)

    log = Log(log_each=1)

    model = resnet50_cbam(pretrained = True).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)


    # 加载模型
    if LOAD_MODEL:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['train_loss']

        print("load success!")
    else:
        epoch = 0
   

    log = Log(log_each=10, initial_epoch=epoch)
    tmp_flag = 1

    while epoch <= args.epochs:

        total_loss = 10
        model.train()
        log.train(len_dataset=len(dataloader_train), flag=tmp_flag)
        tmp_flag = 0
        times = 0
        total_train_correct = 0
        total_train_samples = 0
        total_train_loss = 0
        for batch in dataloader_train:
            times = times + 1
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()

            batch_loss = loss.sum().item()
            total_train_loss += batch_loss
            total_loss = loss
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

                # 显式计算准确率，以便记录
                batch_correct = correct.sum().item()
                steps = targets.size(0)
                total_train_correct += batch_correct
                total_train_samples += steps

                if times % 5 == 0:
                    if wandb.run is not None:
                        wandb.log({
                            "Batch Train Accuracy": batch_correct / steps,
                            "Batch Train Loss": batch_loss / steps
                        })

                

        train_accuracy = total_train_correct / total_train_samples
        train_loss = total_train_loss / total_train_samples
        

        if wandb.run is not None:
            # print("train info upload!")
            wandb.log({
                "Epoch": epoch,
                "Train Accuracy": train_accuracy,
                "Train Loss": train_loss
            },commit=False)
        else:
            print("wandb not running!")

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        model.eval()
        log.eval(len_dataset=len(dataset_test))

    
        with torch.no_grad():
            for batch in dataloader_val:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                total_loss = loss

                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

                total_val_loss += loss.sum().item()
                total_val_correct += correct.sum().item()
                total_val_samples += targets.size(0)


        val_accuracy = total_val_correct / total_val_samples  # 计算平均准确率
        val_loss = total_val_loss / total_val_samples

        if wandb.run is not None:
            wandb.log({
                "Val Accuracy": val_accuracy,
                "Val Loss": val_loss
            },commit=True)

        else:
            print("wandb not running!")

        
        if epoch % 5 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': total_loss
            }

            total_loss = round(log.epoch_state["loss"] / log.epoch_state["steps"], 4)
            name = './model_data/resmodel-' + str(epoch) + '-' + str(total_loss) + '.pt'
            torch.save(checkpoint, name)

        epoch = epoch + 1

    log.flush()
    wandb.finish()


# 数据增强

