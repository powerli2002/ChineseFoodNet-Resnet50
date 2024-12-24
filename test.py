import torch.cuda
from utils.ChineseFoodNetSet import ChineseFoodNetTrainSet, ChineseFoodNetTestSet
from PIL import Image
from torchvision.transforms import transforms
if __name__ == '__main__':
    # img = Image.open('./ChineseFoodNet/release_data/train/000/000085.jpg')

    # img = transforms.Compose([
    #     transforms.CenterCrop(500),
    #     transforms.Resize(128)
    # ])(img)
    # img.show()
    # epoch = 1
    # loss = 0.2
    # name = 'model ' + str(epoch) + str(loss)
    # print(name)

    import torch
    print(f"pytorch版本: {torch.__version__}")
    print(f"CUDA是否可用? {torch.cuda.is_available()}")
    print(f"当前CUDA 版本: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"当前CUDA ID:{torch.cuda.current_device()}")

    print(f"CUDA设备名称:{torch.cuda.get_device_name(cuda_id)}")


# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html