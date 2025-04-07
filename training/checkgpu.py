import torch
import torchvision
import torchaudio

def check_torch_setup():
    print(f"Torch version     : {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchaudio version : {torchaudio.__version__}")

    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"GPU name          : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (PyTorch compiled): {torch.version.cuda}")
    else:
        print("CUDA is NOT available. You're using CPU.")

if __name__ == "__main__":
    check_torch_setup()