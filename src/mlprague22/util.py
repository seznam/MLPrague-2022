import os


def mount_gdrive(path):
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        return f"/content/gdrive/MyDrive/{path}", True
    except:
        # init local CUDA device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return "..", False
