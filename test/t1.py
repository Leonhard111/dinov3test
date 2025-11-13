# /home/leonhard/workshop/dinov3/dinov3-main/pth/
import torch

REPO_DIR = "/home/leonhard/workshop/dinov3/dinov3-main"

# DINOv3 ViT models pretrained on web images
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights="/home/leonhard/workshop/dinov3/dinov3-main/pth/dinov3_vits16_pretrain_lvd1689m.pth")
