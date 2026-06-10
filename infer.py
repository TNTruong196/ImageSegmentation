import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from model import UNet
from dataset import PetDataset, val_transform

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model và nạp Trọng số đã học
    model = UNet(num_classes=3, depth=5, merge_mode='concat')
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. Lấy dữ liệu test
    raw_val = OxfordIIITPet(root="./data", split="test", target_types="segmentation", download=True)
    val_dataset = PetDataset(raw_val, transform=val_transform)

    # Lấy thử 1 mẫu bất kỳ (ví dụ mẫu thứ 10)
    image, mask = val_dataset[10]

    # 3. Dự đoán (Inference)
    # Thêm chiều batch_size: (3, H, W) -> (1, 3, H, W)
    input_tensor = image.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor) # shape: (1, 3, H, W)
        output = output.squeeze(0)   # shape: (3, H, W)
        pred_mask = torch.argmax(output, dim=0).cpu().numpy() # shape: (H, W), chứa các nhãn 0, 1, 2

    # 4. Trực quan hóa kết quả
    # Chuẩn bị ảnh để hiển thị (đưa từ Tensor chuẩn hóa về ảnh RGB thông thường)
    img_show = image.permute(1, 2, 0).cpu().numpy()
    # Đảo ngược chuẩn hóa để ảnh trông tự nhiên hơn
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_show = std * img_show + mean
    img_show = np.clip(img_show, 0, 1)

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_show)
    axes[0].set_title("Ảnh Gốc (Input)")
    axes[0].axis("off")

    axes[1].imshow(mask.cpu().numpy(), cmap="viridis")
    axes[1].set_title("Mặt Nạ Thực Tế (Ground Truth)")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="viridis")
    axes[2].set_title("Mặt Nạ Dự Đoán (Prediction)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
