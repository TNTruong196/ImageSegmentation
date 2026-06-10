from torchvision.datasets import OxfordIIITPet

train_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=True
)

test_dataset = OxfordIIITPet(
    root="./data",
    split="test",
    target_types="segmentation",
    download=True
)