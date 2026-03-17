import torch
from model import CropDiseaseModel
from data_loader import get_task_loaders
from ewc_utils import EWC


def main():
    loaders = get_task_loaders(batch_size=4, num_workers=0)
    train_loader = loaders['Spring'][0]
    model = CropDiseaseModel(num_classes=10, pretrained=False).to('cpu')
    ewc = EWC(model)

    images, labels = next(iter(train_loader))
    print("Images shape:", images.shape, "Labels shape:", labels.shape)
    logits = model(images)
    print("Logits shape:", logits.shape)
    loss, ce, pen = ewc.loss(logits, labels)
    print("Loss:", loss.item(), "CE:", ce.item(), "EWC penalty:", float(pen))
    loss.backward()
    print('SMOKE OK - forward and backward pass succeeded')


if __name__ == '__main__':
    main()
