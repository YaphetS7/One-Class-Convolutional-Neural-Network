# encoding: utf-8
import torch
import torch.nn as nn
from torchvision import models


class OCN(nn.Module):
    def __init__(self, num_classes, batchSize=3, device='cuda:0'):

        backbone = models.mobilenet_v2(weights='IMAGENET1K_V2')
        super(OCN, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.backbone = nn.ModuleList(backbone.children())[:-1]
        self.backbone = nn.Sequential(*self.backbone)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


        self.classifier = nn.Sequential(nn.Linear(1280, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, self.num_classes))

        self.relu = nn.ReLU(inplace=True)

        self.labels = torch.zeros((batchSize,)).long().to(self.device)

    def forward(self, x, labels, is_train=True):
        
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if is_train:
            gaussian = torch.normal(0, 0.1, x.shape)
            if x.is_cuda:
                gaussian = gaussian.to(self.device)
                self.labels = self.labels.to(self.device)

            labels = torch.cat((labels, self.labels), dim=0)
            x = torch.cat((x, gaussian), dim=0)
            x, labels = self.shuffle(x, labels)

            x = self.relu(x)

        x = self.classifier(x)

        return x, labels

    @staticmethod
    def shuffle(img, labels):
        shuffle = torch.randperm(img.shape[0])
        img = img[shuffle, ...]
        labels = labels[shuffle]
        return img, labels


# if __name__ == "__main__":
#     model = OCN(num_classes=2).cuda()
#     ins = torch.randn((3, 3, 224, 224)).cuda()
#     model(ins, torch.zeros((3,)).long().cuda(), True)