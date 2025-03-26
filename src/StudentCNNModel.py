import torch

class StudentCNNModel(torch.nn.Module):
    def __init__(self):
        super(StudentCNNModel, self).__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*64*64, out_features=1024),  # Updated in_features
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=1024, out_features=64*64*64)
        )

        self.upsample = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        self.final_conv = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sequence(x)
        x = x.view(x.size(0), -1)  # Flattening the tensor
        x = self.classifier(x)
        x = x.view(x.size(0), 64, 64, 64)
        x = self.upsample(x)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x