import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(9, 200)
        self.layer2 = torch.nn.Linear(200, 800)
        self.layer3 = torch.nn.Linear(800, 1500)
        self.layer4 = torch.nn.Linear(1500, 3000)
        # commented out the largest layers due to them slowing down training drastically with no accuracy increase
        # self.layer5 = torch.nn.Linear(3000, 6000)
        # self.layer6 = torch.nn.Linear(6000, 3000)
        self.layer7 = torch.nn.Linear(3000, 1500)
        self.layer8 = torch.nn.Linear(1500, 800)
        self.layer9 = torch.nn.Linear(800, 200)
        self.layer10 = torch.nn.Linear(200, 1)
        self.Activate = torch.nn.ReLU()

    def forward(self, input):
        input = self.layer1(input)
        input = self.Activate(input)
        input = self.layer2(input)
        input = self.Activate(input)
        input = self.layer3(input)
        input = self.Activate(input)
        input = self.layer4(input)
        input = self.Activate(input)
        # input = self.layer5(input)
        # input = self.Activate(input)
        # input = self.layer6(input)
        # input = self.Activate(input)
        input = self.layer7(input)
        input = self.Activate(input)
        input = self.layer8(input)
        input = self.Activate(input)
        input = self.layer9(input)
        input = self.Activate(input)
        input = self.layer10(input)

        return input
