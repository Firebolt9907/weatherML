import torch
import pandas as pd
import re

df = pd.read_csv('weather.csv')

LEARN_RATE = 0.00025

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

filterDf = df[['DATE', 'HourlyDryBulbTemperature']].dropna()

inputs = []
dates = pd.to_datetime(filterDf['DATE'], format='%Y-%m-%dT%H:%M:%S')
for date in dates:
    inputs.append([date.year, date.month,
                  date.day, date.hour, date.minute])

# layers = [4, 200, 800, 1500, 800, 400, 200, 1]
print(filterDf.head(8117))
exit()

actualOutputs = [int(temp.replace("s", '')) if isinstance(
    temp, str) else temp for temp in filterDf['HourlyDryBulbTemperature']]
predictedOutputs = []

tensor_thing = torch.FloatTensor(inputs).to(device)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 200)
        self.layer2 = torch.nn.Linear(200, 800)
        self.layer3 = torch.nn.Linear(800, 1500)
        self.layer4 = torch.nn.Linear(1500, 800)
        self.layer5 = torch.nn.Linear(800, 200)
        self.layer6 = torch.nn.Linear(200, 1)
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
        input = self.layer5(input)
        input = self.Activate(input)
        input = self.layer6(input)
        return input



model = Model().to(device)
loss_function = torch.nn.HuberLoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

for i in range(len(inputs)):
    inp = torch.FloatTensor(inputs[i]).to(device)
    outp = torch.FloatTensor(actualOutputs[i]).to(device)

    prediction = model(inp)
    loss = loss_function(prediction, outp)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"current loss at day {str(inputs[i][1])}/{str(inputs[i][2])}/{str(inputs[i][0] % 100)} at {str(inputs[i][3])}:{str(inputs[i][4])}, iteration {str(i)}: {str(loss.item())}")
    if loss.item() < 400:
        LEARN_RATE = 0.000025
    elif loss.item() < 40:
        LEARN_RATE = 0.0000025

