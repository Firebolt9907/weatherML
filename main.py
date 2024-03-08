import torch
import pandas as pd
from sklearn.utils import shuffle
import re

df = pd.read_csv('weather.csv')

LEARN_RATE = 0.00025

print(1)

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

filterDf = df[['DATE', 'HourlyDryBulbTemperature']].dropna()

print(2)

filterDf = filterDf.iloc[-1:1000]

print(3)

inputs = []
dates = pd.to_datetime(filterDf['DATE'], format='%Y-%m-%dT%H:%M:%S')
for date in dates:
    inputs.append([date.year, date.month,
                   date.day, date.hour, date.minute])

print(4)

# layers = [4, 200, 800, 1500, 800, 400, 200, 1]

actualOutputs = [int(temp.replace("s", '')) if isinstance(
    temp, str) else temp for temp in filterDf['HourlyDryBulbTemperature']]
predictedOutputs = []

print(5)

tensor_thing = torch.FloatTensor(inputs).to(device)

print(6)

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
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

print(7)

output = torch.FloatTensor(actualOutputs).to(device)

print(8)

for i in range(len(inputs)):
    print(9)
    inp = torch.FloatTensor(inputs[i]).to(device)

    prediction = model(inp)
    loss = loss_function(prediction, output[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(10)
    print(
        f"current loss at day {str(inputs[i][1])}/{str(inputs[i][2])}/{str(inputs[i][0] % 100)} at {str(inputs[i][3])}:{str(inputs[i][4])}, iteration {str(i)} loss: {str(loss.item())}, predicted: {str(prediction.item())}, actual: {str((output[i].item()))}")

torch.save(model.state_dict(), 'model.pt')
