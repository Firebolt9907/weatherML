import torch
import pandas as pd
import re

df = pd.read_csv('weather.csv')

# newdf = df.drop(columns=null_columns)

# print(df.columns)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

filterDf = df[['DATE', 'HourlyDryBulbTemperature']].copy().dropna()

inputs = []
dates = pd.to_datetime(filterDf['DATE'], format='%Y-%m-%dT%H:%M:%S')
for hour in len(dates)
inputs.append(dates.dt.year)
inputs.append(dates.dt.month)
inputs.append(dates.dt.day)
inputs.append(dates.dt.hour)

print(len(inputs[0]))
exit()
layers = [4, 200, 800, 1500, 800, 400, 200, 1]

actualOutputs = [int(temp.replace("s", '')) if isinstance(temp, str) else temp for temp in
                 filterDf['HourlyDryBulbTemperature']]
predictedOutputs = []

tensor_thing = torch.FloatTensor(inputs)


class H(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 200)
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
        print(input)
        return input


def costFunction(predicted, expected):
    difference = predicted - expected
    return difference * difference


model = H()
output = model(tensor_thing)
print(output)

for i in range(0, inputs):
    inp = torch.FloatTensor(inputs[i])
    outp = torch.FloatTensor(actualOutputs[i])

    prediction = model(inp)
    print('cost: ' + costFunction(prediction, actualOutputs[i]))

# def deriv(x, y):
#
