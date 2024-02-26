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

tensors = []
inputs = pd.to_datetime(filterDf['DATE'], format='%Y-%m-%dT%H:%M:%S')
tensors.append(torch.tensor(inputs.dt.year.values).to(device))
tensors.append(torch.tensor(inputs.dt.month.values).to(device))
tensors.append(torch.tensor(inputs.dt.day.values).to(device))
tensors.append(torch.tensor(inputs.dt.hour.values).to(device))
tensors.append(torch.tensor(inputs.dt.hour.values).to(device))

layers = [5, 200, 800, 1500, 800, 400, 200, 1]

actualOutputs = [int(temp.replace("s", '')) if isinstance(temp, str) else temp for temp in filterDf['HourlyDryBulbTemperature']]
predictedOutputs = []

tensor_thing = torch.FloatTensor(tensors).to(device)


class H(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for f in range(len(layers) - 1):
            if f != len(layers) - 1:
                self.input_layer_exe[f] = torch.nn.Linear(
                    layers[f], layers[f + 1])  # input: x value
        self.Activate = torch.nn.ReLU()
        # self.output_layer_exe = torch.nn.Linear(10, 1)  # output: y value

    def forward(self, input):
        for layer in self.input_layer_exe:
            input = layer(input)
            if layer != self.input_layer_exe[len(self.input_layer_exe) - 1]:
                input = self.Activate(input)
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
