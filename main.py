import torch
import pandas as pd
import re
import plotly.graph_objects as go

df = pd.read_csv('weather.csv')

LEARN_RATE = 0.15
EPOCHS = 50
DATA_PER_EPOCH = 500
new = False

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)


class useful:
    def findFromTS(date, data):
        return data.loc[data['DATE'] == useful.getTS(date)]

    def findFromInputs(year, month, day, hour, minute, data):
        return data.loc[data['DATE'] == useful.getTSFromInputs(year, month, day, hour, minute)]

    def getTS(date):
        return str(date.year) + "-" + useful.leadZeroes(date.month) + "-" + useful.leadZeroes(
            date.day) + "T" + useful.leadZeroes(
            date.hour) + ":" + str(date.minute) + ":00"

    def getTSFromInputs(year, month, day, hour, minute):
        return str(year) + "-" + useful.leadZeroes(month) + "-" + useful.leadZeroes(day) + "T" + useful.leadZeroes(
            hour) + ":" + str(
            minute) + ":00"

    def leadZeroes(number):
        return "{:02d}".format(number)


filterDf = df[['DATE', 'HourlyDryBulbTemperature']
           ].dropna().iloc[500:].reset_index(
    drop=True)
randomDf = filterDf.sample(frac=1).reset_index(
    drop=True).iloc[0:EPOCHS * DATA_PER_EPOCH]

inputs = []
dates = pd.to_datetime(randomDf['DATE'], format='%Y-%m-%dT%H:%M:%S')
for date in dates:
    # dayBefore = df.loc[filterDf['DATE'] == "2014-11-22T07:35:00"]
    if (date.minute - 15) >= 0 and (date.minute - 15) % 20 == 0:
        inputs.append([date.year, date.month,
                       date.day, date.hour, date.minute])
        print("current = " + useful.getTS(date))
        print("day before = " + filterDf.iloc[useful.findFromTS(date, filterDf).index[0] - 72]['DATE'])
    else:
        print(useful.findFromTS(date, randomDf).iloc[0].index[0])
        randomDf.drop([useful.findFromTS(date, randomDf).iloc[0].index[0]])

actualOutputs = [int(temp.replace("s", '')) if isinstance(
    temp, str) else temp for temp in randomDf['HourlyDryBulbTemperature']]
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
if not new:
    model.load_state_dict(torch.load(
        'model.pt', map_location=torch.device(device)))
    model.to(device)

loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

output = torch.FloatTensor(actualOutputs).to(device)
input = torch.FloatTensor(inputs).to(device)

loss_graph_figure = go.Figure()
loss_graph_values = []
loss_graph_figure.add_scatter(y=loss_graph_values)
loss_graph_figure.write_html("loss.html")

currentEpoch = 0
epochLosses = []

for i in range(len(inputs)):
    if i % DATA_PER_EPOCH == 0 and i != 0:
        currentEpoch += 1
        print(f"Epoch {currentEpoch}")
        epochAvgLoss = sum(epochLosses) / len(epochLosses)
        loss_graph_values.append(epochAvgLoss)
        print(f' Average Loss : {epochAvgLoss}')
        print(f' Learn Rate : {optimizer.param_groups[0]["lr"]}')

        loss_graph_figure.write_html("loss.html")
        optimizer.param_groups[0]['lr'] *= 0.95

        epochLosses = []

    prediction = model(input[i])
    loss = loss_function(prediction, output[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    epochLosses.append(loss.item())

    # print(
    #     f"current loss at day {str(inputs[i][1])}/{str(inputs[i][2])}/{str(inputs[i][0] % 100)} at {str(inputs[i][3])}:{str(inputs[i][4])}, iteration {str(i)} loss: {str(loss.item())}, predicted: {str(prediction.item())}, actual: {str((output[i].item()))}")

torch.save(model.state_dict(), 'model.pt')
