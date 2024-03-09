import torch
import pandas as pd
import re
import plotly.graph_objects as go

df = pd.read_csv('filtered.csv')

LEARN_RATE = 0.15
EPOCHS = 50
DATA_PER_EPOCH = 500
new = True

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)


class useful:
    def findRowDataFromTS(date, data):
        return data.loc[data['DATE'] == useful.getTS(date)]

    def findRowDataFromInputs(year, month, day, hour, minute, data):
        return data.loc[data['DATE'] == useful.getTSFromInputs(year, month, day, hour, minute)]

    def getTS(date):
        return str(date.year) + "-" + useful.leadZeroes(date.month) + "-" + useful.leadZeroes(
            date.day) + "T" + useful.leadZeroes(
            date.hour) + ":" + useful.leadZeroes(date.minute) + ":00"

    def getTSFromInputs(year, month, day, hour, minute):
        return str(year) + "-" + useful.leadZeroes(month) + "-" + useful.leadZeroes(day) + "T" + useful.leadZeroes(
            hour) + ":" + useful.leadZeroes(
            minute) + ":00"

    def leadZeroes(number):
        return "{:02d}".format(number)


filterDf = df[['DATE', 'HourlyDryBulbTemperature', 'HourlyDewPointTemperature', 'HourlyRelativeHumidity',
               'HourlyVisibility', 'HourlyWindDirection', 'HourlyWindSpeed', '1d', '2d', '5d', '10d']
           ].dropna().iloc[0:].reset_index(
    drop=True)
randomDf = filterDf.sample(frac=1).reset_index(
    drop=True).iloc[0:EPOCHS * DATA_PER_EPOCH]

inputs = []
dates = pd.to_datetime(randomDf['DATE'], format='%Y-%m-%dT%H:%M:%S')
for date in dates:
    # dayBefore = df.loc[filterDf['DATE'] == "2014-11-22T07:35:00"]
    if useful.findRowDataFromTS(date, filterDf).index[0] > 720:
        # inputs: yyyy, mm, dd, hh, mm, temp from 1d ago, 2d ago, 5d ago, 10d ago
        inputs.append([date.year, date.month,
                       date.day, date.hour, date.minute,
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72]["HourlyDryBulbTemperature"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 144][
                           "HourlyDryBulbTemperature"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 360][
                           "HourlyDryBulbTemperature"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 720][
                           "HourlyDryBulbTemperature"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                           "HourlyDewPointTemperature"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72]["HourlyRelativeHumidity"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72]["HourlyVisibility"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72]["HourlyWindDirection"],
                       filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72]["HourlyWindSpeed"],
                       ])
    else:
        print("dropping = " + useful.findRowDataFromTS(date, randomDf)['DATE'])
        randomDf.drop([useful.findRowDataFromTS(date, randomDf).index[0]])

print("Done filtering")

actualOutputs = [
    [int(item['HourlyDryBulbTemperature'].replace("s", '')) if isinstance(
        item['HourlyDryBulbTemperature'], str) else item['HourlyDryBulbTemperature'],
     item["HourlyDewPointTemperature"],
     item["HourlyRelativeHumidity"],
     item["HourlyVisibility"],
     item["HourlyWindDirection"],
     item["HourlyWindSpeed"],
     ] for item in randomDf
]
predictedOutputs = []

tensor_thing = torch.FloatTensor(inputs).to(device)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(13, 200)
        self.layer2 = torch.nn.Linear(200, 800)
        self.layer3 = torch.nn.Linear(800, 1500)
        self.layer3a = torch.nn.Linear(1500, 3000)
        self.layer3b = torch.nn.Linear(3000, 6000)
        self.layer3c = torch.nn.Linear(6000, 3000)
        self.layer3d = torch.nn.Linear(3000, 1500)
        self.layer4 = torch.nn.Linear(1500, 800)
        self.layer5 = torch.nn.Linear(800, 200)
        self.layer6 = torch.nn.Linear(200, 6)
        self.Activate = torch.nn.ReLU()

    def forward(self, input):
        input = self.layer1(input)
        input = self.Activate(input)
        input = self.layer2(input)
        input = self.Activate(input)
        input = self.layer3(input)
        input = self.Activate(input)
        input = self.layer3a(input)
        input = self.Activate(input)
        input = self.layer3b(input)
        input = self.Activate(input)
        input = self.layer3c(input)
        input = self.Activate(input)
        input = self.layer3d(input)
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

        loss_graph_scatter = loss_graph_figure.data[0]
        loss_graph_scatter.y = loss_graph_values
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
