import json
import torch
import pandas as pd
import re
import plotly.graph_objects as go
from model import Model

EPOCHS = 50  # 200
DATA_PER_EPOCH = 50  # 50
NEW_MODEL = False
CLEAN_DATA = True
LR_SCALE = 0.95  # 0.95
# new models need a higher learn rate
if NEW_MODEL:
    LEARN_RATE = 2e-3  # 2e-3
else:
    LEARN_RATE = 4e-4  # 2e-3


device = (
    "cuda"
    if torch.cuda.is_available()
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

    def removeS(inp):
        if isinstance(inp, str):
            inp = inp.replace("s", '')
            match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', inp)
            if match:
                return float(match.group())
            else:
                print(inp)
                raise ValueError("No valid number found in the input string")
        else:
            return float(inp)

if CLEAN_DATA:
    df = pd.read_csv('data/filtered.csv')
    og = pd.read_csv('data/weather.csv')
    filterDf = df.reset_index(
        drop=True)
    randomDf = filterDf.sample(frac=1).reset_index(
        drop=True).iloc[0:EPOCHS * DATA_PER_EPOCH]

    inputs = [[]]
    expectedOutputs = [[]]

    for i, day in randomDf.iterrows():
        if day['1d'] == "-" or day['2d'] == "-" or day['5d'] == "-" or day['10d'] == "-":
            print("skipping")
        else:
            inputs.append([day['year'], day['month'], day["day"], day["hour"], day["minute"],
                           useful.removeS(day['1d']),
                           useful.removeS(day['2d']),
                           useful.removeS(day['5d']),
                           useful.removeS(day['10d']),
                           ])
            temp = og.loc[
                useful.findRowDataFromInputs(day['year'], day['month'], day["day"], day["hour"], day["minute"], og).index[0]]
            expectedOutputs.append([useful.removeS(temp['HourlyDryBulbTemperature'])])

        if i % (DATA_PER_EPOCH * 10) == 0 and i != 0:
            print("Epoch " + str(i/DATA_PER_EPOCH) + " cleaned")
    del randomDf, filterDf, df, og
else:
    inputs = json.load(open("data/inputs.json"))["stuff"]
    expectedOutputs = json.load(open("data/expectedoutputs.json"))["stuff"]


inputs = inputs[1:]
expectedOutputs = expectedOutputs[1:]
# print(inputs)
# print(expectedOutputs)
print("Done filtering")

predictedOutputs = []

output = torch.FloatTensor(expectedOutputs).to(device)
input = torch.FloatTensor(inputs).to(device)



model = Model().to(device)
if not NEW_MODEL:
    model.load_state_dict(torch.load(
        'data/model.pt', map_location=torch.device(device)))
    model.to(device)

loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

loss_graph_figure = go.FigureWidget()
# widget = go.FigureWidget(loss_graph_figure)
loss_graph_values = []
loss_graph_figure.add_scatter(y=loss_graph_values)
loss_graph_figure.write_html("data/loss.html")

currentEpoch = 0
epochLosses = []
bestLoss = float('inf')

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
        loss_graph_figure.write_html("data/loss.html")
        optimizer.param_groups[0]['lr'] *= LR_SCALE


        epochLosses = []
        if epochAvgLoss < bestLoss:
            torch.save(model.state_dict(), 'data/newmodel.pt')
            bestLoss = epochAvgLoss


    prediction = model(input[i])
    loss = loss_function(prediction, output[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    epochLosses.append(loss.item())

    # loss_graph_values.append(loss.item())
    # loss_graph_scatter = loss_graph_figure.data[0]
    # loss_graph_scatter.y = loss_graph_values
