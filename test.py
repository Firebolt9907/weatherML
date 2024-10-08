import datetime

import torch
import requests

from model import Model

tmrw = 1 # 0 if today, 1 if tmrw
hourOffset = -24 # can be between -24 and 0

device = (
    # "cuda"
    # if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    # else "cpu"
    "cpu"
)

model = Model()
model.load_state_dict(torch.load('data/model.pt', map_location=torch.device(device)))
model.to(device)

today = datetime.datetime.now()
url = ("https://api.open-meteo.com/v1/forecast?latitude=41.6117&longitude=-93.8852&hourly=temperature_2m"
       "&temperature_unit=fahrenheit&timezone=America%2FChicago&start_date=") + str(datetime.datetime.fromtimestamp(
    today.timestamp() - 950400).date()) + "&end_date=" + str(today.date())
response = requests.get(url)
data = response.json()

actual = []
predicted = []

for i in range(24):
    tod = data["hourly"]["temperature_2m"][264 + today.hour + hourOffset]

    print("Actual temperature: " + str(tod))
    actual.append(tod)

    # if tmrw > 0:


    inp = torch.FloatTensor([
        today.year,                                         # year
        today.month,                                        # month
        today.day + tmrw,                                   # day
        today.hour,                                         # hour
        (round((today.minute - 15) / 20) * 20 - 5),         # nearest minute model allows
        data["hourly"]["temperature_2m"][216+24 + today.hour + tmrw*24 + hourOffset], # yesterday's temp
        data["hourly"]["temperature_2m"][192+24 + today.hour + tmrw*24 + hourOffset], # ereyesterday's temp
        data["hourly"]["temperature_2m"][120+24 + today.hour + tmrw*24 + hourOffset], # temp 5 days ago
        data["hourly"]["temperature_2m"][24+today.hour + tmrw*24 + hourOffset]        # temp 10 days ago
    ]).to(device)

    print("Predicted temperature: " + str(model(inp)[0].item()))
    predicted.append(model(inp)[0].item())
    hourOffset += 1

print(actual)
print(predicted)

