import datetime

import torch
import requests

from model import Model

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = Model()
model.load_state_dict(torch.load('data/model.pt', map_location=torch.device(device)))
model.to(device)

today = datetime.datetime.now()
url = ("https://api.open-meteo.com/v1/forecast?latitude=41.6117&longitude=-93.8852&hourly=temperature_2m"
       "&temperature_unit=fahrenheit&timezone=America%2FChicago&start_date=") + str(datetime.datetime.fromtimestamp(
    today.timestamp() - 864000).date()) + "&end_date=" + str(today.date())
response = requests.get(url)
data = response.json()

tod = data["hourly"]["temperature_2m"][240 + today.hour]

print("Actual temperature: " + str(tod))

inp = torch.FloatTensor([
    today.year,                                         # year
    today.month,                                        # month
    today.day,                                          # day
    today.hour,                                         # hour
    (round((today.minute - 15) / 20) * 20 - 5),         # nearest minute model allows
    data["hourly"]["temperature_2m"][216 + today.hour], # yesterday's temp
    data["hourly"]["temperature_2m"][192 + today.hour], # ereyesterday's temp
    data["hourly"]["temperature_2m"][120 + today.hour], # temp 5 days ago
    data["hourly"]["temperature_2m"][today.hour]        # temp 10 days ago
]).to(device)

print("Predicted temperature: " + str(model(inp)[0].item()))
