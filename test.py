import datetime

import torch
import requests

# from weatherML.main import Model

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(9, 200)
        self.layer2 = torch.nn.Linear(200, 800)
        self.layer3 = torch.nn.Linear(800, 1500)
        self.layer3a = torch.nn.Linear(1500, 3000)
        self.layer3b = torch.nn.Linear(3000, 6000)
        self.layer3c = torch.nn.Linear(6000, 3000)
        self.layer3d = torch.nn.Linear(3000, 1500)
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
        input = self.layer3a(input)
        input = self.Activate(input)
        # input = self.layer3b(input)
        # input = self.Activate(input)
        # input = self.layer3c(input)
        # input = self.Activate(input)
        input = self.layer3d(input)
        input = self.Activate(input)
        input = self.layer4(input)
        input = self.Activate(input)
        input = self.layer5(input)
        input = self.Activate(input)
        input = self.layer6(input)
        return input

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = Model()
model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
model.to(device)

today = datetime.datetime.now()
url = ("https://api.open-meteo.com/v1/forecast?latitude=41.6117&longitude=-93.8852&hourly=temperature_2m"
       "&temperature_unit=fahrenheit&timezone=America%2FChicago&start_date=") + str(datetime.datetime.fromtimestamp(
    today.timestamp() - 864000).date()) + "&end_date=" + str(today.date())
response = requests.get(url)
data = response.json()

tod = data["hourly"]["temperature_2m"][240 + today.hour]
d1 = data["hourly"]["temperature_2m"][216 + today.hour]
d2 = data["hourly"]["temperature_2m"][192 + today.hour]
d5 = data["hourly"]["temperature_2m"][120 + today.hour]
d10 = data["hourly"]["temperature_2m"][today.hour]

print("Actual temperature: " + str(tod))

inp = torch.FloatTensor([today.year, today.month, today.day, today.hour, (round((today.minute - 15) / 20) * 20 - 5), d1, d2, d5, d10]).to(device)

print("Predicted temperature: " + str(model(inp)[0].item()))
