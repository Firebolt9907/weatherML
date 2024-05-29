import re

import torch
import pandas as pd
import numpy as np

df = pd.read_csv('weather.csv')

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

    def removeS(inp):
        if isinstance(inp, str):
            tem = re.findall('[-+]?\d+', inp.replace("s", ''))
            outpu = ""
            for letter in tem:
                outpu += letter
            return int(float(outpu))
        else:
            return int(float(inp))

    def removeV(inp):
        if isinstance(inp, str):
            tem = re.findall('[-+]?\d+', inp.replace("V", ''))
            outpu = ""
            for letter in tem:
                outpu += letter
            return int(float(outpu))
        else:
            return int(float(inp))


filterDf = df[['DATE', 'HourlyDryBulbTemperature', 'HourlyDewPointTemperature', 'HourlyRelativeHumidity',
               'HourlyVisibility', 'HourlyWindDirection', 'HourlyWindSpeed']
           ].dropna().iloc[0:].reset_index(
    drop=True)
# randomDf = filterDf.sample(frac=1).reset_index(
#     drop=True).iloc[0:EPOCHS * DATA_PER_EPOCH]

inputs = pd.DataFrame({'DATE': [0, 0], 'year': [0, 0], 'month': [0, 0], 'day': [0, 0], 'hour': [0, 0], 'minute': [0, 0],
                       '1d': [0, 0], '2d': [0, 0], '5d': [0, 0], '10d': [0, 0], '1dHDPT': [0, 0], '1dHRH': [0, 0],
                       '1dHV': [0, 0],
                       '1dHWS': [0, 0], "HDBT": [0, 0],
                       'HDPT': [0, 0], 'HRH': [0, 0], 'HV': [0, 0], 'HWS': [0, 0]})

dates = pd.to_datetime(filterDf['DATE'], format='%Y-%m-%dT%H:%M:%S')[720:]

dp = 0
for date in dates:
    # dayBefore = df.loc[filterDf['DATE'] == "2014-11-22T07:35:00"]
    if dp % 200 == 0:
        print("iteration " + str(dp))
    clean = True
    # if useful.findRowDataFromTS(date, filterDf).index[
    #     0] > 720:
    #     if filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
    #                 "HourlyDryBulbTemperature"] == '-' or filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 144][
    #                 "HourlyDryBulbTemperature"] == '-' or filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 360][
    #                 "HourlyDryBulbTemperature"] == '-' or filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 720][
    #                 "HourlyDryBulbTemperature"] == '-' or filterDf.loc[useful.findRowDataFromTS(date, filterDf)["HourlyDryBulbTemperature"].index[0]] == '-':
    #         clean = False

    if (date.minute - 15) >= 0 and (date.minute - 15) % 20 == 0 and useful.findRowDataFromTS(date, filterDf).index[
        0] > 720 and clean:
        # inputs: yyyy, mm, dd, hh, mm, temp from 1d ago, 2d ago, 5d ago, 10d ago
        inputs.loc[len(inputs.index)] = np.asarray([
            date, date.year, date.month,  # DATE, year, month
            date.day, date.hour, date.minute,  # day, hour, min
            useful.removeS(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyDryBulbTemperature"]),  # 1d
            useful.removeS(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 144][
                "HourlyDryBulbTemperature"]),  # 2d
            useful.removeS(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 360][
                "HourlyDryBulbTemperature"]),  # 5d
            useful.removeS(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 720][
                "HourlyDryBulbTemperature"]),  # 10d
            # int(float(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
            #     "HourlyDewPointTemperature"])),  # 1dHDPT
            # useful.removeV(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
            #     "HourlyRelativeHumidity"]),  # 1dHRH
            # useful.removeV(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
            #     "HourlyVisibility"]),  # 1dHV
            # int(float(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
            #     "HourlyWindSpeed"])),  # 1dHWS
            #   EXPECTED OUTPUTS BELOW
            useful.removeS(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0]][
                               "HourlyDryBulbTemperature"]),  # HDBT
            # int(float(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0]]["HourlyDewPointTemperature"])),  # HDPT
            # useful.removeV(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0]]["HourlyRelativeHumidity"]),  # HRH
            # useful.removeV(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0]]["HourlyVisibility"]),  # HV
            # int(float(filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0]]["HourlyWindSpeed"])),  # HWS
        ], dtype="object")
    else:
        print("dropping = " + useful.findRowDataFromTS(date, filterDf)['DATE'])
        # filterDf.drop([useful.findRowDataFromTS(date, filterDf).index[0]])
    dp+=1

inputs = inputs[2:]

inputs.to_csv('filterd.csv', encoding='utf-8', index=False)
print("finished")
