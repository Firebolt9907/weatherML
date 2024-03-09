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


filterDf = df[['DATE', 'HourlyDryBulbTemperature', 'HourlyDewPointTemperature', 'HourlyRelativeHumidity',
               'HourlyVisibility', 'HourlyWindDirection', 'HourlyWindSpeed']
           ].dropna().iloc[0:].reset_index(
    drop=True)
# randomDf = filterDf.sample(frac=1).reset_index(
#     drop=True).iloc[0:EPOCHS * DATA_PER_EPOCH]

inputs = pd.DataFrame({'DATE': [0, 0], 'year': [0, 0], 'month': [0, 0], 'day': [0, 0], 'hour': [0, 0], 'minute': [0, 0],
                       '1d': [0, 0], '2d': [0, 0], '5d': [0, 0], '10d': [0, 0], '1dHDPT': [0, 0], '1dHRH': [0, 0],
                       '1dHV': [0, 0], '1dHWD': [0, 0],
                       '1dHWS': [0, 0], "HDBT": [0, 0],
                       'HDPT': [0, 0], 'HRH': [0, 0], 'HV': [0, 0],
                       'HWD': [0, 0], 'HWS': [0, 0]})
dates = pd.to_datetime(filterDf['DATE'], format='%Y-%m-%dT%H:%M:%S')

dp = 0
for date in dates:
    # dayBefore = df.loc[filterDf['DATE'] == "2014-11-22T07:35:00"]
    if dp % 200 == 0:
        print("iteration " + str(dp))
    if (date.minute - 15) >= 0 and (date.minute - 15) % 20 == 0 and useful.findRowDataFromTS(date, filterDf).index[
        0] > 720:
        # inputs: yyyy, mm, dd, hh, mm, temp from 1d ago, 2d ago, 5d ago, 10d ago
        inputs.loc[len(inputs.index)] = np.asarray([
            date, date.year, date.month,  # DATE, year, month
            date.day, date.hour, date.minute,  # day, hour, min
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyDryBulbTemperature"],  # 1d
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 144][
                "HourlyDryBulbTemperature"],  # 2d
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 360][
                "HourlyDryBulbTemperature"],  # 5d
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 720][
                "HourlyDryBulbTemperature"],  # 10d
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyDewPointTemperature"],  # 1dHDPT
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyRelativeHumidity"],  # 1dHRH
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyVisibility"],  # 1dHV
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyWindDirection"],  # 1dHWD
            filterDf.loc[useful.findRowDataFromTS(date, filterDf).index[0] - 72][
                "HourlyWindSpeed"],  # 1dHWS
            #   EXPECTED OUTPUTS BELOW
            useful.findRowDataFromTS(date, filterDf)["HourlyDryBulbTemperature"],  # HDBT
            useful.findRowDataFromTS(date, filterDf)["HourlyDewPointTemperature"],  # HDPT
            useful.findRowDataFromTS(date, filterDf)["HourlyRelativeHumidity"],  # HRH
            useful.findRowDataFromTS(date, filterDf)["HourlyVisibility"],  # HV
            useful.findRowDataFromTS(date, filterDf)["HourlyWindDirection"],  # HWD
            useful.findRowDataFromTS(date, filterDf)["HourlyWindSpeed"],  # HWS
        ], dtype="object")
    else:
        print("dropping = " + useful.findRowDataFromTS(date, filterDf)['DATE'])
        # filterDf.drop([useful.findRowDataFromTS(date, filterDf).index[0]])
    dp+=1

inputs.to_csv('filtered.csv', encoding='utf-8', index=False)
print("finished")
