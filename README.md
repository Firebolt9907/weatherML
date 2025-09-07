# WeatherML

## About This Project

This is a website made to predict the unpredictable: Midwest Weather. Forecasts are often wrong, and I don't believe it's the meteorologists' fault. I decided to create a feed-forward neural network to try to predict the weather, and it worked pretty well! It is often within 3 degrees Fahrenheit off of the actual temperature, which is a difference almost no one can notice. 

## <code>// TODO:</code>

- Try to predict other parts of the forecast like wind
- Extrapolate predictions beyond one day (this would very likely be either inaccurate)
- Reduce model size (currently uses ~2 GB of VRAM to train)
- Run in an app or website for actual usage

## Features

- Temperature predictions 24 hours in the future
- Option to check current weather based on past temperature to check accuracy

## Technologies Used

- NumPy for creating a database
- Pandas for manipulating the database
- PyTorch for a high performance ML framework

## Data Used

- NOAA weather data for Clive IA between 2013 and 2023
