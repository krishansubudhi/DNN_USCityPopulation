# DNN_USCityPopulation
DNN model to predict US city population.
Libraries used are *python numpy, pandas, keras*

## Task
Predict population of a US CITY from the following input
Area LAND
Area WATER
LATitude
LONGitude
STATE
UrbanAreatype

## Data Source
Data from US census report 2010
https://www.census.gov/geo/maps-data/data/gazetteer2010.html

## Data import and cleanup

![Alt text](images/dataframe.PNG?raw=true "Original data converted to data frame")

## Training
This repo contains all the experiments done with hyper parameters. As per my analysis,

After multiple experiemnts with the number of layers, batch_size a 4 layered DNN with a batch size of 50 produced best result.
But the error was still high. 
The reason was the loss function. Sunce city population has a high deviation, use of Mean Squared Error(MSE) as a loss function shifted the model inclination towards high values. Predicting the small town population wrong did not penalise the model much.

Hence Mean Absolute Percentage Error (MAPE) was chosen as the loss function. This reduced both MAE(Mean Absolute Error) and MAPE of the model.

```
model = createmodel([32,16,16,16])
history = model.fit(x_train,y_train,
         validation_data=(x_validate,y_validate),
         epochs=120,
         batch_size = 50)
 ```        
![Alt text](images/loss_error.PNG?raw=true "loss and mae after training")

## Evaluation
The final MAPE was 25% and MAE was 14%, which in my view is a good performance considering the limited number of features provided.
This model can still be improved and open to pull requests.
![Alt text](images/result.PNG?raw=true "sample predictions on validation set") 

