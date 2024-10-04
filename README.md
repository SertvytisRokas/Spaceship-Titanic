# Spaceship Titanic
The relevant project is located in spaceship_titanic.ipynb.

To run the notebook, first install dependencies using pip:
```
pip install -r requirements.txt
```

## Introduction
In this project by objective was to develop a model that's capable of predicting whether a passenger was transported from the spaceship titanic crash with at least 79% accuracy.

## Data structure
- `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
- `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
- `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
- `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
- `Destination` - The planet the passenger will be debarking to.
- `Age` - The age of the passenger.
- `VIP` - Whether the passenger has paid for special VIP service during the voyage.
- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
- `Name` - The first and last names of the passenger.
- `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

## Project structure
- Exploratory data analysis
- Correlation analysis
- Missing values analysis and kNN imputation
- Categorical feature encoding
- Outlier analysis
- Statistical inference
- Numerical data normalization
- Hyperparameter tuning
- Feature reduction
- Model ensemble
- Cross-validation
- Conclusion
- Potential improvements
