
You can find the dataset on kaggle: https://www.kaggle.com/datasets/fratkse/hamburger-ingredients


HOW TO USE?


Before running any codes, run the following command on the command prompt while being in the current directory. (For Windows)

```pip install -r requirements.txt```



"hamburger_ingredients_classifier.py" trains the model using the data that is stored in the "data" folder,
saves the model to the "model" folder, and displays stats of the model.


"app.py" is the GUI app that either trains a new model using "hamburger_ingredients_classifier.py"
or reads a model selected from any directory according to users choice and uses it for further predictions.


"input_data" folder includes some example data for user to test the app and is not used in any codes.
You do not have to put the input image that you want to predict to "input_data" folder,
you can select any image from any directory.


"output_data" folder stores outputs saved by the user on the app.

