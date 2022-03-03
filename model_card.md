# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
### Model developer
This model was developed by [Tommaso Fougier](https://github.com/tom-ph) as an exercise for the [Udacity Machine Learning DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).
### Model date
The model was developed in February 2022.
### Model version
The model is at its first version. Currently new versions are not expected.
### Model type
The model is a [Catboost Classifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)
### License
The model and the entire project has the [Unlicense](https://unlicense.org). Anyone is free to use it and should not consider it more than an experiment. No type of support is guaranteed.

## Intended Use
This model tries to predict if someone has a yearly income of more than 50k given some anagraphic data.

## Training Data
The model was trained on the [Census Income public dataset](https://archive.ics.uci.edu/ml/datasets/census+income).

## Evaluation Data
The model was trained using cross-validation over the entire dataset.

## Training tecnique
The training was performed using hyperparameter tuning, with 500 iterations for each combination of hyperparameters.
The following hyperparameters were evaluated:
- learning_rate: 
    - 0.05
    - 0.1
- depth: 
    - 6
    - 8
    - 10

The best results were obtained with a **0.05** learning rate and a depth of **6**.
To address the class imbalance, the *auto-class-weights* parameter of the CatBoost was set to **SqrtBalanced**.

## Metrics
During the training, the model was evaluated with the test loss. The best test loss result was 0.313.
To better evaluate the model, three additional metrics were chosen: **Precision**, **Recall** and **F1-score**.
The best model reached a **71.3%** precision, **79.9%** recall and **75.3%** F1-score.
However some sliced metrics had way lower results, specially in terms of recall (for example some group of *education* and *race*). You should always consider these sliced metrics when you use this model for inference. 

## Ethical Considerations
This model should not be seen as an indicator to determine if someone deserves more or less money.

## Caveats and Recommendations
Feel free to reach me if you have reccomendations about this project.