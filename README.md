
# Aviation Failure 

This project is an Aviation Failure Prediction system developed using Django and various machine learning techniques. The system aims to predict potential failures in aviation components based on historical data, improving maintenance schedules and enhancing safety measures.





## Features

- Data ingestion and preprocessing
- Machine learning model training and evaluation
- Failure prediction based on input data
- User authentication and authorization
- Interactive dashboard for data visualization
- RESTful API for integration with other systems



## Requirements

- Python 3.x
- Django 3.x or higher
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Django REST framework

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DShubhamBhardwaj/AviationFailure.git
cd Aviation

```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:



```bash
pip install -r requirements.txt

```

4. Start the development server:

```bash
python manage.py runserver

```

## Usage

1. Open a web browser and go to http://127.0.0.1:8000/.

2. Access the prediction interface to input data and receive failure predictions.

3. Explore the interactive dashboard to visualize data and model performance.1

## Screenshots
- Landing Page\
![Landing Page](https://github.com/DShubhamBhardwaj/AviationFailure/blob/main/Screenshots/LandingPage.png)
- Dashboard\
![Dashboard Data](https://github.com/DShubhamBhardwaj/AviationFailure/blob/main/Screenshots/DashboardData.png)
- Data Processing\
![Data Processing](https://github.com/DShubhamBhardwaj/AviationFailure/blob/main/Screenshots/DataProcessing.png)
- Remaining useful Life Calculator\
![Remaing USeful Cycle Caclulator](https://github.com/DShubhamBhardwaj/AviationFailure/blob/main/Screenshots/RemainingUsefulLifeCalc.png)


## How It Works

1. Data Ingestion and Preprocessing: Raw aviation data is ingested and preprocessed to make it suitable for training machine learning models. This involves cleaning the data, handling missing values, and feature engineering.

2. Model Training and Evaluation: Various machine learning models are trained on the historical data to predict failures. The models are evaluated using appropriate metrics to select the best performing one.

3. Failure Prediction: The selected model is used to predict potential failures based on new input data. Users can input data through the web interface or via the RESTful API.

4. Interactive Dashboard: An interactive dashboard is provided for visualizing the data and the performance of the machine learning models. This helps in understanding trends and making informed decisions.


## Acknowledgements

 - [Django](https://www.djangoproject.com/)
 - [scikit-learn](https://scikit-learn.org/)
- [Pandas]([https://www.php.net/](https://pandas.pydata.org/))
 - [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)



## Documentation

- please find the Complete Project report : [Aviation Failure Project Report](https://drive.google.com/file/d/11vgg8svXjhcwNCa4VCeGLw_yCVY7-xgG/view?usp=sharing)
- Paper written after research : [FORECASTING FAILURE IN AVAITION INDUSTRY](https://drive.google.com/file/d/1KZf1OteE6GNJlZ5d97jhrHNwlCn4xi3S/view?usp=sharing)

