<h2 align="center">Parole Hearing Verdict Pipeline</h2>

## Task Description

The task is to build a **machine learning pipeline** that will predict whether an inmate will get parole or not based on
their prisoner profile. A prisoner profile is a collection of features that are used to predict parole. Since the
target is binary, it is a **classification task**.

The best model so far is `RandomForestClassifier` (baseline) with a **10-Fold validation** on test set of `67%` (F1 score).

The most important feature is `years_from_parole_eligibility` (The number of years since the inmate was eligible for
parole). It was created by using other features such
as `'sentence_date', 'offence_date', 'parole_eligibility_date', 'maximum_sentence_date'`

## About the Data

The data is from the [Texas Department of Criminal Justice](https://data.texas.gov/dataset/). The dataset contains on
hand inmate population with relevant demographic, offense, and parole information. The data is
**scrapped** from TDCJ and stored in a private **AWS S3** bucket.

Attributes:

- SID Numberdata file: Numerical ID that serves as a unique identifier for each data entry
- TDCJ Number: Texas Department of Criminal Justice number is a numerical identifier for each inmate.
- Name: Name of an inmate.
- Current Facility: Name of correctional facility that the inmate is staying
- Gender: Gender of an inmate. (Male or Female)
- Race: Race of inmate. (Asian, Black, Hispanic, Indian, Other, Unknown, White)
- Age: Age of inmate.
- Projected Release: The projected release date of an inmate at sentencing. If the sentence is for life in prison, the
  projected release date is 1/1/9999
- Maximum Sentence Date: The maximum sentence release date of an inmate at sentencing. If the sentence is for life in
  prison, the projected maximum release date is 1/1/9999
- Parole Eligibility Date: Date that the inmate is eligible for parole. The attribute is left blank if the inmate is not
  eligible for parole.
- Case Number: Text field that serves as a unique identifier for the criminal case.
- County: County in Texas where the offense took place.
- Offense Code: Numerical code that refers to the criminal offense that the inmate was found guilty of.
- TDCJ Offense: The name of the Texas Department of Criminal Justice Offense that the inmate was found guilty of.
- Sentence Date: Date in which the inmate was sentenced.
- Offense Date: Date in which the inmate committed the crime.
- Sentence (Years): The number of years the inmate was sentenced for committing the crime. If the inmate was sentenced
  to life in prison or the death penalty, the entry reads “Life”, “Capital Life” or “Death”.
- Last Parole Decision: The decision of the inmate’s last parole hearing. If the inmate did not have a hearing yet, the
  entry is “none”.
- Next Parole Review Date:  Date of inmate’s next parole hearing.
- Parole Review Status: Does the inmate currently have a parole review in progress

## AWS S3

```
├── texas-data-bucket
    ├── data           <- data from different stages of the pipeline e.g. clean, processed, etc.
        ├── clean
             ├── 20220507-005720.csv
             ├── 20220507-155720.csv
             ├── 20220507-005220.csv
             ├── 20220507-156722.csv
        ├── processed
             ├── 20220507-001972.csv
             ├── 20220507-191820.csv
             ├── 20220507-009280.csv
             ├── 20220507-198222.csv     
    ├── scratch        <- scratch space for intermediate data
    ├── paroledata       <- raw data from TDCJ 
```

## Model Wrapper

`src/model/classifiers.py` contains the wrapper for the models.
The wrapper is used to train the models and test the models.

Here is the code for an abstract class that implements the wrapper.

```python
class Model(ABC):
    """Abstract class for models."""

    def __init__(self, features: List[str] = None, label: str = None, params: dict = None):
        if features is None:
            features = []
        if label is None:
            label = []
        if params is None:
            params = {}
        self.label = label
        self.features = features
        self.params = params
        self.model = None

    def preprocess(self, df: pd.DataFrame):
        """ Any model specific preprocessing that needs to be done before training the model."""
        pass

    def split(self, X, Y, test_size: float):
        """Split the dataset into training and tests sets."""
        pass

    def normalize(self, X):
        """Normalize the dataset."""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the labels for the given dataset."""
        pass

    @abstractmethod
    def evaluate(self, X, Y):
        """Evaluate the model."""
        pass

    @abstractmethod
    def cross_validate(self, X, Y, n_splits: int = 10):
        """Cross validate the model."""
        pass

    def feature_importance(self, X, Y):
        """Get the feature importance."""
        pass
```

## Roadmap

- [x] Prototype
- [x] Pipeline

## Dependencies

    $ pip install -r requirements.txt

## Running the pipeline

    $ git clone repo.git
    $ cd repo

    // Install the dependencies
    $ python setup.py install

    // Run the pipeline
    (scrape -> clean -> feature_engineering -> train ->  evaluate)
    $ python main/scrape.py
    $ python main/clean.py
    $ python main/feature_engineering.py
    $ python main/train_model.py

## Running the tests

    py.test tests

## License

Distributed under the MIT License. See `LICENSE.md` for more information.

## Contact

Vineet Verma - vineetver@hotmail.com - [Goodbyeweekend.io](https://www.goodbyeweekend.io/)

Matthew Bhaya - mbhaya@bu.edu
