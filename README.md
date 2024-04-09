# Blotter - Transform

This is a python-package which allows for blotter-transformations using extending the `pd.Dataframe` and `pd.Series` class respectively. Its targeted use is  the context of financial data.

## Getting Started

To install the package `pip` must be installed, also python `3.9` or newer.

### Installation

To use the software simply clone the repository into a location of your choice.
Create an python-environment for the code to be set-up and libraries to be installed through issuing in a Terminal:
```
$python3 -m venv venv
```
Then you would activate the environment through:
```
$source venv/bin/activate
```
Then install the package through referencing the local code by:
```
(venv)$ pip install /<abs_path>/<to>/<blotter-transform>/
```
This should install all of the required dependencies and libraries.

#### Dependencies

The `pyproject.toml` file specifies the `build-system` and dependencies for this library. The following dependencies will be installed

* `pandas`: which classes will be extended, good for data-handling
* `scipy`: used for some statistical calulations
* `fuzzywuzzy` and `python-Levenstein` used in text-cleanup and inference.
* `pre-commit` allows for pre-commit hooks used for running tests and ensuring of code quality
* `pytest` for running unit tests
* `tox` for automatizing testing.

### How To Use

To run the simple testcase and to compute the daily return csv, create a new python file:
```
import blotter_transform as bt

in_path = ("/<abs-path>/<to>/blotter.csv")
out_path = ("/<abs-path>/<to>/blotter_out.csv")

bt.main(in_path, out_path)
```
The main-py is simply a dummy-function to show the functionality and calls some of the implemented functions.

### Functionality

#### Unit-Tests
For running the unit tests - `cd` to the repo-folder and issue:
```
$pytest
```
For more extensive testing including linting, coverage-report issue `$tox`.

### Considerations
#### Date and Text inference
For `date inference` computing the `z-score` is across all dates is being used in order to catch outliers such as spelling mistakes. High Z-scores indicate the distance to the mean relative of the dataset. This gives allows us to replace an outlier with the mean of neighboring values. However if the dataset is too polluted, hence we have too much messy data, this approach will fail.

For `text` inference we are using `fuzzywuzzy` library which is using the Levenshtein distance between values to calculate similarities between them. If a value is very similar to another and has low frequency, it will be replaced by the one with high frequency.

#### A package
Why a package? A package allows for easy sharability between projects and allows for a standardized way to encapsulate code. Its ease of use and structure make it suitable for future changes and implementations.
Furthermore, the package Framework could be published to PyPI.

### Limitations and next Steps
Some tests have been implemented for the `utils.py` , tests for both `BlotterSeries` and `BlotterDataframe` are missing, this is due to lack of time, but would be considered top-priority for next steps.

Functionalities could be better scoped-out and custom error and warning- messages implemented.

### TODOS

* More extensive unit-test providing for a higher coverage.
* Performance and completion on different, larger datasets?
