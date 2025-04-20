# Introduction

This repository presents my research work on the topic: **Detection of anomalies
in multivariate time series using artificial intelligence methods**.

The data used in this research comes from NASA's Wind satellite and is publicly
available at the following link:
https://cdaweb.gsfc.nasa.gov/pub/data/wind/waves/wav_h1/2020/

The research was conducted in collaboration with astrophysicists from the Basic
Department of Space Physics at the Space Research Institute of the Russian
Academy of Sciences.

This research does not yet have scientific publications and requires additional
verification by specialists.

# Repository Structure

```
|-- src
    |-- experiments
        |-- baseline
            |-- test.ipynb
        |-- cv/train
            |-- classification_cv.ipynb
        |-- extended_data
            |-- extended_data.ipynb    
            |-- test.ipynb
    |-- preprocessing.ipynb
|-- text
    |-- code
        |-- classic.py
        |-- cv.py
        |-- preprocessing.py
        |-- processing.py
        |-- spectrograms.py
    |-- pic
        |-- ...images...
    |-- diploma.pdf
    |-- diploma.tex
```

# Instructions for Reproducing Research Results

These instructions describe the process of reproducing the research results
presented in this repository. The instructions contain a step-by-step guide for
launching all stages of data processing, model training, and testing results.

## Preparing the Working Environment

### Cloning the Repository

First, you need to clone the repository with the source code:

```bash
git clone https://github.com/QQQiwi/diploma.git
cd diploma
```

### Installing Dependencies

Before running the scripts, you need to install all required dependencies. The
repository may contain a requirements.txt file with a list of necessary
libraries:

```bash
pip install -r requirements.txt
```

If this file is not available in the repository, it is recommended to install
the following basic libraries for working with data and training models:

```bash
pip install numpy pandas matplotlib scikit-learn transformers spacepy
```

## Steps to Reproduce Results

The code that needs to be executed is located in the `text\code` directory. It
is necessary to study the main functions of each script, as the source data for
correct execution of the code must be named in a specific way. The main code
fragments of the scripts contain detailed descriptions of the functions used.

### Data Preprocessing

The first step is to convert the original data from CDF format to CSV:

```bash
python preprocessing.py
```

This script performs the following tasks:
- Reading source files in CDF format
- Extracting necessary parameters and values
- Converting data to CSV format
- Saving processed data for further use

### Data Processing and Sample Preparation

After converting the data, it is necessary to process, balance, and divide it
into training and test samples:

```bash
python processing.py
```

This script performs:
- Loading converted data from CSV files
- Cleaning data from outliers and incorrect values
- Balancing the sample for uniform class representation
- Dividing data into training and test samples
- Saving prepared datasets

### Creating Spectrograms for Computer Vision Algorithms

To use computer vision algorithms, it is necessary to create spectrograms:

```bash
python spectrograms.py
```

This script performs:
- Loading preprocessed data
- Converting time series to spectrograms
- Creating and saving a dataset for training computer vision models

### Training and Testing Classical Machine Learning Models

To start the process of training and testing classical machine learning models,
use the script:

```bash
python classic.py
```

This file contains code for:
- Loading prepared data
- Training various machine learning models
- Evaluating model quality
- Visualizing results and saving metrics

### Training and Testing Computer Vision Models

The final stage includes training and testing computer vision models:

```bash
python cv.py
```

This script performs:
- Loading the dataset with spectrograms
- Training computer vision models
- Evaluating model quality
- Visualizing results and saving metrics

## Results Analysis

After completing all stages of data processing and model training, the results
of each script will be saved in the corresponding directories of the repository.
You can analyze the obtained results by comparing them with those described in
the text of the work, which is also available in the repository.

## Possible Problems and Their Solutions

If you encounter problems reproducing the results, it is recommended to:
- Check the current versions of the libraries used
- Ensure all necessary source data is available
- Refer to the documentation in the repository or contact the author of the work

By following these instructions, you will be able to fully reproduce the
research results presented in the repository.

# Research Work Plan for the Diploma

1. Literature review on the topic of anomaly detection in multivariate time series.
2. Study of anomaly detection methods using artificial intelligence and selection of the most promising ones for my task.
3. Data preparation, applying additional preprocessing methods such as normalization and outlier removal.
4. Initial experiments for anomaly detection in the data.
5. Comparing the performance of different models on the data.
6. Performing hyperparameter optimization (GridSearch) to improve the performance of selected models.
7. Determining the best model.
8. Data visualization and reduction of the task to CV (Computer Vision).
9. The same research, experiments with models, and determination of the best one, but within the CV framework.
10. Attempting to improve quality: data augmentation, changing other preprocessing parameters of the models.
11. Final chapters of the thesis describing methodology, results, and discussion.
12. Completion of the practical part of the thesis.
13. Formatting the work, preparing the bibliography, tables, and figures.
14. Preparing a presentation for the defense.
15. Finishing work on the text, submitting for final edits.
16. Checking and formatting the work
