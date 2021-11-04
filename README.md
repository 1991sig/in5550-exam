
Despite my best efforts, I was not able to complete everything 100% yet. All the models, evaluations, and etc are done but I need to do some cleaning here and there for things to be reproducible. I will do this ASAP

# in5550-exam
Repository for the home exam

# NSR

Negation Scope Resolution (NSR) contains:
1. Data
    The StarSEM2012 dataset is defined as a `torchtext.data.Dataset`, and the class code is based on the generic class examples in the `torchtext` package.

    The data fields are defined outside the class, so experimenting with different data processing and structuring is easy.

2. Models
    2.1 BiLSTM - Bidirectional LSTM model based on the implementation in Fancellu et al. (2016)
    Intended baseline model.

3. Runners
    Contains an abstract base class containing the common functionality.
    3.2 MultiClassRunner - A runner for performing multiclass classification/sequence tagging.

4. Utils
    General utility functionality used here and there in the other modules.

5. Process
    5.1 Process - A class for representing the configurations of experiments and initiating the parts needed for an experiment. Enables easy experimentation with different hyperparameters and so on, and makes it easy to save and load models as well as the configurations they were run with. This is inspired by the `pytorch-template` repo.


# Negation

Contains the "starting package" for the exam. The most important parts used in the NSR package is the datasets (`cdd.epe`, `cdt.epe`, `cde.epe`) used in the NSR package, and the code for metric computations in line with the official ACL scoring.



