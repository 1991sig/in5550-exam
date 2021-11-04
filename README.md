# in5550-exam
Repository for the home exam. This covers the Negation Scope Resolution track.

I have used notebooks for conducting the experiments:

1. Data Inspection - Getting familiar with the data format. 
2. Data Processing - Using Apache Beam for pre-processing the data into the desired format
3. BiLSTM - Baseline model
4. BiLSTM-CE - Extended model using BERT as word embeddings
5. Error Inspection - Inspecting and evaluating performance

Most of the code has been implemented in a Python module, NSR. 

# NSR

Negation Scope Resolution (NSR) contains:
1. Data

    The StarSEM2012 dataset is defined as a `torchtext.data.Dataset`, and the class code is based on the generic class examples in the `torchtext` package.

    The data fields are defined outside the class, so experimenting with different data processing and structuring is easy.

2. Models

    * BiLSTM - Bidirectional LSTM model based on the implementation in Fancellu et al. (2016). Intended baseline model.

3. Runners
    Contains an abstract base class containing the common functionality.
    
    * MultiClassRunner - A runner for performing multiclass classification/sequence tagging.

4. Utils
    
    General utility functionality used here and there in the other modules.

5. Process
    
    * Process - A class for representing the configurations of experiments and initiating the parts needed for an experiment. Enables easy experimentation with different hyperparameters and so on, and makes it easy to save and load models as well as the configurations they were run with. This is inspired by the `pytorch-template` repo.


# Negation

Contains the "starting package" for the exam. The most important parts used in the NSR package is the datasets (`cdd.epe`, `cdt.epe`, `cde.epe`) used in the NSR package, and the code for metric computations in line with the official ACL scoring.



