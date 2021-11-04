"""Abstract Runner Class for training and evaluation of models."""
from abc import ABC, abstractmethod
import torch
import os
import copy
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from NSR.Utils import append2dict


class Runner(ABC):
    """Run training and evaluation workloads.

    Arguments
    ---------
    model: torch.nn.Module
        Model to optimize or test.
    criterion: torch.optim
        Loss function.
    optimizer: torch.optim
        Optimizer for weights and biases.
    labels: list
        List containing the labels in index form.
        I.e. the keys in the label field vocab.

    """

    def __init__(self, model, criterion, optimizer, labels):
        """Initialize the runner."""
        self.model = model
        self.best_model = None
        self.checkpoint = {'epoch': None,
                           'model_name': None,
                           'model_state_dict': None,
                           'optimizer_name': None,
                           'optimizer_state_dict': None,
                           'train': None,
                           'eval': None
                           }
        self.criterion = criterion
        self.optimizer = optimizer
        self.labelvals = labels
        self.labels = [*range(len(labels))]

        # Performances
        self.reset()

    @abstractmethod
    def train(self, iterator):
        """Train over batched data.

        Parameters
        ----------
        iterator : torchtext.data.iterator.BucketIterator
            The batched data

        Returns
        -------
        dict
            The performance and metrics of the training session.

        """
        raise Exception("method not implemented")

    @abstractmethod
    def evaluate(self, iterator):
        """Evaluate one time the model on iterator data.

        Parameters
        ----------
        iterator : torchtext.data.iterator.BucketIterator
            Iterator containing batch samples of data. Each batch must have
            text and label fields.

        Returns
        -------
        dict
            The performance and metrics of the evaluation session.

        """
        raise Exception("method not implemented")

    def _update_checkpoint(self, epoch, results_train=None, results_eval=None):
        """Update the training checkpoint of the model.

        Parameters
        ----------
        epoch : int
            Epoch at the current training state.
        results_train : dict, optional
            Metrics for the training session at epoch. The default is None.
        results_eval : dict, optional
            Metrics for the evaluation session at epoch. The default is None.

        Returns
        -------
        None

        """
        self.best_model = copy.deepcopy(self.model)
        self.checkpoint = {"epoch": epoch,
                           "model_name": type(self.best_model).__name__,
                           "model_state_dict": self.best_model.state_dict(),
                           "optimizer_name": type(self.optimizer).__name__,
                           "optimizer_state_dict": self.optimizer.state_dict()
                           }
        self.checkpoint_stats = {"train": results_train,
                                 "eval": results_eval}

    def save(self, dirpath=".", checkpoint=True):
        """Save the model.

        Parameters
        ----------
        filename : str, optional
            Name to save model as
        dirpath : str, optional
            Path to folder to save to

        Returns
        -------
        None

        """
        filename = "model.pt"
        path = os.path.join(dirpath, filename)
        torch.save(self.best_model, path)
        torch.save(self.performance, os.path.join(dirpath, "stats"))

        if checkpoint:
            checkname = "model_epoch"+str(self.checkpoint['epoch']) + '.pt'
            torch.save(self.checkpoint, os.path.join(dirpath, checkname))
            torch.save(self.checkpoint_stats,
                       os.path.join(dirpath, "checkpoint_stats"))

    def reset(self):
        """Clear the metrics for the current run.

        Returns
        -------
        None

        """
        self.performance = {"train": {"loss": [],
                                      "accuracy": [],
                                      "precision": [],
                                      "recall": [],
                                      "macro_f1": [],
                                      "confusion_matrix": []},
                            "eval":  {"loss": [],
                                      "accuracy": [],
                                      "precision": [],
                                      "recall": [],
                                      "macro_f1": [],
                                      "confusion_matrix": []}}

    def get_accuracy(self, y_hat, y):
        """Compute accuracy from predicted classes and gold labels."""
        correct = (y_hat == y).nonzero().size(0)
        return correct / y_hat.size(0)

    def get_metrics(self, y, y_hat):
        """Evaluate metrics for the provided labels and predictions.

        Parameters
        ----------
        y : torch.tensor
            The true outcomes
        y_hat : torch.tensor
            The predicted outcomes
        labs : list
            The labels

        Returns
        -------
        metrics : dict
            - precision: float
            - recall: float
            - macro_f1: float
            - confusion matrix: list(list(int))

        """
        prec = float(precision_score(y, y_hat,
                                     labels=self.labels, average='macro'))
        rec = float(recall_score(y, y_hat,
                                 labels=self.labels, average='macro'))
        mac = float(f1_score(y, y_hat,
                             labels=self.labels, average='macro'))
        confmat = confusion_matrix(y, y_hat,
                                   labels=self.labels).tolist()
        metrics = {"precision": prec,
                   "recall":    rec,
                   "macro_f1":  mac,
                   "confusion_matrix": confmat}

        return metrics

    def run(self, epochs, train_iter, eval_iter, *args, **kwargs):
        """Run training and eval job.

        Parameters
        ----------
        epochs : int
            Number of full iterations over the data
        train_iter : torchtext.data.iterator.BucketIterator
            Batched training data
        eval_iter : torchtext.data.iterator.BucketIterator, optional
            Batched eval data
        verbose : bool, optional
            If `True` display a progress bar and metrics at each epoch.
            The default is True.

        Returns
        -------
        None

        """
        eval_f1_star = 0
        n_no_improve = 0

        for epoch in range(epochs):

            train_res = self.train(train_iter, *args, **kwargs)
            eval_res = self.evaluate(eval_iter, *args, **kwargs)

            print(eval_res)

            append2dict(self.performance["train"],
                        train_res)
            append2dict(self.performance["eval"],
                        eval_res)

            if eval_f1_star < self.performance["eval"]["macro_f1"][-1]:
                eval_f1_star = self.performance["eval"]["macro_f1"][-1]
                self._update_checkpoint(epoch+1, train_res, eval_res)
                n_no_improve = 0
            else:
                n_no_improve += 1

            if n_no_improve == 5:
                print("Stopping after no improvement for 5 epochs")
                break
