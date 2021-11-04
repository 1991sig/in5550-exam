"""Runners for training and evaluation of models."""
import torch
from .AbstractRunner import Runner
from NSR.Utils import override
import tqdm


class MultiClassRunner(Runner):
    """Run training and evaluation workloads for multi-label problems.

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
        I.e. the values in the label field vocab.

    """

    def __init__(self, model, criterion, optimizer, labels):
        """Initialize the MultiClassRunner."""
        super().__init__(model, criterion, optimizer, labels)

    @override
    def get_accuracy(self, y_hat, y):
        """Compute global accuracy."""
        correct = (y_hat == y).nonzero().size(0)
        return correct / y_hat.size(1)

    def train(self, iters):
        """Train over batched data.

        Parameters
        ----------
        iters : torchtext.data.iterator.BucketIterator
            The batched data

        Returns
        -------
        dict
            The performance and metrics of the training session.

        """
        epoch_loss = 0

        self.model.train()
        for batch in tqdm.tqdm(iters):
            y_tilde_b = self.model(batch.form, batch.cue).transpose(1, 2)
            loss = self.criterion(y_tilde_b, batch.label)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        results_train = {
            "loss": epoch_loss / len(iters),
        }

        return results_train

    def evaluate(self, iters):
        """Evaluate over batched data.

        Parameters
        ----------
        iters : torchtext.data.iterator.BucketIterator
            The batched data

        Returns
        -------
        dict
            The performance and metrics of the training session.

        """
        tp = 0
        n = 0

        device = iters.device

        y_hat = torch.tensor([], dtype=torch.long).to(device)
        y = torch.tensor([], dtype=torch.long).to(device)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(iters):
                y_tilde_b = self.model(batch.form, batch.cue)
                y_hat_b = y_tilde_b.argmax(dim=-1)
                y_b = batch.label

                tp += (y_hat_b == y_b).nonzero().size(0)
                n += y_b.size(1)

                y_hat = torch.cat((y_hat, y_hat_b.view(-1)))
                y = torch.cat((y, y_b.view(-1)))

        results_eval = {
            "accuracy": tp / n,
            **self.get_metrics(y.cpu(), y_hat.cpu())
        }

        return results_eval
