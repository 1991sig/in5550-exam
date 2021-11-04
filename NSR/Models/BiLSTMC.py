"""Bidirectional LSTM sequence labeller based on Fancellu et al. (2016)."""
import torch.nn as nn


class BiLSTMC(nn.Module):
    """Bidirectional LSTM model.

    Aims to replicate the BiLSTM-C model used by
    Fancellu et al. (2016).

    Parameters
    ----------
    input_dim : int
        Input dimensions
    embedding_dim : int
        Dimensionality of word embeddings
    n_neurons : int
        Number of neurons of the model
    output_dim : int
        Output dimensions
    n_layers : int
        Number of hidden layers
    bidir : bool
        Bidirectional if TRUE
    batch_first : bool
        If TRUE, inputs are assumed to have shape
            [batch size X # tokens]
        If FALSE:
            [# tokens X batch size]
    vecs : torch.Tensor
        A tensor with pre-trained word vectors
    train_emb : bool
        Whether to train the embedding layer further

    """

    def __init__(self, input_dim, embedding_dim, n_neurons, output_dim,
                 n_layers):
        """Initialize the model."""
        super().__init__()

        self.w_embedding = nn.Embedding(input_dim,
                                        embedding_dim)

        self.c_embedding = nn.Embedding(input_dim,
                                        embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            n_neurons,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        self.fc = nn.Linear(n_neurons * 2,
                            output_dim)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X, C):
        """Perform a forward pass.

        Given sample(s) X, predicts the raw/unscaled class probabilities, which
        can then be converted to probabilities by sigmoid or softmax.

        Input
        -----
        X : torch.Tensor or tuple(torch.Tensor, torch.Tensor)

        Returns
        -------
        The linear predicted values, a torch.tensor
        with dimension [batchsize X `output_dim`]

        """
        X, lens = X
        w_embs = self.w_embedding(X)
        c_embs = self.c_embedding(C)
        embs = w_embs + c_embs
        pack_emb = nn.utils.rnn.pack_padded_sequence(
            embs, lens, batch_first=True, enforce_sorted=False
        )
        pack_O, _ = self.lstm(pack_emb)
        O, _ = nn.utils.rnn.pad_packed_sequence(pack_O, batch_first=True)
        O_d = self.dropout(O)
        return self.fc(O_d)
