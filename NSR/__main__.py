"""Main script for training and evaluating models."""
import argparse
import random
import torch
import torchtext
from torchtext import data
import NSR

args = argparse.ArgumentParser(description="NSR - Negation Scope Resolution")
args.add_argument("-f", "--setupfile", default=None, type=str,
                  help="Path to JSON configuration file (default: None)")


def main(setup):
    """Execute job with arguments in setup."""
    ID = data.RawField(preprocessing=lambda x: int(x))
    SRC = data.RawField()
    NEGS = data.RawField(preprocessing=lambda x: int(x))
    FORM = data.Field(batch_first=True, include_lengths=True)
    LEMMA = data.RawField()
    XPOS = data.RawField()
    LABS = data.Field(batch_first=True)
    CUE = data.RawField()
    SCOPE = data.RawField()

    fields = {
        "id": ("id", ID),
        "source": ("source", SRC),
        "negations": ("negations", NEGS),
        "form": ("form", FORM),
        "lemma": ("lemma", LEMMA),
        "xpos": ("xpos", XPOS),
        "negation": ("label", LABS),
        "cue": ("cue", CUE),
        "scope": ("scope", SCOPE),
    }

    Xtrain, Xdev = NSR.StarSEM2012.splits(
        "DataFiles",
        fields=fields,
        test=None
    )

    vecs = torchtext.vocab.Vectors(setup["params"]["vectors"])

    FORM.build_vocab(Xtrain,
                     max_size=setup["params"]["vocab_size"],
                     min_freq=setup["params"]["min_freq"],
                     vectors=vecs)

    LABS.build_vocab(Xtrain)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vecs = FORM.vocab.vectors

    model = setup.load("model", NSR.Models,
                       input_dim=len(FORM.vocab),
                       embedding_dim=vecs.shape[-1],
                       output_dim=len(LABS.vocab),
                       vecs=vecs,
                       train_emb=setup["params"]["train_emb"]
                       ).to(device)

    optimizer = setup.load("optimizer", torch.optim, model.parameters())

    criterion = setup.load("criterion", torch.nn).to(device)

    runner = setup.load("runner", NSR.Runners, model,
                        criterion, optimizer, LABS.vocab.itos)

    batch_size = setup["params"]["batchsize"]
    epochs = setup["params"]["epochs"]

    trn_iter = data.BucketIterator(
        Xtrain,
        device=device,
        shuffle=True,
        batch_size=batch_size
    )

    val_iter = data.BucketIterator(
        Xdev,
        device=device,
        shuffle=True,
        batch_size=1
    )

    runner.run(epochs, trn_iter, val_iter)
    res = runner.evaluate(val_iter)

    print("*"*21)
    print("{} {}".format(setup["identifier"], setup["ID"]))
    print("\nResults")
    print(NSR.Utils.describe_stats(res))

    runner.save(dirpath=setup.destdir, checkpoint=True)


if __name__ == "__main__":
    args = args.parse_args()
    setup = NSR.Process(args)

    RSEED = setup["randomseed"]
    random.seed(RSEED)
    torch.manual_seed(RSEED)
    torch.backends.cudnn.deterministic = True
    main(setup)
