"""*SEM2012 Dataset."""
import os
from torchtext import data


class StarSEM2012(data.Dataset):
    """Defines the representation for the *SEM2012 negation annotations."""

    @staticmethod
    def sort_key(ex):
        """Key for sorting samples during batch."""
        return len(ex.form)

    def __init__(self, path, fields, encoding="utf-8", sep="\n",
                 **kwargs):
        """Construct a *SEM2012 dataset.

        Parameters
        ----------
        path : str
            Path to the dataset file
        fields : iterable
            An iterable containing the fields to be used.
        encoding: str
            The encoding of the file
            Default: 'utf-8'
        sep : str
            String separator

        Returns
        -------
        A torchtext dataset object

        """
        examples = []

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                examples.append(data.Example.fromJSON(line, fields))

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(StarSEM2012, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, train='cdt.epe',
               validation='cdd.epe', test='cde.epe', **kwargs):
        """Construct the splits of the *SEM2012 dataset.

        Parameters
        ----------
        path : str
            Path to folder where data files are located
            Default : None
        train : str
            Training data file.
            Default : 'cdt.epe'
        validation : str
            Validation data file.
            Default : 'cdd.epe'
        test : str
            Test data file.
            Default : 'cde.epe'
        kwargs :
            Keyword arguments passed onto constructor

        Returns
        -------
        Torchtext dataset objects for each of the provided paths

        """
        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
