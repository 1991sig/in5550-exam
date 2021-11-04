"""Parser for JSON configuration files."""
import os
import json
from collections import OrderedDict
from pathlib import Path


class Process():
    """Setup processor for PyTorch experiments.

    Parses the runtime options set in the JSON file
    at the path given to the argparser.

    Inputs
    ------
    args : ArgumentParser
        Argparser with arguments
    """

    def __init__(self, args):
        """Parse JSON to OrderedDict and setup directories."""
        p = Path(args.setupfile)
        with p.open("rt") as f:
            self.setup = json.load(f, object_hook=OrderedDict)

        key = self.setup["identifier"]
        outdir = Path(self.setup["resultsdir"])

        outdir.mkdir(parents=True, exist_ok=True)

        ID = self.setup["ID"]

        self.destdir = outdir / key / str(ID)

        if os.path.exists(self.destdir) and os.path.isdir(self.destdir):
            if os.listdir(self.destdir):
                m1 = "A directory with this name and key already exists.\n"
                m2 = "Either use another name/key, or run incremental"
                m3 = "training on existing models.\n"
                raise Exception(m1+m2+m3)

        self.destdir.mkdir(parents=True, exist_ok=True)

        jsonout = Path(self.destdir / "setup.json")
        with jsonout.open("wt") as f:
            json.dump(self.setup, f, indent=4, sort_keys=False)

    def load(self, argname, module, *args, **kwargs):
        """Return objects specified in configfile.

        For `argname` in configile, get the attribute specified by `type`
        from `module` called with args and kwargs.

        Inputs
        ------
        argname : str
            The argument name as specified in config file
        module : Python module
            The module where the object/attribute is contained
        args :
            Positional args to send
        kwargs :
            Keyword arguments to send

        Returns
        -------
        Attribute initialized by args and kwargs

        """
        item = self.setup[argname]["type"]
        item_args = dict(self.setup[argname]["args"])
        if sum([kwarg in item_args for kwarg in kwargs]) != 0:
            raise ValueError("Args set in config file cannot be overwritten")
        item_args.update(kwargs)
        return getattr(module, item)(*args, **item_args)

    def __getitem__(self, argname):
        """Get the entry for `argname`."""
        return self.setup[argname]
