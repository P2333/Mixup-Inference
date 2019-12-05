import argparse as _argparse


class _my_argparse(_argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(_my_argparse, self).__init__(**kwargs)
        self.flag_for_overwrite = False

    def add_argument_overwrite(self, *args, **kwargs):
        kwargs = self._get_optional_kwargs(*args, **kwargs)
        for action in self._actions:
            if action.dest == kwargs["dest"]:
                for key in kwargs:
                    if hasattr(action, key):
                        action.__setattr__(key, kwargs[key])
                return action

        return self.add_argument(*args, **kwargs)

    def add_argument_general(self, *args, **kwargs):
        if self.flag_for_overwrite is True:
            return self.add_argument_overwrite(*args, **kwargs)
        else:
            return self.add_argument(*args, **kwargs)


class _FlagValues(object):
    """Global container and accessor for flags and their values."""

    def __init__(self):
        self.__dict__["__flags"] = {}
        self.__dict__["__actions"] = {}
        self.__dict__["__parsed"] = False

    def _parse_flags(self, args=None):
        result, unparsed = _global_parser.parse_known_args(args=args)
        for flag_name, val in vars(result).items():
            if val is MUST_INPUT:
                raise Exception(
                    "{} must be specified, can not be None.".format(flag_name))
            self.__dict__["__flags"][flag_name] = val
        self.__dict__["__parsed"] = True
        return unparsed

    def get_dict(self):
        if not self.__dict__["__parsed"]:
            self._parse_flags()

        return self.__dict__["__flags"]

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the flag --name."""
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        if name not in self.__dict__["__flags"]:
            raise AttributeError(name)
        return self.__dict__["__flags"][name]

    def __setattr__(self, name, value):
        """Sets the 'value' attribute of the flag --name."""
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        self.__dict__["__flags"][name] = value


SUPPRESS = _argparse.SUPPRESS
MUST_INPUT = None
_global_parser = _my_argparse()
FLAGS = _FlagValues()


def Enable_OverWrite():
    _global_parser.flag_for_overwrite = True


def Disable_OverWrite():
    _global_parser.flag_for_overwrite = False


def DEFINE_argument(*args, default=MUST_INPUT, rep=False, **kwargs):

    if rep is True:
        kwargs["nargs"] = "+"

    _global_parser.add_argument_general(*args, default=default, **kwargs)


def DEFINE_boolean(*args, default=MUST_INPUT, docstring=None, **kwargs):
    """Defines a flag of type 'boolean'.

    Args:
      flag_name: The name of the flag as a string.
      default_value: The default value the flag should take as a boolean.
      docstring: A helpful message explaining the use of the flag.
    """

    # Register a custom function for 'bool' so --flag=True works.
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    docstring = "" if docstring is None else docstring
    _global_parser.add_argument_general(*args,
                                        help=docstring,
                                        default=default,
                                        type=str2bool,
                                        **kwargs)
