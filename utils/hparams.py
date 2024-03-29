import yaml

def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)

    hparam_dict = dict()
    for doc in docs:
        for key, value in doc.items():
            hparam_dict[key] = value

    return hparam_dict


class Dotdict(dict):
    """
     a dictionary that supports dot notation
     as well as dictionary access notation
     usage: d = DotDict() or d = DotDict({'val1':'first'})
     set attributes: d.val2 = 'second' or d['val2'] = 'second'
     get attributes: d.val2 or d['val2']
     """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        super(Dotdict, self).__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class HParam(Dotdict):
    def __init__(self, filename):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(filename)
        hp_dotdict = Dotdict(hp_dict)
        for key, value in hp_dotdict.items():
            setattr(self, key, value)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__