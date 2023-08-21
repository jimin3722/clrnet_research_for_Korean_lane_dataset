import inspect

import six

# borrow from mmdetection


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str
    
print(type(Registry('datasets')))