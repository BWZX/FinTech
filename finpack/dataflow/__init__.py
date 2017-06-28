from pkgutil import iter_modules
import os
import os.path

from . import augs

__all__ = ['augs']

def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)

__SKIP = ['augs']
for _, module_name, __ in iter_modules(
        [os.path.dirname(__file__)]):
    if not module_name.startswith('_') and \
    		module_name not in __SKIP:
        _global_import(module_name)
