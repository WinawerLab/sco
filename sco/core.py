####################################################################################################
# core.py
# The code that handles modules for the SCO project.
# By Noah C. Benson


import numpy                 as     np
import scipy                 as     sp
import neuropythy            as     neuro
import nibabel.freesurfer.io as     fsio

from   neuropythy.immutable  import Immutable
from   numbers               import Number
from   pysistence            import make_dict
from   functools             import wraps
from   types                 import (FunctionType, TypeType)

import os, math, itertools, collections, abc, inspect


####################################################################################################
# Private Functions
def _check_names(names, where):
    for name in names:
        if not isinstance(name, basestring):
            raise ValueError('value %s from %s must be an identifier' % (name, where))
    return names
def _argspec_expects(argspec):
    (args, vargs, kwargs, defaults) = argspec
    if defaults is None: defaults = []
    if vargs is not None:
        raise ValueError('calculation/initialize/validate functions may not take variable args')
    # Ignore arg 1, as it's the object/self param
    return set(args[:-len(defaults)] if len(defaults) > 0 else args)
def _expectation(f):
    return _argspec_expects(inspect.getargspec(f))
def _provision(f):
    # All expectations are de facto provisions
    outputs = set(f.expectation if hasattr(f, 'expectation') else _expectation(f))
    if hasattr(f, 'provision'): outputs.update(f.provision)
    return outputs
def _prep_argspec(argspec, data_pool):
    # look at the function's argument declaration:
    (argnames, vargs, vkwargs, defaults) = argspec
    if defaults is None: defaults = []
    dfltargs = argnames[-len(defaults):]
    argnames = argnames[:-len(defaults)] if len(defaults) > 0 else argnames
    args = [data_pool[arg] for arg in argnames] + \
           [(data_pool[arg] if arg in data_pool else val) for (arg,val) in zip(dfltargs,defaults)]
    return (args, {} if vkwargs is None else data_pool)
def _argspec_add_defaults(argspec, data_pool):
    (argnames, vargs, kwargs, defaults) = argspec
    if defaults is None: defaults = []
    dn = len(defaults)
    for (k,v) in zip(argnames[-dn:], defaults):
        if k not in data_pool:
            data_pool[k] = v
    return data_pool
def _merge_dictargs(dictargs, kwargs):
    data_pool = {}
    for da in dictargs:
        if not isinstance(da, dict):
            raise ValueError('Calculations only accepts dictionaries and keywords as parameters')
        data_pool.update(da)
    data_pool.update(kwargs)
    return data_pool
def _calculator_call(f, argspec, names, expc, prov, args, kwargs):
    data_pool = _merge_dictargs(args, kwargs)
    missing = tuple(expc - set(data_pool.keys()))
    if len(missing) > 0: raise ValueError('Expected arguments missing: %s' % (missing,))
    (params, kwparams) = _prep_argspec(argspec, data_pool)
    rval = f(*params, **kwparams)
    if not isinstance(rval, dict):
       raise ValueError('Function with @provides decorator failed to return dict')
    for name in names:
        if name not in rval:
            raise ValueError('Function @provides %s but failed to return it' % name)
    rval = _argspec_add_defaults(argspec, rval)
    data_pool.update(rval)
    missing = tuple(set(prov - set(data_pool.keys())))
    if len(missing) > 0:
        raise ValueError('Provided values missing: %s' % (missing,))
    return data_pool



####################################################################################################
# Decorators
def calculates(*names, **argtr):
    '''
    @calculates(a,b...) declares that the function that follows must yield a dictionary containing 
    the data keys with the names given in a, b, etc. This annotation makes the function into a 
    calculation module; as such, it cannot take a varargs argument; though it can take a keyword
    argument (though such use is discouraged).
    Keyword options may be given to calculate to indicate that the arguments should be translated
    to expect a different name than the argument name given; for example:
     @calculates('z', arg1='x', arg2='y')
     def f(arg1, arg2):
        return {'z': arg1 + arg2}
    is equivalent to:
     @calculates('z')
     def f(x, y):
        return {'z': x + y}
    Note that these translations only work for normal parameters, not **kwargs-style parameters.
    '''
    if not all(isinstance(name, basestring) for name in names):
        raise ValueError('@provides decorator requires strings as the arguments')
    def _calc_dec(view):
        if hasattr(view, '_calc'):
            view.provision.update(names)
            return view
        argspec = [a for a in inspect.getargspec(view)]
        argspec[0] = [argtr[a] if a in argtr else a for a in argspec[0]]
        expc = _argspec_expects(argspec)
        prov = set(names) | set(argspec[0])
        def _calculates(*args, **kwargs):
            return _calculator_call(view, argspec, names, expc, prov, args, kwargs)
        _calculates.provision = prov
        _calculates.expectation = expc
        _calculates._calc = True
        _calculates.argspec = argspec
        return _calculates
    return _calc_dec

####################################################################################################
# Miscellaneous Utilities
def iscalc(obj):
    '''
    iscalc(f) yields True if f is a function that has been decorated with the @calculation form;
    i.e., a function that can be used as a calculation in the sco calc library. Otherwise, yields
    False.
    '''
    return all(hasattr(obj, x) for x in ['_calc', 'provision', 'expectation'])
def calc_chain(*args):
    '''
    calc_chain(a, b, ...) yields a callable calculation module object that incorporates the given
    calculation functions a, b, etc. into a chain of calculations. The calculations are run
    in order, and any of the arguments may be either tuples or lists, in which case these are
    flattened.
    '''
    if len(args) == 0: raise ValueError('calc_chain() requires at least one argument.')
    modules = [calc_chain(*a) if isinstance(a, (list, tuple)) else a for a in args]
    for m in modules:
        if not iscalc(m):
            raise ValueError('given object %s is not a calculation!' % m)
    # Calculate the expectation and provision;
    # The expectation is tricky: it's everything expected prior to being provided
    prov_net = set([])
    expc_net = set([])
    for m in modules:
        expc_net.update(m.expectation - prov_net)
        prov_net.update(m.provision)
    # Okay, now we make a wrapper:
    def _calc_chain(*dictargs, **kwargs):
        data_pool = _merge_dictargs(dictargs, kwargs)
        missing = tuple(expc_net - set(data_pool.keys()))
        if len(missing) > 0: raise ValueError('Expected arguments missing: %s' % (missing,))
        for m in modules:
            data_pool.update(m(data_pool))
        missing = tuple(prov_net - set(data_pool.keys()))
        if len(missing) > 0:
            raise ValueError('Provided values missing: %s' % (missing,))
        return data_pool
    _calc_chain.expectation = expc_net
    _calc_chain.provision = prov_net
    _calc_chain._calc = True
    _calc_chain.calc_modules = modules
    return _calc_chain

        
