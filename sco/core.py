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
from   types                 import (FunctionType, TypeType, DictType)

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
def _argspec_options(argspec):
    (args, vargs, kwargs, defaults) = argspec
    if defaults is None: defaults = []
    if vargs is not None:
        raise ValueError('calculation/initialize/validate functions may not take variable args')
    # Ignore arg 1, as it's the object/self param
    dn = len(defaults)
    return {k:v for (k,v) in zip(args[-dn:], defaults[-dn:])}
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
        if not isinstance(da, DictType):
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
    if not isinstance(rval, DictType):
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
        opts = _argspec_options(argspec)
        def _calculates(*args, **kwargs):
            return _calculator_call(view, argspec, names, expc, prov, args, kwargs)
        _calculates.provision   = prov
        _calculates.expectation = expc
        _calculates.options     = opts
        _calculates.argspec     = argspec
        _calculates.calc_layers = None
        _calculates._calc       = True
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
def _flatten_calc_chain_args(args):
    llist = [
        ([arg]                         if isinstance(arg, DictType)      else
         [arg]                         if hasattr(arg, '__iter__') and
                                          len(arg) == 2 and
                                          isinstance(arg[0], basestring) else
         _flatten_calc_chain_args(arg) if hasattr(arg, '__iter__')       else
         [(arg.__name__, arg)])
        for arg in args]
    return [l for ll in llist for l in ll]
def calc_chain(*args, **kw):
    '''
    calc_chain(a, b, ...) yields a callable calculation module object that incorporates the given
    calculation functions a, b, etc. into a chain of calculations. The calculations are run
    in order, and any of the arguments may be either tuples or lists, in which case these are
    flattened. A tuple (string, calc) is considered a calc with a name; a calc not appearing in such
    a pair will be given the name according to it's __name__ attribute.
    A sequence of 1 or more dictionary arguments may follow the calculation objects, as well as
    a sequence of keywords, all of which will be merged left-to-right into a single dictionary and
    used to replace the elements of the given chain; the keys in the replacement dictionary may be
    either the calculations to replace themselves or their names, and the replacements must also be
    calculations.
    '''
    calc_layers = []
    repl = None
    for arg in _flatten_calc_chain_args(args):
        if isinstance(arg, DictType):
            if repl is None: repl = {}
            repl.update(arg)
        elif repl is None:
            calc_layers.append(arg)
        else:
            raise ValueError('All replacements must come after calculations')
    if len(calc_layers) == 0: raise ValueError('calc_chain() requires at least one argument.')
    # Make replacements
    repl = {} if repl is None else repl
    repl.update(kw)
    calc_layers = [
        (nm, (repl[nm] if nm in repl else
              repl[m]  if m  in repl else
              m))
        for (nm,m) in calc_layers]
    modules = [m for (nm,m) in calc_layers]
    # Check the calc layers
    for (nm,m) in calc_layers:
        if not iscalc(m):
            raise ValueError('given object %s (%s) is not a calculation!' % (nm, m))
    # Calculate the expectation and provision;
    # The expectation is tricky: it's everything expected prior to being provided
    prov_net = set([])
    expc_net = set([])
    opts_net = {}
    for (nm,m) in calc_layers:
        expc_net.update(m.expectation - prov_net)
        for (o,v) in m.options.iteritems():
            if o not in prov_net and o not in opts_net:
                opts_net[o] = v
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
    _calc_chain.provision   = prov_net
    _calc_chain.options     = opts_net
    _calc_chain._calc       = True
    _calc_chain.calc_layers = calc_layers
    return _calc_chain

def calc_translate(calc, *args, **kw):
    '''
    calc_translate(calc, replacements...) yields a version of the calculation calc in which the
    expectations and provisions have been changed to the given replacements but which otherwise
    functions identically. If calc is a list or tuple, then it is passed to calc_chain first.
    The replacements arguments may be supplied as zero 0 or more dictionaries of replacements
    followed by zero or more keyword arguments. These are merged left-to-right.
    '''
    if isinstance(calc, (tuple, list)): calc = calc_chain(calc)
    # Merge the dictionaries
    itr = _merge_dictargs(args, kw)
    ftr = {v:k for (k,v) in itr.iteritems()}
    # This is a pretty easy wrapper actually...
    def _calc_translate(*dictargs, **kwargs):
        # Translate...
        data_pool = {(ftr[k] if k in ftr else k):v
                     for (k,v) in _merge_dictargs(dictargs, kwargs).iteritems()}
        rval = calc(data_pool)
        rval = {(itr[k] if k in itr else k):v for (k,v) in rval.iteritems()}
        return rval
    _calc_translate.expectation = set([itr[k] if k in itr else k   for k     in calc.expectation])
    _calc_translate.provision   = set([itr[k] if k in itr else k   for k     in calc.provision])
    _calc_translate.options     = {(itr[k] if k in itr else k):v
                                   for (k,v) in calc.options.iteritems()}
    _calc_translate._calc       = True
    _calc_translate.calc_layers = None
    return _calc_translate
        
        
