"""Reuse audited receipt transport under a separate immutable I/O namespace."""
from hypothesis_scope_multiseed import transport as audited
from . import VARIANT,MODEL,EFFORT,PARAMETERS
BASE=(audited.ROOT/'experiments/runs'/VARIANT).resolve()

def configure():
    audited.BASE=BASE
    audited.VARIANT=VARIANT
    audited.MODEL=MODEL
    audited.EFFORT=EFFORT
    audited.P=PARAMETERS

def request(*args,**kwargs):
    configure()
    return audited.request(*args,**kwargs)

def pending():
    configure()
    return audited.pending()

def assign(*args,**kwargs):
    configure()
    return audited.assign(*args,**kwargs)

def complete(*args,**kwargs):
    configure()
    return audited.complete(*args,**kwargs)

def seal(*args,**kwargs):
    configure()
    return audited.seal(*args,**kwargs)
