"""Connector registry: config source name -> Connector class."""

from .official import BLS, CFTC, EPU, FRED, GPR, NFCI, NasaPower, Shiller, Treasury, UMich
from .yahoo import YahooCrossAsset

CONNECTORS = {c.name: c for c in (YahooCrossAsset, Treasury, NFCI, UMich, BLS, Shiller, EPU, GPR, CFTC,
                                  NasaPower, FRED)}

__all__ = ["CONNECTORS"]
