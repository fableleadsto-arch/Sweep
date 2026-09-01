"""Tool runners for the capability engine.

Each module implements one domain (numerics, data, ML, symbolic, vision,
NLP, graphs, AI frameworks) and every framework import is lazy — importing
this package never imports NumPy, let alone PyTorch. The engine in
`companion/capabilities.py` maps capabilities to these runners.
"""
