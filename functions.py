import numpy as n

def to_float(val):
    try:
        return float(val) # resolvi converter para float por ter números float na tabela
    except (ValueError, TypeError):
        return val