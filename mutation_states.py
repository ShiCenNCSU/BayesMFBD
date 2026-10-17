def mutation_presence(value):
    """Encode a mutation probability as absent (0) or present (1)."""
    return int(float(value) >= 0.5)
