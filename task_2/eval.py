def _normalize_name(name):
    name = name.strip().lower()
    if name.startswith("<") and name.endswith(">") and len(name) >= 2:
        return name[1:-1]
    return name


def novelty_rate(generated, training_set):
    if not generated:
        return 0.0

    normalized_training = {_normalize_name(n) for n in training_set}
    normalized_generated = [_normalize_name(n) for n in generated]
    return len([n for n in normalized_generated if n not in normalized_training]) / len(
        normalized_generated
    )


def diversity(generated):
    if not generated:
        return 0.0

    normalized_generated = [_normalize_name(n) for n in generated]
    return len(set(normalized_generated)) / len(normalized_generated)


def evaluate_generated_names(generated, training_set):
    return {
        "novelty": novelty_rate(generated, training_set),
        "diversity": diversity(generated),
    }
