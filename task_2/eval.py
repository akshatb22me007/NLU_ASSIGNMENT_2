def novelty_rate(generated, training_set):
    if not generated:
        return 0.0
    return len([n for n in generated if n not in training_set]) / len(generated)


def diversity(generated):
    if not generated:
        return 0.0
    return len(set(generated)) / len(generated)


def evaluate_generated_names(generated, training_set):
    return {
        "novelty": novelty_rate(generated, training_set),
        "diversity": diversity(generated),
    }
