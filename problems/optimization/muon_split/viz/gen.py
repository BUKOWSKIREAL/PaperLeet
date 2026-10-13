from problems.optimization.muon_split.solution import headwise_orthogonalize
from problems.optimization.muon_split.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Muon Split: independent head scales", "problem": "optimization/muon_split",
            **run(headwise_orthogonalize)}
