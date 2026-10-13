"""Reference-only preview; judge artifacts separately execute the learner code."""
from pathlib import Path
import yaml
from paperleet.judge.loader import load_implementation


def generate():
    directory = Path(__file__).parents[1]
    metadata = yaml.safe_load((directory / "problem.yaml").read_text())
    impl = load_implementation(directory / "solution.py")
    toy = load_implementation(directory / "toy.py")
    result = toy.run_bimodal(impl)
    provenance = metadata['toy']
    return {"id": metadata['id'], "kind": "toy_experiment", "title": metadata['title_en'],
            "provenance": {"kind": provenance['provenance'], "source": provenance['source'],
                           "adaptation": provenance['description_en']}, **result}
