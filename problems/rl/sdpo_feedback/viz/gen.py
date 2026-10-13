from problems.rl.sdpo_feedback.experiment import run
from problems.rl.sdpo_feedback.solution import select_feedback, feedback_reverse_kl

def generate():
    return run(select_feedback, feedback_reverse_kl)
