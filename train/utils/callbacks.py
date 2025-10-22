from transformers import TrainerCallback


class LogMetricsCallback(TrainerCallback):
    def __init__(self, log_every, test_size):
        self.log_every = log_every
        self.test_size = test_size

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0 and state.global_step > 0:
            metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset.select(range(self.test_size)))
            print(f"Step {state.global_step}: metrics={metrics}")
