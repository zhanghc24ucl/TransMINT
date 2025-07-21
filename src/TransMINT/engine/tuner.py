import optuna
from optuna.samplers import TPESampler

class Tuner:
    def __init__(self, train_config, tune_config, data_builder):
        self.sampler = TPESampler()
        self.data_builder = data_builder

        self._lr_range = tune_config.lr_range

    def _objective(self, trial):
        hparams = suggest_hparams(trial)

        cfg = self.config.deepcopy()
        cfg[''].learning_rate = trial.suggest_float('learning_rate', *self._lr_range)
        batch_size = trial.suggest_int('batch_size', 32, 256)

        train_dataloader, valid_dataloader, test_dataloader = \
            self.data_builder.build_dataloaders(batch_size=batch_size)

        # Init your training loop abstraction
        trainer = Trainer(cfg, train_dataloader, valid_dataloader, trial=trial)  # pass `trial` if you support pruning
        trainer.fit()

        # Assume `trainer.evaluate` returns the validation Sharpe (the *higher* the better)
        val_score = trainer.evaluate(test_dataloader)

        return val_score  # Optuna will maximise by default if we set direction="maximize"

    def tune(self):
        study = optuna.create_study(sampler=self.sampler)
