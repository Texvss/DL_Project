from datetime import datetime
import comet_ml

class CometMLWriter:
    """
    Class for experiment tracking via CometML, customized for your project.
    """

    def __init__(
        self,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            workspace (str | None): name of the workspace inside experiment tracker.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name is given.
            mode (str): if online, log data to the remote server. If offline, log locally.
        """
        try:
            comet_ml.login()

            self.run_id = run_id

            resume = False
            if project_config.get("trainer", {}).get("resume_from") is not None:
                resume = True

            if resume:
                if mode == "offline":
                    exp_class = comet_ml.ExistingOfflineExperiment
                else:
                    exp_class = comet_ml.ExistingExperiment
                self.exp = exp_class(experiment_key=self.run_id)
            else:
                if mode == "offline":
                    exp_class = comet_ml.OfflineExperiment
                else:
                    exp_class = comet_ml.Experiment
                self.exp = exp_class(
                    project_name=project_name,
                    workspace=workspace,
                    experiment_key=self.run_id,
                    log_code=kwargs.get("log_code", False),
                    log_graph=kwargs.get("log_graph", False),
                    auto_metric_logging=kwargs.get("auto_metric_logging", False),
                    auto_param_logging=kwargs.get("auto_param_logging", False),
                )
                self.exp.set_name(run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                self.exp.log_parameters(parameters=project_config)

        except ImportError:
            print("For use comet_ml install it via \n\t pip install comet_ml")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", (self.step - previous_step) / duration.total_seconds())
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name with the current mode.
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.
        """
        self.exp.log_model(name="checkpoints", file_or_folder=checkpoint_path, overwrite=True)

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.
        """
        self.exp.log_metrics({self._object_name(scalar_name): scalar}, step=self.step)

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.
        """
        self.exp.log_metrics({self._object_name(k): v for k, v in scalars.items()}, step=self.step)