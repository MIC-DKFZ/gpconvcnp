import matplotlib

matplotlib.use("agg")

import numpy as np
import os
import time
import plotly.graph_objs as go
import ot

import torch
from torch import nn, optim, distributions
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = False
if "CUDNN_DETERMINISTIC" in os.environ:
    if os.environ["CUDNN_DETERMINISTIC"] not in (0, False, "false", "FALSE", "False"):
        cudnn.benchmark = False
        cudnn.deterministic = True

from batchgenerators.dataloading import MultiThreadedAugmenter
from trixi.util import Config, ResultLogDict
from trixi.experiment import PytorchExperiment

from neuralprocess.util import set_seeds, tensor_to_loc_scale
from neuralprocess.experiment.util import (
    get_default_experiment_parser,
    run_experiment,
)
from neuralprocess.data import (
    GaussianProcessGenerator,
    WeaklyPeriodicKernel,
    StepFunctionGenerator,
    LotkaVolterraGenerator,
    FourierSeriesGenerator,
    TemperatureGenerator,
)
from neuralprocess.data.gp import GaussianKernel, WeaklyPeriodicKernel, Matern52Kernel
from neuralprocess.model import (
    NeuralProcess,
    AttentiveNeuralProcess,
    ConvCNP,
    ConvDeepSet,
    GPConvDeepSet,
    generic,
)


def make_defaults(representation_channels=128):

    DEFAULTS = Config(
        # Base
        name="NeuralProcessExperiment",
        description="Learn a distribution over functions with a Neural Process",
        n_epochs=600000,
        batch_size=256,
        seed=1,
        device="cuda",
        representation_channels=representation_channels,
        # Data
        generator=GaussianProcessGenerator,
        generator_kwargs=dict(
            kernel_type=GaussianKernel,
            kernel_kwargs=dict(lengthscale=0.5),
            x_range=[-3, 3],
            num_context=[3, 100],
            num_target=[3, 100],
            target_larger_than_context=True,
            target_includes_context=False,
            target_fixed_size=False,
            output_noise=0.0,  # additive noise on context values
            linspace=False,  # x values from linspace instead of uniform
            number_of_threads_in_multithreaded=1,  # will use MP if > 1
        ),
        # Model:
        # modules are instantiated first and then passed to the constructor
        # of the model
        model=NeuralProcess,
        model_kwargs=dict(distribution=distributions.Normal),
        modules=dict(prior_encoder=generic.MLP, decoder=generic.MLP),
        modules_kwargs=dict(
            prior_encoder=dict(
                in_channels=2,  # context in + out
                out_channels=2 * representation_channels,  # mean + sigma
                hidden_channels=128,
                hidden_layers=6,
            ),
            decoder=dict(
                in_channels=representation_channels + 1,  # sample + target in
                out_channels=2,  # mean + sigma
                hidden_channels=128,
                hidden_layers=6,
            ),
        ),
        output_transform_logvar=True,  # logvar to sigma transform on sigma outputs
        # Optimization
        optimizer=optim.Adam,
        optimizer_kwargs=dict(lr=1e-3),
        lr_min=1e-6,  # training stops when LR falls below this
        scheduler=optim.lr_scheduler.StepLR,
        scheduler_kwargs=dict(step_size=1000, gamma=0.995),
        scheduler_step_train=True,
        clip_grad=1e3,  # argument for nn.clip_grad_norm
        lr_warmup=0,  # linearly increase LR from 0 during first lr_warmup epochs
        # Logging
        backup_every=1000,
        validate_every=1000,
        show_every=100,
        num_samples=50,
        plot_y_range=[-3, 3],
        # Testing
        test_batches_single=100,
        test_batches_distribution=30,
        test_batches_diversity=100,
        test_batch_size=1024,
        test_num_context=[
            "random",
        ],  # can also have integers in this list
        test_num_context_random=[3, 100],
        test_num_target_single=100,
        test_num_target_distribution=100,
        test_num_target_diversity=100,
        test_latent_samples=100,  # use this many latent samples for NP and ANP
        test_single=True,
        test_distribution=True,
        test_diversity=True,
    )

    MODS = {}

    MODS["ATTENTION"] = Config(  # you also need to set DETERMINISTICENCODER for this
        model=AttentiveNeuralProcess,
        model_kwargs=dict(
            project_to=128,  # embed_dim in attention mechanism
            project_bias=True,
            in_channels=1,  # context and target in
            representation_channels=representation_channels,
        ),
        modules=dict(attention=nn.MultiheadAttention),
        modules_kwargs=dict(attention=dict(embed_dim=128, num_heads=8)),
    )

    MODS["CONVCNP"] = Config(
        model=ConvCNP,
        model_kwargs=dict(
            in_channels=1,
            out_channels=1,
            use_gp=False,
            learn_length_scale=True,
            init_length_scale=0.1,
            use_density=True,
            use_density_norm=True,
            points_per_unit=20,  # grid resolution
            range_padding=0.1,  # grid range extension
            grid_divisible_by=64,
        ),
        modules=dict(conv_net=generic.SimpleUNet),
        modules_kwargs=dict(
            conv_net=dict(
                in_channels=8,
                out_channels=8,
                num_blocks=6,
                input_bypass=True,  # input concatenated to output
            )
        ),
    )

    MODS["GPCONVCNP"] = Config(  # Requires CONVCNP
        model_kwargs=dict(use_gp=True, use_density_norm=False)
    )

    MODS["NOSAMPLE"] = Config(model_kwargs=dict(gp_sample_from_posterior=False))

    MODS["LEARNNOISE"] = Config(  # Requires GPCONVCNP
        model_kwargs=dict(
            gp_noise_learnable=True,
            gp_noise_init=-2.0,
        )
    )

    MODS["LEARNLAMBDA"] = Config(  # Requires GPCONVCNP
        model_kwargs=dict(gp_lambda_learnable=True)
    )

    MODS["MATERNKERNEL"] = Config(generator_kwargs=dict(kernel_type=Matern52Kernel))

    MODS["WEAKLYPERIODICKERNEL"] = Config(
        generator_kwargs=dict(kernel_type=WeaklyPeriodicKernel)
    )

    MODS["STEP"] = Config(
        generator=StepFunctionGenerator,
        generator_kwargs=dict(
            y_range=[-3, 3],
            number_of_steps=[3, 10],
            min_step_width=0.1,
            min_step_height=0.1,
        ),
    )

    MODS["FOURIER"] = Config(
        generator=FourierSeriesGenerator,
        generator_kwargs=dict(
            series_length=[10, 20],
            amplitude=[-1, 1],
            phase=[-1, 1],
            bias=[-1, 1],
            frequency_scale=1.0,
        ),
    )

    MODS["FOURIERSINGLE"] = Config(
        generator=FourierSeriesGenerator,
        generator_kwargs=dict(
            series_length=[1, 2],
            amplitude=[-2, 2],
            phase=[-1, 1],
            bias=[-1, 1],
            frequency_scale=[0.1, 2.0],
        ),
    )

    MODS["LOTKAVOLTERRA"] = Config(
        generator=LotkaVolterraGenerator,
        generator_kwargs=dict(
            num_context=[20, 80],
            num_target=[70, 150],
            number_of_threads_in_multithreaded=8,
            predator_init=[50, 100],
            prey_init=[100, 150],
            rate0=[0.005, 0.01],
            rate1=[0.5, 0.8],
            rate2=[0.5, 0.8],
            rate3=[0.005, 0.01],
            sequence_length=10000,
            y_rescale=0.01,
            x_rescale=0.1,
            max_time=100.0,
            max_population=500,
            super_sample=1.5,
            x_range=[0, 5],
        ),
        model_kwargs=dict(out_channels=2),
        modules_kwargs=dict(
            prior_encoder=dict(in_channels=3),
            decoder=dict(out_channels=4),
            deterministic_encoder=dict(in_channels=3),
        ),
        plot_y_range=[0, 3],
        test_num_context_random=[20, 80],
    )

    MODS["TEMPERATURE"] = Config(
        generator=TemperatureGenerator,
        generator_kwargs=dict(
            num_context=[20, 100],
            num_target=[20, 100],
            sequence_length=30 * 24,  # ca. 1 month
            x_range=(0, 3),
        ),
        test_num_context_random=[20, 100],
    )

    MODS["DETERMINISTICENCODER"] = Config(
        modules=dict(deterministic_encoder=generic.MLP),
        modules_kwargs=dict(
            deterministic_encoder=dict(
                in_channels=2,
                out_channels=representation_channels,
                hidden_channels=128,
                hidden_layers=6,
            ),
            decoder=dict(in_channels=2 * representation_channels + 1),
        ),
    )

    MODS["LONG"] = Config(n_epochs=1200000, scheduler_kwargs=dict(step_size=2000))

    return {"DEFAULTS": DEFAULTS}, MODS


class NeuralProcessExperiment(PytorchExperiment):
    def setup(self):

        set_seeds(self.config.seed, "cuda" in self.config.device)

        self.setup_model()
        self.setup_optimization()
        self.setup_data()

    def setup_model(self):

        modules = dict()
        for mod_name, mod_type in self.config.modules.items():
            modules[mod_name] = mod_type(**self.config.modules_kwargs[mod_name])
        modules.update(self.config.model_kwargs)
        self.model = self.config.model(**modules)

        self.clog.show_text(repr(self.model), "Model")

    def setup_optimization(self):

        self.optimizer = self.config.optimizer(
            self.model.parameters(), **self.config.optimizer_kwargs
        )
        self.scheduler = self.config.scheduler(
            self.optimizer, **self.config.scheduler_kwargs
        )

        if self.config.lr_warmup > 0:
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0

    def setup_data(self):

        self.generator = self.config.generator(
            self.config.batch_size, **self.config.generator_kwargs
        )
        if self.generator.number_of_threads_in_multithreaded > 1:
            self.generator = MultiThreadedAugmenter(
                self.generator,
                None,
                self.generator.number_of_threads_in_multithreaded,
                seeds=np.arange(self.generator.number_of_threads_in_multithreaded)
                + self.generator.number_of_threads_in_multithreaded * self.config.seed,
            )

    def _setup_internal(self):

        super()._setup_internal()

        # manually set up a ResultLogDict with running mean ability
        self.results.close()
        self.results = ResultLogDict(
            "results-log.json",
            base_dir=self.elog.result_dir,
            mode="w",
            running_mean_length=self.config.show_every,
        )

        # save modifications we made to config
        self.elog.save_config(self.config, "config")

    def prepare(self):

        # move everything to specified device
        for name, model in self.get_pytorch_modules().items():
            model.to(self.config.device)

    def train(self, epoch):

        self.model.train()
        self.optimizer.zero_grad()

        batch = next(self.generator)
        batch["epoch"] = epoch  # logging

        context_in = batch["context_in"].to(self.config.device)
        context_out = batch["context_out"].to(self.config.device)
        target_in = batch["target_in"].to(self.config.device)
        target_out = batch["target_out"].to(self.config.device)

        # forward
        # ------------------------------------
        # on rare occasions, the cholesky decomposition in the GP
        # fails, if it does, we just go to the next batch
        skip = False
        try:

            prediction = self.model(
                context_in,
                context_out,
                target_in,
                target_out,
                store_rep=False,
            )
            prediction = tensor_to_loc_scale(
                prediction,
                distributions.Normal,
                logvar_transform=self.config.output_transform_logvar,
                axis=2,
            )
            batch["prediction_mu"] = prediction.loc.detach().cpu().repeat(1, 1, 2)
            batch["prediction_sigma"] = prediction.scale.detach().cpu().repeat(1, 1, 2)

            # loss
            loss_recon = -prediction.log_prob(target_out)
            loss_recon = loss_recon.mean(0).sum()  # batch mean
            loss_total = loss_recon
            if hasattr(self.model, "prior"):
                loss_latent = distributions.kl_divergence(
                    self.model.posterior, self.model.prior
                )
                loss_latent = loss_latent.mean(0).sum()  # batch mean
                loss_total += loss_latent

            # backward, check for NaN gradients
            loss_total.backward()
            if (
                type(self.model) == ConvCNP
                and type(self.model.input_interpolation) == GPConvDeepSet
            ):
                for p in self.model.parameters():
                    if p.grad is not None and torch.any(torch.isnan(p.grad)):
                        skip = True
                        break

        except RuntimeError as re:

            skip = True

        # if Cholesky fails or we get NaN gradients, we skip this batch
        if not skip:
            if self.config.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            self.optimizer.step()
        else:
            self.print("Skipping batch at epoch {}".format(epoch))
            return

        # logging and LR updates
        batch["loss_recon"] = loss_recon.item()
        if hasattr(self.model, "prior"):
            batch["loss_latent"] = loss_latent.item()
            batch["loss_total"] = loss_total.item()
        self.log(batch, validate=False)
        self.step_params(loss_total.item(), epoch, val=False)

    def log(self, summary, validate=False):

        if validate:
            backup = True
            show = True
        else:
            backup = (summary["epoch"] + 1) % self.config.backup_every == 0
            show = (summary["epoch"] + 1) % self.config.show_every == 0

        # add_result logs to self.results and also plots
        for l in ("loss_recon", "loss_latent", "loss_total"):
            if l in summary:
                name = l
                if validate:
                    name += "_val"
                self.add_result(
                    summary[l],
                    name,
                    summary["epoch"],
                    "Loss",
                    plot_result=show,
                    plot_running_mean=not validate,
                )

        self.make_plots(summary, save=backup, show=show, validate=validate)

    def make_plots(self, summary, save=False, show=True, validate=False):

        if not save and not show:
            return

        if hasattr(self.generator, "x_range"):
            x_range = self.generator.x_range
        elif hasattr(self.generator.generator, "x_range"):
            x_range = self.generator.generator.x_range
        else:
            x_range = [-3, 3]

        # we select the first batch item for plotting
        context_in = summary["context_in"][0, :, 0].numpy()
        context_out = summary["context_out"][0].numpy()
        target_in = summary["target_in"][0, :, 0].numpy()
        target_out = summary["target_out"][0].numpy()
        prediction_mu = summary["prediction_mu"][0].numpy()
        prediction_sigma = summary["prediction_sigma"][0].numpy()
        if "samples" in summary:
            samples = summary["samples"][:, 0, :, 0].numpy().T

        if validate:
            name = "val" + os.sep
        else:
            name = ""

        # plotly plot for Visdom
        if show and self.vlog is not None:

            fig = go.Figure()
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=self.config.plot_y_range)

            for c in range(context_out.shape[-1]):
                fig.add_trace(
                    go.Scatter(
                        x=target_in,
                        y=target_out[..., c],
                        mode="lines",
                        line=dict(color="blue", width=1, dash="dash"),
                        name="target[{}]".format(c),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=context_in,
                        y=context_out[..., c],
                        mode="markers",
                        marker=dict(color="blue", size=6),
                        name="context[{}]".format(c),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=target_in,
                        y=prediction_mu[..., c],
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="prediction[{}]".format(c),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=target_in,
                        y=prediction_mu[..., c] - prediction_sigma[..., c],
                        mode="lines",
                        line=dict(width=0, color="black"),
                        name="- 1 sigma[{}]".format(c),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=target_in,
                        y=prediction_mu[..., c] + prediction_sigma[..., c],
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(0,0,0,0.1)",
                        line=dict(width=0, color="black"),
                        name="+ 1 sigma[{}]".format(c),
                    )
                )
            # samples will only be shown for first channel!
            if "samples" in summary:
                for s in range(samples.shape[1]):
                    fig.add_trace(
                        go.Scatter(
                            x=target_in,
                            y=samples[:, s],
                            mode="lines",
                            line=dict(width=1, color="rgba(0,255,0,0.2)"),
                            showlegend=False,
                        )
                    )
                self.vlog.show_plotly_plt(fig, name=name + "samples")
            else:
                id_ = "prior" if validate else "posterior"
                self.vlog.show_plotly_plt(fig, name=name + id_)
            del fig

        # matplotlib plot for saving
        if save and self.elog is not None:

            epoch_str = "{:05d}".format(summary["epoch"])
            fig, ax = matplotlib.pyplot.subplots(1, 1)
            ax.plot(target_in, target_out, "b--", lw=1)
            ax.plot(context_in, context_out, "bo", ms=6)
            ax.plot(target_in, prediction_mu, color="black")
            ax.axis([*x_range, *self.config.plot_y_range])
            for c in range(context_out.shape[-1]):
                ax.fill_between(
                    target_in,
                    prediction_mu[..., c] - prediction_sigma[..., c],
                    prediction_mu[..., c] + prediction_sigma[..., c],
                    color="black",
                    alpha=0.2,
                )
            if "samples" in summary:
                ax.plot(target_in, samples, color="green", alpha=0.2)
                self.elog.show_matplot_plt(
                    fig, name=os.path.join(name, "samples", epoch_str)
                )
            else:
                id_ = "prior" if validate else "posterior"
                self.elog.show_matplot_plt(fig, name=os.path.join(name, id_, epoch_str))
            matplotlib.pyplot.close(fig)

    def validate(self, epoch):

        if (epoch + 1) % self.config.validate_every != 0:
            return

        self.model.eval()

        with torch.no_grad():

            batch = next(self.generator)
            batch["epoch"] = epoch  # logging

            context_in = batch["context_in"].to(self.config.device)
            context_out = batch["context_out"].to(self.config.device)
            target_in = batch["target_in"].to(self.config.device)
            target_out = batch["target_out"].to(self.config.device)

            prediction = self.model(
                context_in, context_out, target_in, target_out, store_rep=False
            )
            prediction = tensor_to_loc_scale(
                prediction,
                distributions.Normal,
                logvar_transform=self.config.output_transform_logvar,
                axis=2,
            )
            batch["prediction_mu"] = prediction.loc.detach().cpu()
            batch["prediction_sigma"] = prediction.scale.detach().cpu()

            loss_recon = -prediction.log_prob(target_out)
            loss_recon = loss_recon.mean(0).sum()  # batch mean
            loss_total = loss_recon
            batch["loss_recon"] = loss_recon.item()  # logging

            if hasattr(self.model, "prior"):
                loss_latent = distributions.kl_divergence(
                    self.model.posterior, self.model.prior
                )
                loss_latent = loss_latent.mean(0).sum()  # batch mean
                loss_total += loss_latent
                batch["loss_latent"] = loss_latent.item()
                batch["loss_total"] = loss_total.item()

        self.log(batch, "val")

        # create plot with different samples
        try:
            summary = self.make_samples()
            summary["epoch"] = epoch
            self.make_plots(summary, save=True, show=True, validate=True)
        except (
            ValueError,
            NotImplementedError,
            AttributeError,
        ) as e:  # if models can't sample
            pass
        except RuntimeError as e:  # numerical instability in GP cholesky
            self.print("Skipped sampling because of numerical instability.")
            pass
        except Exception as e:
            raise e

        # default configuration doesn't do anything here, but when
        # we use something like ReduceLROnPlateau, we need this call
        self.step_params(loss_total.item(), epoch, val=True)

    def make_samples(self):

        with torch.no_grad():

            if isinstance(self.generator, MultiThreadedAugmenter):
                generator = self.generator.generator
            else:
                generator = self.generator

            generator.batch_size = 1
            generator.num_target = 100
            generator.num_context = 10
            batch = next(generator)
            generator.batch_size = self.config.batch_size
            generator.num_target = self.config.generator_kwargs.num_target
            generator.num_context = self.config.generator_kwargs.num_context

            context_in = batch["context_in"].to(self.config.device)
            context_out = batch["context_out"].to(self.config.device)
            target_in = batch["target_in"].to(self.config.device)
            target_out = batch["target_out"].to(self.config.device)

            prediction = self.model(
                context_in, context_out, target_in, target_out, store_rep=True
            )
            prediction = tensor_to_loc_scale(
                prediction,
                distributions.Normal,
                logvar_transform=self.config.output_transform_logvar,
                axis=2,
            )

            samples = self.model.sample(target_in, self.config.num_samples)
            samples = tensor_to_loc_scale(
                samples, distributions.Normal, logvar_transform=True, axis=3
            ).loc

            batch["prediction_mu"] = prediction.loc.cpu()
            batch["prediction_sigma"] = prediction.scale.cpu()
            batch["samples"] = samples.cpu()

        return batch

    def step_params(self, loss, epoch, val):

        if epoch < self.config.lr_warmup:
            for group in self.optimizer.param_groups:
                lr = (
                    self.config.optimizer_kwargs.lr
                    * (epoch + 1)
                    / self.config.lr_warmup
                )
                group["lr"] = lr
            return

        if self.config.scheduler_step_train and val:
            pass
        elif self.config.scheduler_step_train:
            self.scheduler.step(epoch)
        elif val:
            self.scheduler.step(loss)
        else:
            pass

        for group in self.optimizer.param_groups:
            if group["lr"] < self.config.lr_min:
                self.print("Learning rate too small, stopping...")
                self.stop()

    def test(self):

        if self.config.test_single:
            self.test_single()

        if self.config.test_distribution:
            self.test_distribution()

        if self.config.test_diversity:
            self.test_diversity()

    def test_single(self):

        if isinstance(self.generator, MultiThreadedAugmenter):
            generator = self.generator.generator
        else:
            generator = self.generator

        # for this evaluation we want to separate target and context entirely
        generator.target_include_context = False
        generator.batch_size = self.config.test_batch_size
        generator.num_target = self.config.test_num_target_single
        generator.test = True
        self.model.eval()

        info = {}
        info["dims"] = ["instance", "test_num_context", "metric"]
        info["coords"] = {
            "test_num_context": self.config.test_num_context,
            "metric": [
                "predictive_likelihood",
                "predictive_error_squared",
                "reconstruction_likelihood",
                "reconstruction_error_squared",
            ],
        }

        scores = []

        # Catch possible RuntimeError from GP
        fail_counter = 0
        while len(scores) < self.config.test_batches_single and fail_counter < 100:

            try:

                scores_batch = []

                for num_context in self.config.test_num_context:
                    with torch.no_grad():

                        if num_context == "random":
                            generator.num_context = self.config.test_num_context_random
                        else:
                            generator.num_context = int(num_context)

                        batch = next(generator)
                        context_in = batch["context_in"].to(self.config.device)
                        context_out = batch["context_out"].to(self.config.device)
                        target_in = batch["target_in"].to(self.config.device)
                        target_out = batch["target_out"].to(self.config.device)

                        # PREDICTIVE PERFORMANCE
                        prediction = self.model(
                            context_in,
                            context_out,
                            target_in,
                            target_out,
                            store_rep=True,
                        )
                        prediction = tensor_to_loc_scale(
                            prediction,
                            distributions.Normal,
                            logvar_transform=self.config.output_transform_logvar,
                            axis=2,
                        )

                        predictive_error = torch.pow(prediction.loc - target_out, 2)
                        predictive_error = predictive_error.cpu().numpy()
                        predictive_error = np.nanmean(predictive_error, axis=(1, 2))
                        if isinstance(
                            self.model, (NeuralProcess, AttentiveNeuralProcess)
                        ):
                            predictive_ll = []
                            while len(predictive_ll) < self.config.test_latent_samples:
                                prediction = self.model.sample(target_in, 1)[0].cpu()
                                prediction = tensor_to_loc_scale(
                                    prediction,
                                    distributions.Normal,
                                    logvar_transform=self.config.output_transform_logvar,
                                    axis=2,
                                )
                                ll = prediction.log_prob(target_out.cpu()).numpy()
                                ll = np.nanmean(ll, axis=(1, 2))
                                predictive_ll.append(ll)
                            predictive_ll = np.nanmean(predictive_ll, axis=0)
                        elif self.model.use_gp:
                            predictive_ll = []
                            for i in range(20):
                                prediction = self.model.sample(
                                    target_in, self.config.test_latent_samples // 20
                                ).cpu()
                                prediction = tensor_to_loc_scale(
                                    prediction,
                                    distributions.Normal,
                                    logvar_transform=self.config.output_transform_logvar,
                                    axis=3,
                                )
                                ll = prediction.log_prob(
                                    target_out.cpu()
                                    .unsqueeze(0)
                                    .expand_as(prediction.loc)
                                ).numpy()
                                predictive_ll.append(np.nanmean(ll, axis=(2, 3)))
                            predictive_ll = np.concatenate(predictive_ll, 0)
                            predictive_ll = np.nanmean(predictive_ll, 0)
                        else:
                            predictive_ll = prediction.log_prob(target_out)
                            predictive_ll = predictive_ll.cpu().numpy()
                            predictive_ll = np.nanmean(predictive_ll, axis=(1, 2))

                        # RECONSTRUCTION PERFORMANCE
                        reconstruction = self.model(
                            context_in,
                            context_out,
                            context_in,
                            context_out,
                            store_rep=True,
                        )
                        reconstruction = tensor_to_loc_scale(
                            reconstruction,
                            distributions.Normal,
                            logvar_transform=self.config.output_transform_logvar,
                            axis=2,
                        )
                        reconstruction_error = torch.pow(
                            reconstruction.loc - context_out, 2
                        )
                        reconstruction_error = reconstruction_error.cpu().numpy()
                        reconstruction_error = np.nanmean(
                            reconstruction_error, axis=(1, 2)
                        )
                        if isinstance(
                            self.model, (NeuralProcess, AttentiveNeuralProcess)
                        ):
                            reconstruction_ll = []
                            while (
                                len(reconstruction_ll) < self.config.test_latent_samples
                            ):
                                reconstruction = self.model.sample(context_in, 1)[
                                    0
                                ].cpu()
                                reconstruction = tensor_to_loc_scale(
                                    reconstruction,
                                    distributions.Normal,
                                    logvar_transform=self.config.output_transform_logvar,
                                    axis=2,
                                )
                                ll = reconstruction.log_prob(context_out.cpu()).numpy()
                                ll = np.nanmean(ll, axis=(1, 2))
                                reconstruction_ll.append(ll)
                            reconstruction_ll = np.nanmean(reconstruction_ll, axis=0)
                        elif self.model.use_gp:
                            reconstruction_ll = []
                            for i in range(20):
                                reconstruction = self.model.sample(
                                    context_in, self.config.test_latent_samples // 20
                                ).cpu()
                                reconstruction = tensor_to_loc_scale(
                                    reconstruction,
                                    distributions.Normal,
                                    logvar_transform=self.config.output_transform_logvar,
                                    axis=3,
                                )
                                ll = reconstruction.log_prob(
                                    context_out.cpu()
                                    .unsqueeze(0)
                                    .expand_as(reconstruction.loc)
                                ).numpy()
                                reconstruction_ll.append(np.nanmean(ll, axis=(2, 3)))
                            reconstruction_ll = np.concatenate(reconstruction_ll, 0)
                            reconstruction_ll = np.nanmean(reconstruction_ll, 0)
                        else:
                            reconstruction_ll = reconstruction.log_prob(context_out)
                            reconstruction_ll = reconstruction_ll.cpu().numpy()
                            reconstruction_ll = np.nanmean(
                                reconstruction_ll, axis=(1, 2)
                            )

                        score = np.stack(
                            [
                                predictive_ll,
                                predictive_error,
                                reconstruction_ll,
                                reconstruction_error,
                            ],
                            axis=1,
                        )[:, None, :]
                        scores_batch.append(score)

                scores_batch = np.concatenate(scores_batch, 1)
                scores.append(scores_batch)

            except RuntimeError as re:

                fail_counter += 1
                if fail_counter > 100:
                    raise re
                else:
                    continue

        scores = np.concatenate(scores, 0)
        self.elog.save_numpy_data(scores, "test_single.npy")
        self.elog.save_dict(info, "test_single.json")

    def test_distribution(self):

        if isinstance(self.generator, MultiThreadedAugmenter):
            generator = self.generator.generator
        else:
            generator = self.generator

        # For this evaluation we want to always have the same target x values,
        # so we need to include the context and use linspace=True.
        # Wasserstein calculation is pretty expensive, so
        # we use fewer samples than in test_single. Samples in a batch
        # have the same x values for the context, which is fine for large
        # numbers of test cases, but here we reduce correlation by working
        # with batch_size=1 (except for Lotka-Volterra, because that generator
        # is VERY slow...)
        generator.target_include_context = True
        generator.target_fixed_size = True
        generator.linspace = True
        generator.num_target = self.config.test_num_target_distribution
        if not isinstance(generator, LotkaVolterraGenerator):
            generator.batch_size = 1
        else:
            generator.batch_size = 32
        generator.test = True
        self.model.eval()

        info = {}
        info["dims"] = ["test_num_context", "metric"]
        info["coords"] = {
            "test_num_context": self.config.test_num_context,
            "metric": [
                "wasserstein",
            ],
        }

        scores = []

        # test batches are now the inner loop, because the metrics are
        # calculated over all instances
        for num_context in self.config.test_num_context:

            gt_samples = []
            predictions = []

            if num_context == "random":
                generator.num_context = self.config.test_num_context_random
            else:
                generator.num_context = int(num_context)

            # fill the gt_samples and predictions lists, so we can calculate
            # metrics after.
            with torch.no_grad():
                for b in range(
                    self.config.test_batches_distribution
                    * self.config.test_batch_size
                    // generator.batch_size
                ):

                    batch = next(generator)
                    context_in = batch["context_in"].to(self.config.device)
                    context_out = batch["context_out"].to(self.config.device)
                    target_in = batch["target_in"].to(self.config.device)
                    target_out = batch["target_out"].to(self.config.device)

                    prediction = self.model(
                        context_in, context_out, target_in, target_out, store_rep=False
                    )
                    prediction = tensor_to_loc_scale(
                        prediction,
                        distributions.Normal,
                        logvar_transform=self.config.output_transform_logvar,
                        axis=2,
                    ).loc

                    gt_samples.append(target_out.cpu().numpy()[:, :, 0])
                    predictions.append(prediction.cpu().numpy()[:, :, 0])

            gt_samples = np.concatenate(gt_samples, 0)
            predictions = np.concatenate(predictions, 0)

            # we should now have arrays of shape (N*B, test_num_target_distribution)
            sqdist = ot.dist(gt_samples, predictions, "euclidean")
            wdist, _ = ot.emd2(
                np.ones((sqdist.shape[0],)) / sqdist.shape[0],
                np.ones((sqdist.shape[1],)) / sqdist.shape[1],
                sqdist,
                numItermax=1e7,
                log=True,
                return_matrix=True,
            )
            wdist = np.sqrt(wdist)

            scores.append(
                np.array(
                    [
                        wdist,
                    ]
                )
            )

        scores = np.stack(scores)
        self.elog.save_numpy_data(scores, "test_distribution.npy")
        self.elog.save_dict(info, "test_distribution.json")

    def test_diversity(self):

        if isinstance(self.generator, MultiThreadedAugmenter):
            generator = self.generator.generator
        else:
            generator = self.generator

        # Can't sample from regular ConvCNP
        if isinstance(self.model, ConvCNP) and not self.model.use_gp:
            return
        # Can't construct oracle for non-GP generators
        if not isinstance(generator, GaussianProcessGenerator):
            return

        generator.target_include_context = False
        generator.num_target = self.config.test_num_target_diversity
        generator.batch_size = self.config.test_batch_size
        generator.test = True
        self.model.eval()

        info = {}
        info["dims"] = ["instance", "test_num_context", "metric"]
        info["coords"] = {
            "test_num_context": self.config.test_num_context,
            "metric": ["wasserstein"],
        }

        scores = []

        for b in range(self.config.test_batches_diversity):

            scores_batch = []

            for num_context in self.config.test_num_context:
                with torch.no_grad():

                    if num_context == "random":
                        generator.num_context = self.config.test_num_context_random
                    else:
                        generator.num_context = int(num_context)

                    # GP can occasionally because of numerical instability,
                    # so we need a small loop that accounts for this
                    num_failures = 0
                    while num_failures < 1000:

                        batch = next(generator)
                        context_in = batch["context_in"].to(self.config.device)
                        context_out = batch["context_out"].to(self.config.device)
                        target_in = batch["target_in"].to(self.config.device)
                        target_out = batch["target_out"].to(self.config.device)

                        prediction = self.model(
                            context_in,
                            context_out,
                            target_in,
                            target_out,
                            store_rep=True,
                        )

                        try:

                            samples = []
                            while len(samples) < self.config.test_latent_samples:
                                # we draw samples separately, otherwise GPU
                                # memory could become an issue
                                sample = self.model.sample(target_in, 1)[0].cpu()
                                sample = tensor_to_loc_scale(
                                    sample,
                                    distributions.Normal,
                                    logvar_transform=self.config.output_transform_logvar,
                                    axis=2,
                                ).loc
                                samples.append(sample[..., 0].numpy())
                            samples = np.stack(samples)  # (num_samples, B, M)

                            # get samples from oracle GP
                            gt_samples = []
                            for b in range(context_in.shape[0]):
                                x_train = context_in[b].cpu().numpy()  # (N, 1)
                                y_train = context_out[b].cpu().numpy()  # (N, 1)
                                x = target_in[b].cpu().numpy()  # (M, 1)
                                K = generator.kernel(x_train, x_train)
                                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(x_train)))
                                K_s = generator.kernel(x_train, x)
                                L_k = np.linalg.solve(L, K_s)
                                mu = np.dot(L_k.T, np.linalg.solve(L, y_train)).reshape(
                                    -1
                                )
                                K_ss = generator.kernel(x, x)
                                K_ss -= np.dot(L_k.T, L_k)
                                L_ss = np.linalg.cholesky(K_ss + 1e-6 * np.eye(len(x)))
                                samp = np.random.normal(
                                    size=(
                                        L_ss.shape[1],
                                        self.config.test_latent_samples,
                                    )
                                )
                                samp = np.dot(L_ss, samp)
                                samp = samp + mu.reshape(-1, 1)  # (M, num_samples)
                                gt_samples.append(samp)
                            gt_samples = np.stack(gt_samples).transpose(2, 0, 1)

                            break

                        except RuntimeError as e:

                            num_failures += 1
                            continue

                    wdist_scores = []
                    for i in range(samples.shape[1]):
                        sqdist = ot.dist(samples[:, i], gt_samples[:, i], "euclidean")
                        wdist, log = ot.emd2(
                            np.ones((sqdist.shape[0],)) / sqdist.shape[0],
                            np.ones((sqdist.shape[1],)) / sqdist.shape[1],
                            sqdist,
                            numItermax=1e7,
                            log=True,
                            return_matrix=True,
                        )
                        wdist = np.sqrt(wdist)
                        wdist_scores.append(wdist)
                    wdist_scores = np.array(wdist_scores).reshape(-1, 1, 1)
                    scores_batch.append(wdist_scores)

            scores_batch = np.concatenate(scores_batch, 1)
            scores.append(scores_batch)

        scores = np.concatenate(scores, 0)
        self.elog.save_numpy_data(scores, "test_diversity.npy")
        self.elog.save_dict(info, "test_diversity.json")


if __name__ == "__main__":

    parser = get_default_experiment_parser()
    parser.add_argument("-rep", "--representation_channels", type=int, default=128)
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults(args.representation_channels)

    run_experiment(
        NeuralProcessExperiment,
        DEFAULTS,
        args,
        mods=MODS,
        globs=globals(),
        remove_mod_none=args.remove_none,
    )
