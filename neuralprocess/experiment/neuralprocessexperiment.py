import matplotlib
matplotlib.use("agg")

import numpy as np
import os
import time
import plotly.graph_objs as go
import gpytorch

import torch
from torch import nn, optim, distributions
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False
if "CUDNN_DETERMINISTIC" in os.environ:
    if os.environ["CUDNN_DETERMINISTIC"] not in (0, False, "false", "FALSE", "False"):
        cudnn.benchmark = False
        cudnn.deterministic = True

from trixi.util import Config, ResultLogDict
from trixi.experiment import PytorchExperiment

from neuralprocess.util import (
    get_default_experiment_parser,
    run_experiment,
    set_seeds,
    tensor_to_loc_scale
)
from neuralprocess.data import (
    GaussianProcessGenerator,
    WeaklyPeriodicKernel,
    StepFunctionGenerator
)
from neuralprocess.data.gp import (
    GaussianKernel,
    WeaklyPeriodicKernel,
    Matern52Kernel
)
from neuralprocess.model import (
    NeuralProcess,
    AttentiveNeuralProcess,
    ConvCNP,
    ConvDeepSet,
    GPConvDeepSet,
    generic
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
            num_target=(3, 100),
            target_larger_than_context=True,
            target_includes_context=True,
            output_noise=0.,
            linspace=False,
            number_of_threads_in_multithreaded=1
        ),

        # Model
        model=NeuralProcess,
        model_kwargs=dict(distribution=distributions.Normal),
        modules=dict(
            prior_encoder=generic.MLP,
            decoder=generic.MLP
        ),
        modules_kwargs=dict(
            prior_encoder=dict(
                in_channels=2,
                out_channels=2*representation_channels,
                hidden_channels=128,
                hidden_layers=6
            ),
            decoder=dict(
                in_channels=representation_channels+1,
                out_channels=2,
                hidden_channels=128,
                hidden_layers=6
            )
        ),
        custom_init=False,

        # Optimization
        optimizer=optim.Adam,
        optimizer_kwargs=dict(lr=1e-3),
        lr_min=1e-6,
        scheduler=optim.lr_scheduler.StepLR,
        scheduler_kwargs=dict(step_size=1000, gamma=0.995),
        scheduler_step_train=True,
        clip_loss=1e3,
        clip_grad=1e3,
        lr_warmup=0,

        # Logging
        backup_every=1000,
        validate_every=1000,
        show_every=100,
        num_samples=50,
        plot_y_range=[-3, 3],

        # Testing
        test_batches_single=1000,
        test_batches_distribution=30,
        test_batches_diversity=100,
        test_batch_size=1024,
        # test_num_context=[5, 10, 15, 20],
        test_num_context=["random", ],
        test_num_context_random=[5, 50],
        test_num_target_single=100,
        test_num_target_distribution=50,
        test_num_target_diversity=50,
        test_latent_samples=100,
        test_single=True,
        test_distribution=True,
        test_diversity=True,
        test_mmd_alphas=[0.05, 0.1, 0.5, 1., 5.]

    )

    ATTENTION = Config(  # you also need to set DETERMINISTICENCODER for this
        model=AttentiveNeuralProcess,
        model_kwargs=dict(
            project_to=128,  # embed_dim in attention mechanism
            project_bias=True,
            in_channels=1,
            representation_channels=representation_channels,
        ),
        modules=dict(
            attention=nn.MultiheadAttention
        ),
        modules_kwargs=dict(
            attention=dict(
                embed_dim=128,
                num_heads=8
            )
        ),
    )

    CONVCNP = Config(
        model=ConvCNP,
        model_kwargs=dict(
            points_per_unit=20,
            range_padding=0.1,
            grid_divisible_by=64
        ),
        modules=dict(
            input_interpolation=ConvDeepSet,
            convnet=generic.SimpleUNet,
            output_interpolation=ConvDeepSet
        ),
        modules_kwargs=dict(
            input_interpolation=dict(
                kernel=gpytorch.kernels.RBFKernel,
                kernel_kwargs=dict(),
                use_density=True,
                use_density_norm=True,
                project_to=8,
                project_bias=True,
                project_in_channels=2,  # because use_density=True
            ),
            convnet=dict(
                in_channels=8,
                out_channels=8,
                num_blocks=6,
                input_bypass=True,
                encoding_block_type=generic.ConvNormActivationPool,
                encoding_block_kwargs=dict(
                    conv_op=nn.Conv1d,
                    conv_kwargs=dict(
                        kernel_size=5,
                        stride=2,
                        padding=2
                    ),
                    activation_op=nn.ReLU,
                    activation_kwargs=dict(
                        inplace=True
                    )
                ),
                decoding_block_type=generic.UpsampleConvNormActivation,
                decoding_block_kwargs=dict(
                    conv_op=nn.ConvTranspose1d,
                    conv_kwargs=dict(
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1
                    ),
                    activation_op=nn.ReLU,
                    activation_kwargs=dict(
                        inplace=True
                    )
                )
            ),
            output_interpolation=dict(
                kernel=gpytorch.kernels.RBFKernel,
                kernel_kwargs=dict(),
                use_density=False,
                use_density_norm=False,
                project_to=2,
                project_bias=True,
                project_in_channels=16  # because input_bypass=True in convnet
            )
        ),
        custom_init=True
    )

    GPCONVCNP = Config(  # also set CONVCNP!
        modules=dict(input_interpolation=GPConvDeepSet),
        modules_kwargs=dict(
            input_interpolation=dict(
                gp_lambda=0.2,
                gp_sample_from_posterior=0,
            )
        )
    )

    MATERNKERNEL = Config(
        generator_kwargs=dict(kernel_type=Matern52Kernel)
    )

    WEAKLYPERIODICKERNEL = Config(
        generator_kwargs=dict(kernel_type=WeaklyPeriodicKernel)
    )

    STEP = Config(
        generator=StepFunctionGenerator,
        generator_kwargs=dict(
            y_range=[-3, 3],
            number_of_steps=[3, 10],
            min_step_width=0.1,
            min_step_height=0.1,
        )
    )

    DETERMINISTICENCODER = Config(
        modules=dict(deterministic_encoder=generic.MLP),
        modules_kwargs=dict(
            deterministic_encoder=dict(
                in_channels=2,
                out_channels=representation_channels,
                hidden_channels=128,
                hidden_layers=6
            ),
            decoder=dict(
                in_channels=2*representation_channels+1
            )
        )
    )

    LONG = Config(
        n_epochs=1200000,
        scheduler_kwargs=dict(step_size=2000)
    )

    MODS = {
        "ATTENTION": ATTENTION,
        "CONVCNP": CONVCNP,
        "GPCONVCNP": GPCONVCNP,
        "MATERNKERNEL": MATERNKERNEL,
        "WEAKLYPERIODICKERNEL": WEAKLYPERIODICKERNEL,
        "STEP": STEP,
        "DETERMINISTICENCODER": DETERMINISTICENCODER,
        "LONG": LONG  
    }

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

        if self.config.custom_init:
            def init_weights(m):
                if type(m) == nn.Linear:
                    nn.init.xavier_normal_(m.weight, 0.2)
                    if hasattr(m, "bias"):
                        nn.init.constant_(m.bias, 0)
                elif type(m) in (nn.Conv1d, nn.ConvTranspose1d):
                    nn.init.xavier_normal_(m.weight, 1.)
                    if hasattr(m, "bias"):
                        nn.init.constant_(m.bias, 1e-3)
                else:
                    pass
            self.model.apply(init_weights)

        self.clog.show_text(repr(self.model), "Model")

    def setup_optimization(self):

        self.optimizer = self.config.optimizer(self.model.parameters(),
                                               **self.config.optimizer_kwargs)
        self.scheduler = self.config.scheduler(self.optimizer,
                                               **self.config.scheduler_kwargs)

        if self.config.lr_warmup > 0:
            for group in self.optimizer.param_groups:
                group["lr"] = 0.

    def setup_data(self):

        # We can also wrap the generator in a MultithreadedAugmenter,
        # but for now it's fine because the process is cheap
        self.generator = self.config.generator(
            self.config.batch_size, **self.config.generator_kwargs)

    def _setup_internal(self):

        super()._setup_internal()

        # manually set up a ResultLogDict with running mean ability
        self.results.close()
        self.results = ResultLogDict("results-log.json",
                                     base_dir=self.elog.result_dir,
                                     mode="w",
                                     running_mean_length=self.config.show_every)

        # save modifications we made to config
        self.elog.save_config(self.config, "config")

    def prepare(self):

        for name, model in self.get_pytorch_modules().items():
            model.to(self.config.device)

    def train(self, epoch):

        if epoch == 0:
            self.data_times = []
            self.pass_times = []

        t0 = time.time()

        self.model.train()
        self.optimizer.zero_grad()

        batch = next(self.generator)
        batch["epoch"] = epoch  # logging

        t1 = time.time()

        # with torch.autograd.detect_anomaly():

        context_in = batch["context_in"].to(self.config.device)
        context_out = batch["context_out"].to(self.config.device)
        target_in = batch["target_in"].to(self.config.device)
        target_out = batch["target_out"].to(self.config.device)

        with torch.autograd.detect_anomaly():

            try:

                prediction = self.model(context_in,
                                        context_out,
                                        target_in,
                                        target_out,
                                        store_rep=False)
                prediction = tensor_to_loc_scale(prediction,
                                                distributions.Normal,
                                                logvar_transform=True,
                                                axis=2)
                batch["prediction_mu"] = prediction.loc.detach().cpu()
                batch["prediction_sigma"] = prediction.scale.detach().cpu()

                loss_recon = -prediction.log_prob(target_out)
                loss_recon = loss_recon.mean(1).sum()  # batch mean
                loss_total = loss_recon
                if hasattr(self.model, "prior"):
                    loss_latent = distributions.kl_divergence(self.model.posterior,
                                                            self.model.prior)
                    loss_latent = loss_latent.mean(0).sum()  # batch mean
                    loss_total += loss_latent

                loss_total.backward()

            except Exception as e:

                import IPython
                IPython.embed()

        if self.config.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.clip_grad)
        self.optimizer.step()

        t2 = time.time()

        self.data_times.append(t1-t0)
        self.pass_times.append(t2-t1)
        if (epoch + 1) % 1000 == 0:
            print(np.mean(self.data_times))
            print(np.mean(self.pass_times))
            self.data_times = []
            self.pass_times = []

        batch["loss_recon"] = loss_recon.item()  # logging
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

        for l in ("loss_recon", "loss_latent", "loss_total"):
            if l in summary:
                name = l
                if validate:
                    name += "_val"
                self.add_result(summary[l],
                                name,
                                summary["epoch"],
                                "Loss",
                                plot_result=show,
                                plot_running_mean=not validate)

        self.make_plots(summary, save=backup, show=show, validate=validate)

    def make_plots(self, summary, save=False, show=True, validate=False):

        if not save and not show:
            return

        # we select the first batch item for plotting
        context_in = summary["context_in"][:, 0, 0].numpy()
        context_out = summary["context_out"][:, 0, 0].numpy()
        target_in = summary["target_in"][:, 0, 0].numpy()
        target_out = summary["target_out"][:, 0, 0].numpy()
        prediction_mu = summary["prediction_mu"][:, 0, 0].numpy()
        prediction_sigma = summary["prediction_sigma"][:, 0, 0].numpy()
        if "samples" in summary:
            samples = summary["samples"][:, :, 0, 0].numpy().T

        if validate:
            name = "val" + os.sep
        else:
            name = ""

        # plotly plot for Visdom
        if show and self.vlog is not None:

            fig = go.Figure()
            fig.update_xaxes(range=self.generator.x_range)
            fig.update_yaxes(range=self.config.plot_y_range)
            fig.add_trace(go.Scatter(x=target_in,
                                     y=target_out,
                                     mode="lines",
                                     line=dict(color="blue", width=1, dash="dash"),
                                     name="target"))
            fig.add_trace(go.Scatter(x=context_in,
                                     y=context_out,
                                     mode="markers",
                                     marker=dict(color="blue", size=6),
                                     name="context"))
            fig.add_trace(go.Scatter(x=target_in,
                                     y=prediction_mu,
                                     mode="lines",
                                     line=dict(color="black", width=1),
                                     name="prediction"))
            fig.add_trace(go.Scatter(x=target_in,
                                     y=prediction_mu - prediction_sigma,
                                     mode="lines",
                                     line=dict(width=0, color="black"),
                                     name="- 1 sigma"))
            fig.add_trace(go.Scatter(x=target_in,
                                     y=prediction_mu + prediction_sigma,
                                     mode="lines",
                                     fill="tonexty",
                                     fillcolor="rgba(0,0,0,0.1)",
                                     line=dict(width=0, color="black"),
                                     name="+ 1 sigma"))
            if "samples" in summary:
                for s in range(samples.shape[1]):
                    fig.add_trace(go.Scatter(x=target_in,
                                             y=samples[:, s],
                                             mode="lines",
                                             line=dict(width=1, color="rgba(0,255,0,0.2)"),
                                             showlegend=False))
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
            ax.axis([*self.generator.x_range, *self.config.plot_y_range])
            ax.fill_between(target_in,
                            prediction_mu - prediction_sigma,
                            prediction_mu + prediction_sigma,
                            color="black",
                            alpha=0.2)
            if "samples" in summary:
                ax.plot(target_in, samples, color="green", alpha=0.2)
                self.elog.show_matplot_plt(
                    fig, name=os.path.join(name, "samples", epoch_str))
            else:
                id_ = "prior" if validate else "posterior"
                self.elog.show_matplot_plt(
                    fig, name=os.path.join(name, id_, epoch_str))
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

            prediction = self.model(context_in,
                                    context_out,
                                    target_in,
                                    target_out,
                                    store_rep=False)
            prediction = tensor_to_loc_scale(prediction,
                                             distributions.Normal,
                                             logvar_transform=True,
                                             axis=2)
            batch["prediction_mu"] = prediction.loc.detach().cpu()
            batch["prediction_sigma"] = prediction.scale.detach().cpu()

            loss_recon = -prediction.log_prob(target_out)
            loss_recon = loss_recon.mean(1).sum()  # batch mean
            loss_total = loss_recon
            batch["loss_recon"] = loss_recon.item()  # logging

            if hasattr(self.model, "prior"):
                loss_latent = distributions.kl_divergence(
                    self.model.posterior, self.model.prior)
                loss_latent = loss_latent.mean(0).sum()  # batch mean
                loss_total += loss_latent
                batch["loss_latent"] = loss_latent.item()
                batch["loss_total"] = loss_total.item()

        self.log(batch, "val")

        try:
            summary = self.make_samples()
            summary["epoch"] = epoch
            self.make_plots(summary, save=True, show=True, validate=True)
        except (ValueError, NotImplementedError) as e:  # if models can't sample
            pass
        except Exception as e:
            raise e

        self.step_params(loss_total.item(), epoch, val=True)

    def make_samples(self):

        with torch.no_grad():

            self.generator.batch_size = 1
            self.generator.num_target = 100
            self.generator.num_context = 10
            batch = next(self.generator)
            self.generator.batch_size = self.config.batch_size
            self.generator.num_target = self.config.generator_kwargs.num_target
            self.generator.num_context = self.config.generator_kwargs.num_context

            context_in = batch["context_in"].to(self.config.device)
            context_out = batch["context_out"].to(self.config.device)
            target_in = batch["target_in"].to(self.config.device)
            target_out = batch["target_out"].to(self.config.device)

            prediction = self.model(context_in,
                                    context_out,
                                    target_in,
                                    target_out,
                                    store_rep=True)
            prediction = tensor_to_loc_scale(prediction,
                                             distributions.Normal,
                                             logvar_transform=True,
                                             axis=2)
                    
            samples = self.model.sample(target_in, self.config.num_samples)
            samples = tensor_to_loc_scale(samples,
                                          distributions.Normal,
                                          logvar_transform=True,
                                          axis=3).loc

            batch["prediction_mu"] = prediction.loc.cpu()
            batch["prediction_sigma"] = prediction.scale.cpu()
            batch["samples"] = samples.cpu()

        return batch       

    def step_params(self, loss, epoch, val):

        if epoch < self.config.lr_warmup:
            for group in self.optimizer.param_groups:
                lr = self.config.optimizer_kwargs.lr * (epoch+1) / self.config.lr_warmup
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
                print("Learning rate too small, stopping...")
                self.stop()

    def test(self):

        if self.config.test_single:
            self.test_single()

        if self.config.test_distribution:
            self.test_distribution()

        if self.config.test_diversity:
            self.test_diversity()

    def test_single(self):

        # for the evaluation we want to separate target and context entirely
        target_include_context = self.config.target_include_context
        self.config.target_include_context = False
        self.model.eval()

        info = {}
        info["dims"] = ["instance", "test_num_context", "metric"]
        info["coords"] = {
            "test_num_context": self.config.test_num_context,
            "metric": [
                "predictive_likelihood",
                "predictive_error_squared",
                "predictive_likelihood_sample",
                "predictive_error_squared_sample",
                "reconstruction_likelihood",
                "reconstruction_error_squared",
                "reconstruction_likelihood_sample",
                "reconstruction_error_squared_sample",
                "kl",
                "kernel_weighted_difference",
                "likelihood_under_gp_prior"
            ]
        }

        scores = []

        for b in range(self.config.test_batches_single):

            print("Single test batch", b)
            t0 = time.time()

            scores_batch = []

            for num_context in self.config.test_num_context:
                with torch.no_grad():

                    if num_context == "random":
                        num_context = np.random.randint(*self.config.test_num_context_random)

                    context_in, context_out, target_in, target_out = self.make_batch(self.config.test_batch_size,
                                                                                self.config.test_x_range,
                                                                                num_context,
                                                                                self.config.test_num_target_single,
                                                                                linspace=True)

                    if hasattr(self.model, "encode_posterior"):
                        self.model.encode_posterior(context_in, context_out, target_in, target_out)
                    prediction = self.model(context_in, context_out, target_in, store_rep=True)

                    # sample predictive performance for GPConvCNP
                    if isinstance(self.model, convcnp.ConvCNP):
                        # if isinstance(self.model.l0, convcnp.GPConvDeepSet):
                        #     sample = self.model.sample(target_in, 1)
                        #     predictive_likelihood_sample = self.model.distribution(*sample).log_prob(target_out.unsqueeze(0))[0].cpu().numpy()
                        #     predictive_likelihood_sample = np.nanmean(predictive_likelihood_sample,
                        #                                               axis=tuple(range(1, predictive_likelihood_sample.ndim)))
                        #     predictive_error_squared_sample = (target_out.cpu().numpy() - sample[0][0].cpu().numpy())**2
                        #     predictive_error_squared_sample = np.nanmean(predictive_error_squared_sample,
                        #                                                 axis=tuple(range(1, predictive_error_squared_sample.ndim)))
                        # else:
                        #     predictive_likelihood_sample = np.zeros((context_in.shape[0], )) * np.nan
                        #     predictive_error_squared_sample = np.zeros((context_in.shape[0], )) * np.nan
                        predictive_likelihood_sample = np.zeros((context_in.shape[0], )) * np.nan
                        predictive_error_squared_sample = np.zeros((context_in.shape[0], )) * np.nan

                    # calculate the predictive likelihood now
                    # for models with a latent space we will overwrite this later
                    if not self.config.loc_scale_prediction:
                        prediction = (prediction, torch.ones_like(prediction) * np.sqrt(0.5))
                    predictive_likelihood = np.nanmean(self.model.distribution(*prediction).log_prob(target_out).cpu().numpy(),
                                                        axis=tuple(range(1, target_out.ndim)))
                    prediction = prediction[0]
                    prediction = prediction.cpu().numpy()
                    predictive_error_squared = (target_out.cpu().numpy() - prediction)**2
                    predictive_error_squared = np.nanmean(predictive_error_squared, axis=tuple(range(1, predictive_error_squared.ndim)))

                    if hasattr(self.model, "posterior"):
                        kl = distributions.kl_divergence(self.model.posterior, self.model.prior).cpu().numpy()
                        while kl.ndim > 1:
                            kl = kl.sum(-1)
                    else:
                        kl = np.zeros_like(predictive_error_squared) * np.nan

                    self.process.fit(target_in[0].cpu().numpy(), None)
                    likelihood_under_gp_prior = []
                    for i in range(prediction.shape[0]):
                        likelihood_under_gp_prior.append(self.process.log_likelihood(prediction[i]))
                    likelihood_under_gp_prior = np.array(likelihood_under_gp_prior)

                    kwd = kernel_weighted_difference(prediction[:, :, 0], self.process.K, norm=2)

                    # now draw multiple samples for likelihoods if we can
                    if hasattr(self.model, "prior") and self.model.prior is not None:
                        predictive_likelihood = []
                        while len(predictive_likelihood) < self.config.test_latent_samples:
                            sample = self.model.prior.sample()
                            prediction = self.model.reconstruct(target_in, sample)
                            if not self.config.loc_scale_prediction:
                                prediction = (prediction, torch.ones_like(prediction) * np.sqrt(0.5))
                            predictive_likelihood.append(torch.mean(self.model.distribution(*prediction).log_prob(target_out), (1, 2)))
                            predictive_error_squared_sample = (target_out.cpu().numpy() - prediction[0].cpu().numpy())**2  # automatically keeps the last one
                            predictive_error_squared_sample = np.nanmean(predictive_error_squared_sample,
                                                                         axis=tuple(range(1, predictive_error_squared_sample.ndim)))
                        predictive_likelihood_sample = predictive_likelihood[-1].cpu().numpy()
                        predictive_likelihood = torch.stack(predictive_likelihood).mean(0).cpu().numpy()
                        
                    # on to the reconstruction performance
                    prediction = self.model(context_in, context_out, context_in, store_rep=True)

                    # sample reconstruction performance for GPConvCNP
                    if isinstance(self.model, convcnp.ConvCNP):
                        # if isinstance(self.model.l0, convcnp.GPConvDeepSet):
                        #     sample = self.model.sample(context_in, 1)
                        #     reconstruction_likelihood_sample = self.model.distribution(*sample).log_prob(context_out.unsqueeze(0))[0].cpu().numpy()
                        #     reconstruction_likelihood_sample = np.nanmean(reconstruction_likelihood_sample,
                        #                                                 axis=tuple(range(1, reconstruction_likelihood_sample.ndim)))
                        #     reconstruction_error_squared_sample = (context_out.cpu().numpy() - sample[0][0].cpu().numpy())**2
                        #     reconstruction_error_squared_sample = np.nanmean(reconstruction_error_squared_sample,
                        #                                                     axis=tuple(range(1, reconstruction_error_squared_sample.ndim)))
                        # else:
                        #     reconstruction_likelihood_sample = np.zeros((context_in.shape[0], )) * np.nan
                        #     reconstruction_error_squared_sample = np.zeros((context_in.shape[0], )) * np.nan
                        reconstruction_likelihood_sample = np.zeros((context_in.shape[0], )) * np.nan
                        reconstruction_error_squared_sample = np.zeros((context_in.shape[0], )) * np.nan

                    if not self.config.loc_scale_prediction:
                        prediction = (prediction, torch.ones_like(prediction) * np.sqrt(0.5))
                    reconstruction_likelihood = np.nanmean(self.model.distribution(*prediction).log_prob(context_out).cpu().numpy(),
                                                           axis=tuple(range(1, target_out.ndim)))
                    prediction = prediction[0]
                    reconstruction_error_squared = (context_out.cpu().numpy() - prediction.cpu().numpy())**2
                    reconstruction_error_squared = np.nanmean(reconstruction_error_squared, axis=tuple(range(1, reconstruction_error_squared.ndim)))

                    # again overwrite likelihood for models with a latent space
                    if hasattr(self.model, "prior") and self.model.prior is not None:
                        reconstruction_likelihood = []
                        while len(reconstruction_likelihood) < self.config.test_latent_samples:
                            sample = self.model.prior.sample()
                            prediction = self.model.reconstruct(context_in, sample)
                            if not self.config.loc_scale_prediction:
                                prediction = (prediction, torch.ones_like(prediction) * np.sqrt(0.5))
                            reconstruction_likelihood.append(torch.mean(self.model.distribution(*prediction).log_prob(context_out), (1, 2)))
                            reconstruction_error_squared_sample = (context_out.cpu().numpy() - prediction[0].cpu().numpy())**2  # automatically keeps the last one
                            reconstruction_error_squared_sample = np.nanmean(reconstruction_error_squared_sample,
                                                                             axis=tuple(range(1, reconstruction_error_squared_sample.ndim)))
                        reconstruction_likelihood_sample = reconstruction_likelihood[-1].cpu().numpy()
                        reconstruction_likelihood = torch.stack(reconstruction_likelihood).mean(0).cpu().numpy()

                    score = np.stack([predictive_likelihood,
                                      predictive_error_squared,
                                      predictive_likelihood_sample,
                                      predictive_error_squared_sample,
                                      reconstruction_likelihood,
                                      reconstruction_error_squared,
                                      reconstruction_likelihood_sample,
                                      reconstruction_error_squared_sample,
                                      kl,
                                      kwd,
                                      likelihood_under_gp_prior], axis=1)[:, None, :]
                    scores_batch.append(score)

            scores_batch = np.concatenate(scores_batch, 1)
            scores.append(scores_batch)

            print(time.time() - t0)

        scores = np.concatenate(scores, 0)
        self.elog.save_numpy_data(scores, "test_single.npy")
        self.elog.save_dict(info, "test_single.json")

        self.config.target_include_context = target_include_context

    def test_distribution(self):
        import ot

        # for the evaluation we want to always have the same x, so we need to include the context
        target_include_context = self.config.target_include_context
        self.config.target_include_context = True
        self.model.eval()

        info = {}
        info["dims"] = ["test_num_context", "metric"]
        info["coords"] = {
            "test_num_context": self.config.test_num_context,
            "metric": [
                "mmd",
                "mmd_sample",
                "wasserstein",
                "wasserstein_sample"
            ]
        }

        scores = []

        for num_context in self.config.test_num_context:

            print(num_context)

            gp_samples = []
            predictions = []
            samples = []

            if num_context == "random":

                with torch.no_grad():
                    for b in range(self.config.test_batches_distribution*self.config.test_batch_size):

                        nc = np.random.randint(*self.config.test_num_context_random)

                        context_in, context_out, target_in, target_out = self.make_batch(1,
                                                                                    self.config.test_x_range,
                                                                                    nc,
                                                                                    self.config.test_num_target_distribution - nc,
                                                                                    linspace=True)

                        prediction = self.model(context_in, context_out, target_in, store_rep=True)
                        if self.config.loc_scale_prediction:
                            prediction = prediction[0]

                        if isinstance(self.model, convcnp.ConvCNP):
                            # if isinstance(self.model.l0, convcnp.GPConvDeepSet):
                            #     sample = self.model.sample(target_in, 1)[0][0]
                            # else:
                            #     sample = None
                            sample = None
                        elif hasattr(self.model, "prior") and self.model.prior is not None:
                            sample = self.model.prior.sample()
                            sample = self.model.reconstruct(target_in, sample)
                            if self.config.loc_scale_prediction:
                                sample = sample[0]
                        else:
                            sample = None
                        
                        gp_samples.append(target_out.cpu().numpy()[:, :, 0])
                        predictions.append(prediction.cpu().numpy()[:, :, 0])
                        if sample is not None:
                            samples.append(sample.cpu().numpy()[:, :, 0])

            else:

                with torch.no_grad():
                    for b in range(self.config.test_batches_distribution):

                        context_in, context_out, target_in, target_out = self.make_batch(self.config.test_batch_size,
                                                                                    self.config.test_x_range,
                                                                                    num_context,
                                                                                    self.config.test_num_target_distribution - num_context,
                                                                                    linspace=True)

                        prediction = self.model(context_in, context_out, target_in, store_rep=True)
                        if self.config.loc_scale_prediction:
                            prediction = prediction[0]

                        if isinstance(self.model, convcnp.ConvCNP):
                            # if isinstance(self.model.l0, convcnp.GPConvDeepSet):
                            #     sample = self.model.sample(target_in, 1)[0][0]
                            # else:
                            #     sample = None
                            sample = None
                        elif hasattr(self.model, "prior") and self.model.prior is not None:
                            sample = self.model.prior.sample()
                            sample = self.model.reconstruct(target_in, sample)
                            if self.config.loc_scale_prediction:
                                sample = sample[0]
                        else:
                            sample = None
                        
                        gp_samples.append(target_out.cpu().numpy()[:, :, 0])
                        predictions.append(prediction.cpu().numpy()[:, :, 0])
                        if sample is not None:
                            samples.append(sample.cpu().numpy()[:, :, 0])

            gp_samples = np.concatenate(gp_samples, 0)
            predictions = np.concatenate(predictions, 0)
            if len(samples) > 0:
                samples = np.concatenate(samples, 0)
            else:
                samples = None

            t0 = time.time()

            # we should now have arrays of shape (N*B, test_num_target_distribution)
            sqdist = ot.dist(gp_samples, predictions, "euclidean")
            wdist, log = ot.emd2(np.ones((sqdist.shape[0], )) / sqdist.shape[0],
                                    np.ones((sqdist.shape[1], )) / sqdist.shape[1],
                                    sqdist,
                                    numItermax=1e7,
                                    log=True,
                                    return_matrix=True)
            wdist = np.sqrt(wdist)

            if samples is not None:
                sqdist = ot.dist(gp_samples, predictions, "euclidean")
                wdist_sample, log = ot.emd2(np.ones((sqdist.shape[0], )) / sqdist.shape[0],
                                        np.ones((sqdist.shape[1], )) / sqdist.shape[1],
                                        sqdist,
                                        numItermax=1e7,
                                        log=True,
                                        return_matrix=True)
                wdist_sample = np.sqrt(wdist_sample)
            else:
                wdist_sample = np.nan

            print("Wasserstein", time.time() - t0)
            t0 = time.time()

            mmd = mmd_squared(gp_samples, predictions, self.config.test_mmd_alphas, unbiased=True, sqdist12=sqdist, block_size=25*1024)
            if samples is not None:
                mmd_sample = mmd_squared(gp_samples, samples, self.config.test_mmd_alphas, unbiased=True, sqdist12=sqdist, block_size=25*1024)
            else:
                mmd_sample = np.nan

            scores.append(np.array([mmd, mmd_sample, wdist, wdist_sample]))

            print("MMD", time.time() - t0)

        scores = np.stack(scores)
        self.elog.save_numpy_data(scores, "test_distribution.npy")
        self.elog.save_dict(info, "test_distribution.json")

        self.config.target_include_context = target_include_context

    def test_diversity(self):
        import ot
        
        # for the evaluation we want to separate target and context entirely
        target_include_context = self.config.target_include_context
        self.config.target_include_context = False
        self.model.eval()

        info = {}
        info["dims"] = ["instance", "test_num_context", "metric"]
        info["coords"] = {
            "test_num_context": self.config.test_num_context,
            "metric": [
                "wasserstein"
            ]
        }

        scores = []

        for b in range(self.config.test_batches_diversity):

            print("Diversity test batch", b)
            t0 = time.time()

            scores_batch = []

            for num_context in self.config.test_num_context:
                with torch.no_grad():

                    num_failures = 0
                    while num_failures < 100:

                        if num_context == "random":
                            num_context = np.random.randint(*self.config.test_num_context_random)

                        context_in, context_out, target_in, target_out = self.make_batch(self.config.test_batch_size,
                                                                                    self.config.test_x_range,
                                                                                    num_context,
                                                                                    self.config.test_num_target_diversity,
                                                                                    linspace=True)

                        prediction = self.model(context_in, context_out, target_in, store_rep=True)  # (B, N, 1)

                        try:

                            samples = []
                            while len(samples) < self.config.test_latent_samples:
                                if isinstance(self.model, convcnp.ConvCNP):
                                    if isinstance(self.model.l0, convcnp.GPConvDeepSet):
                                        sample = self.model.sample(target_in, 1)[0][0].cpu()
                                    else:
                                        sample = None
                                elif hasattr(self.model, "prior") and self.model.prior is not None:
                                    sample = self.model.prior.sample()
                                    sample = self.model.reconstruct(target_in, sample)
                                    if self.config.loc_scale_prediction:
                                        sample = sample[0]
                                    sample = sample.cpu()
                                else:
                                    sample = None
                                samples.append(sample)
                            samples = torch.stack(samples).cpu().numpy()[..., 0].transpose(1, 0, 2)  # (B, num_samples, N)

                            samples_gp = []
                            for i in range(samples.shape[0]):
                                self.process.fit(context_in[i].cpu().numpy(), context_out[i].cpu())
                                _, _, sample = self.process.predict(target_in[i].cpu().numpy(), return_samples=self.config.test_latent_samples)  # (N, num_samples)
                                samples_gp.append(sample)
                            samples_gp = np.stack(samples_gp).transpose(0, 2, 1)  # (B, num_samples, N)

                            wdist_scores = []
                            for i in range(samples.shape[0]):
                                sqdist = ot.dist(samples[i], samples_gp[i], "euclidean")
                                wdist, log = ot.emd2(np.ones((sqdist.shape[0], )) / sqdist.shape[0],
                                                        np.ones((sqdist.shape[1], )) / sqdist.shape[1],
                                                        sqdist,
                                                        numItermax=1e7,
                                                        log=True,
                                                        return_matrix=True)
                                wdist = np.sqrt(wdist)
                                wdist_scores.append(wdist)
                            wdist_scores = np.array(wdist_scores).reshape(-1, 1, 1)

                            scores_batch.append(wdist_scores)

                            break

                        except RuntimeError:

                            num_failures += 1
                            continue

            scores_batch = np.concatenate(scores_batch, 1)
            scores.append(scores_batch)

        scores = np.concatenate(scores, 0)
        self.elog.save_numpy_data(scores, "test_diversity.npy")
        self.elog.save_dict(info, "test_diversity.json")

        self.config.target_include_context = target_include_context                    



if __name__ == '__main__':

    parser = get_default_experiment_parser()
    parser.add_argument("-rep", "--representation_channels", type=int, default=128)
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults(args.representation_channels)

    run_experiment(NeuralProcessExperiment,
                   DEFAULTS,
                   args,
                   mods=MODS,
                   globs=globals(),
                   remove_mod_none=args.remove_none)
