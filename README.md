# GP-ConvCNP

This repository contains examples and pretrained models for our UAI 2021 paper called "GP-ConvCNP: Better Generalization for Convolutional Conditional Neural Processes on Time Series Data". You can explore examples interactively in the `examples.ipynb` (in the same folder as this README) Jupyter notebook and also re-run our experiments. Further instructions are below.

## Installation

If you just want to look at the examples, it's enough to install the package in this repository via

    pip install path/to/this/folder

which will create a package called `neuralprocess` in your current Python environment. The package includes our GP-ConvCNP implementation, but also implementations of other major Neural Process variants (ConvCNP, NP, ANP), hence the name. We recommend you create a new virtualenv or conda environment before installing. If you also want to be able to run the experiments yourself, you need to install with the `experiment` option

    pip install path/to/this/folder[experiment]

Alternatively you can run

    pip install -r requirements_experiment.txt

after the regular installation. You might receive an error that trixi requires a specific version of scikit-learn, which you can safely ignore.

## Example Notebook

The notebook `examples.ipynb` allows you to create plots similar to the ones in the paper. Simply run

    jupyter notebook

from this folder (or above), open the notebook and follow the instructions inside.

## Running the Experiments

Our experiment script is part of the `neuralprocess` package and can be found at `neuralprocess/experiment/neuralprocessexperiment.py`. You will find that our experiment uses the PytochExperiment class from [trixi](https://trixi.readthedocs.io/en/develop/) for logging and configuration. This gives us a great deal of flexibility, but we will only list the relevant options to reproduce the experiments from the paper. You basically only need to run the following command

    python neuralprocessexperiment.py LOG_DIR -m MODS

The modifications are defined at the top of the file. The default configuration will be a Neural Process, trained on samples from a GP with an RBF kernel. You can apply the following modifications:

* `DETERMINISTICENCODER` adds a deterministic path to Neural Process. Required for Attentive Neural Processes.
* `ATTENTION` will use an ANP instead of a NP. Requires the `DETERMINISTICENCODER` mod.
* `CONVCNP` will use a ConvCNP instead of a NP.
* `GPCONVCNP` will use a GP-ConvCNP instead of a NP. Requires the `CONVCNP` mod to be set.
* `LEARNNOISE` makes sigma^2 in the GP learnable. This was used in the experiments in the paper.
* `MATERNKERNEL` will train on functions from GP with a Matern-5/2 kernel.
* `WEAKLYPERIODICKERNEL` will train on functions from a GP with a weakly periodic kernel as defined in the ConvCNP paper.
* `STEP` will train on step functions
* `FOURIER` will train on random Fourier series.
* `LOTKAVOLTERRA` will train on population dynamics generated from Lotka-Volterra equations.
* `TEMPERATURE` will train on temperature measurements taken from [here](https://www.kaggle.com/selfishgene/historical-hourly-weather-data).
* `LONG` will double the number of training epochs.

Beyond that you can modify any value in the configuration directly, including deeper levels. For example, if you have the `ATTENTION` option activated, but you only want to use 4 heads in the attention mechanism, you could add the flag `--modules_kwargs.attention.num_heads 4`. Other useful flags are

* `-v` will log to Visdom. You need to start a Visdom server beforehand with `python -m visdom.server --port 8080`
* `-ad` will generate a description for the experiment by looking at the difference to the default configuration. The description will be saved as part of the config in the logging directory you specify.

To give a more illustrative example, let's assume you want to run GP-ConvCNP on step functions with twice the default amount of epochs, but with a larger initial learning rate. You also want to run the tests at the end, but for some reason only those for prediction and reconstruction ability. The training should be logged to Visdom and you want an automatic description generated. Your command would look like this:

    python neuralprocessexperiment.py LOG_DIR -v -ad -m STEP CONVCNP GPCONVCNP LONG --optimizer_kwargs.lr 1e-2 --test --test_diversity false

