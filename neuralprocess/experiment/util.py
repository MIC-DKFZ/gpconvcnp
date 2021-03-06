import argparse
import os
import glob
import shutil
from trixi.util import Config, GridSearch


def get_default_experiment_parser():
    """
    Construct an argument parser with many options to run experiments.

    Returns:
        argparse.ArgumentParser: The argument parser.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Working directory for experiment.")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to a config file."
    )
    parser.add_argument(
        "-v", "--visdomlogger", action="store_true", help="Use visdomlogger."
    )
    parser.add_argument("-tx", "--tensorboardxlogger", type=str, default=None)
    parser.add_argument("-tl", "--telegramlogger", action="store_true")
    parser.add_argument(
        "-dc",
        "--default_config",
        type=str,
        default="DEFAULTS",
        help="Select a default Config",
    )
    parser.add_argument("-ad", "--automatic_description", action="store_true")
    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="Path to resume from"
    )
    parser.add_argument(
        "-irc",
        "--ignore_resume_config",
        action="store_true",
        help="Ignore Config in experiment we resume from.",
    )
    parser.add_argument(
        "-test", "--test", action="store_true", help="Run test instead of training"
    )
    parser.add_argument(
        "-g", "--grid", type=str, help="Path to a config for grid search"
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip configs for which an experiment exists, only for grid search",
    )
    parser.add_argument(
        "-m",
        "--mods",
        type=str,
        nargs="+",
        default=None,
        help="Mods are Config stubs to update only relevant parts for a certain setup.",
    )
    parser.add_argument(
        "-ct",
        "--copy_test",
        action="store_true",
        help="Copy test files to original experiment.",
    )
    parser.add_argument(
        "-rmn",
        "--remove_none",
        action="store_true",
        help="Remove items that a mod sets to None.",
    )

    return parser


def run_experiment(
    experiment, configs, args, mods=None, remove_mod_none=False, **kwargs
):
    """
    Convenience function to run an experiment. Main purpose is to allow
    selecting from multiple configs and applying different modifications
    to a config.

    Args:
        experiment (trixi.experiment.Experiment): A trixi experiment.
        configs (dict): A dictionary of trixi.Config to choose from.
        args (argparse.Namespace): Result from an argument parser.
        mods (dict): A dictionary of trixi.Config that can update values
            in configs.
        remove_mod_none (bool): If True, setting a value to None in a mod
            will remove the entry from the config (instead of updating it
            to None).

    """

    # set a few defaults
    if "explogger_kwargs" not in kwargs:
        kwargs["explogger_kwargs"] = dict(
            folder_format="{experiment_name}_%Y%m%d-%H%M%S"
        )
    if "explogger_freq" not in kwargs:
        kwargs["explogger_freq"] = 1
    if "resume_save_types" not in kwargs:
        kwargs["resume_save_types"] = ("model", "simple", "th_vars", "results")

    # construct the experiment config
    config = Config(file_=args.config) if args.config is not None else Config()
    config.update_missing(configs[args.default_config].deepcopy())
    if args.mods is not None and mods is not None:
        for mod in args.mods:
            config.update(mods[mod])
            if remove_mod_none:
                for key, val in mods[mod].flat().items():
                    if val is None:
                        del config[key]
    config = Config(config=config, update_from_argv=True)

    # get existing experiments to be able to skip certain configs.
    # this is useful for grid search.
    if args.skip_existing:
        existing_configs = []
        for exp in os.listdir(args.base_dir):
            try:
                existing_configs.append(
                    Config(
                        file_=os.path.join(args.base_dir, exp, "config", "config.json")
                    )
                )
            except Exception as e:
                pass

    # construct grid search
    if args.grid is not None:
        grid = GridSearch().read(args.grid)
    else:
        grid = [{}]

    for combi in grid:

        config.update(combi)

        if args.skip_existing:
            skip_this = False
            for existing_config in existing_configs:
                if existing_config.contains(config):
                    skip_this = True
                    break
            if skip_this:
                continue

        if "backup_every" in config:
            kwargs["save_checkpoint_every_epoch"] = config["backup_every"]

        # construct logger options
        loggers = {}
        if args.visdomlogger:
            loggers["v"] = ("visdom", {}, 1)
        if args.tensorboardxlogger is not None:
            if args.tensorboardxlogger == "same":
                loggers["tx"] = ("tensorboard", {}, 1)
            else:
                loggers["tx"] = (
                    "tensorboard",
                    {"target_dir": args.tensorboardxlogger},
                    1,
                )

        if args.telegramlogger:
            kwargs["use_telegram"] = True

        # create an automatic description from the difference of the
        # current config to the defaults
        if args.automatic_description:
            difference_to_default = Config.difference_config_static(
                config, configs["DEFAULTS"]
            ).flat(keep_lists=True, max_split_size=0, flatten_int=True)
            description_str = ""
            for key, val in sorted(difference_to_default.items()):
                val = val[0]
                description_str = "{} = {}\n{}".format(key, val, description_str)
            config.description = description_str

        # initalize the actual experiment object
        exp = experiment(
            config=config,
            base_dir=args.base_dir,
            resume=args.resume,
            ignore_resume_config=args.ignore_resume_config,
            loggers=loggers,
            **kwargs
        )

        trained = False

        # run training
        if args.resume is None or args.test is False:
            exp.run()
            trained = True

        # run test
        if args.test:
            exp.run_test(setup=not trained)
            if isinstance(args.resume, str) and exp.elog is not None and args.copy_test:
                for f in glob.glob(os.path.join(exp.elog.save_dir, "test*")):
                    if os.path.isdir(f):
                        shutil.copytree(
                            f, os.path.join(args.resume, "save", os.path.basename(f))
                        )
                    else:
                        shutil.copy(f, os.path.join(args.resume, "save"))