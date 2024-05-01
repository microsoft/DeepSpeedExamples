import argparse
from .clients import client_config_classes
from .benchmark_runner import BenchmarkConfig


def parse_args_to_configs():
    def add_model(parser, model):
        fields = model.model_fields
        for name, field in fields.items():
            nargs = None
            field_type = field.annotation
            if getattr(field.annotation, "_name", "") == "List":
                nargs = "+"
                field_type = field.annotation.__args__[0]
            parser.add_argument(
                f"--{name}",
                dest=name,
                nargs=nargs,
                type=field_type,
                required=getattr(field, "required", False),
                default=getattr(field, "default", None),
                help=getattr(field, "description", ""),
            )

    parser = argparse.ArgumentParser()
    add_model(parser, BenchmarkConfig)
    args, remaining_args = parser.parse_known_args()
    unused_args = set(remaining_args)
    benchmark_config = BenchmarkConfig(**vars(args))

    client_config_class = client_config_classes[benchmark_config.api]
    parser = argparse.ArgumentParser()
    add_model(parser, client_config_class)
    args, remaining_args = parser.parse_known_args()
    unused_args = unused_args.intersection(remaining_args)
    client_config = client_config_class(**vars(args))

    if unused_args:
        raise ValueError(f"Unused arguments: {unused_args}")

    return benchmark_config, client_config
