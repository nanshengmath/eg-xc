import optax
from dataclasses import dataclass

from optax import Schedule
from typing import Literal, Dict, Any


@dataclass
class ScheduleConfig:
    base_rate: float
    min_rate: float
    warmup_steps: int
    decay_steps: int
    warmup_schedule: Literal['linear'] = 'linear'
    decay_schedule: Literal['linear', 'cosine'] = 'cosine'


@dataclass
class PlateauConfig:
    factor: float
    patience: int
    cooldown: int
    accumulation_size: int
    min_scale: float


@dataclass
class OptConfig:
    name: Literal['adam', 'prodigy']
    weight_decay: float
    schedule_config: ScheduleConfig
    plateau_handling: PlateauConfig | None
    apply_every: int
    clip_grad_max_norm: float | None
    skip_nans: int

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'OptConfig':
        return cls(
            name=config['name'],
            weight_decay=config['weight_decay'],
            schedule_config=ScheduleConfig(**config['schedule']),
            plateau_handling=PlateauConfig(**config.get('plateau_handling', None)),
            apply_every=config['apply_every'],
            clip_grad_max_norm=config['clip_grad_max_norm'],
            skip_nans=config['skip_nans'],
        )


def get_warmup_schedule(config: ScheduleConfig) -> Schedule:
    if config.warmup_schedule == 'linear':
        return optax.linear_schedule(
            config.min_rate, config.base_rate, config.warmup_steps
        )
    else:
        raise ValueError(f'Invalid warmup schedule: {config.warmup_schedule}')


def get_decay_schedule(config: ScheduleConfig) -> Schedule:
    if config.decay_schedule == 'linear':
        return optax.linear_schedule(
            config.base_rate, config.min_rate, config.decay_steps
        )
    elif config.decay_schedule == 'cosine':
        alpha = config.min_rate / config.base_rate
        return optax.cosine_decay_schedule(config.base_rate, config.decay_steps, alpha)
    else:
        raise ValueError(f'Invalid decay schedule: {config.decay_schedule}')


def get_plateau_schedule(config: PlateauConfig) -> optax.GradientTransformationExtraArgs:
    return optax.contrib.reduce_on_plateau(
        factor=config.factor,
        patience=config.patience,
        cooldown=config.cooldown,
        accumulation_size=config.accumulation_size,
        min_scale=config.min_scale,
    )


def get_adam(schedule: Schedule, weight_decay: float) -> optax.GradientTransformation:
    if weight_decay > 0.0:
        optimizer = optax.adamw(schedule, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate=schedule)
    return optimizer


def get_prodigy(schedule: Schedule, weight_decay: float) -> optax.GradientTransformation:
    assert (
        OptConfig.schedule_config.base_rate == 1
    ), 'Prodigy optimizer should use base learning rate of 1'
    return optax.contrib.prodigy(
        learning_rate=schedule,
        weight_decay=weight_decay,
    )


def get_optimizer(config: OptConfig) -> optax.GradientTransformation | optax.GradientTransformationExtraArgs:
    """
    Get the optimizer and its state from the configuration dictionary.
    """
    warmup_schedule = get_warmup_schedule(config.schedule_config)
    decay_schedule = get_decay_schedule(config.schedule_config)
    schedule = optax.join_schedules(
        [warmup_schedule, decay_schedule],
        boundaries=(config.schedule_config.warmup_steps,),
    )

    gradient_transforms = []
    if config.name == 'adam':
        gradient_transforms.append(get_adam(schedule, config.weight_decay))
    elif config.name == 'prodigy':
        pass
    else:
        raise ValueError(f'Invalid optimizer: {config.name}')

    if config.plateau_handling is not None:
        gradient_transforms.append(get_plateau_schedule(config.plateau_handling))

    if config.clip_grad_max_norm is not None:
        gradient_transforms.append(optax.clip_by_global_norm(config.clip_grad_max_norm))

    if config.apply_every > 1:
        gradient_transforms.append(optax.apply_every(config.apply_every))

    opt = optax.chain(*gradient_transforms)

    if config.skip_nans > 0:  # make resilient to occasional NaNs
        opt = optax.apply_if_finite(opt, max_consecutive_errors=config.skip_nans)

    return opt
