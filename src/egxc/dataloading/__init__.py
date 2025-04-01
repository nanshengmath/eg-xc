from .base import (
    RawSample,
    BaseDataset,
    PresplitDataset,
    PartiallySplitDataset,
    UnsplitDataset,
    Targets,
    DatasetEnsemble,
    SupportsIndex,
)
from .oqdc_sets import DES370K
from .md17 import MD17
from .qm9 import QM9

from .transform import get_preload_transform, get_jax_transform, ToJaxTransform
from .dataloader import (
    get_dataloaders,
    get_sample_for_model_init,
    DataLoaders,
)

from typing import Dict, Callable

key_to_dataset: Dict[str, Callable] = {
    'md17': MD17,
    'des370k': DES370K,
}
