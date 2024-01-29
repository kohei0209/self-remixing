from .trainer_fuss import FUSSTrainer
from .trainer_libricss import LibriCSSTrainer
from .trainer_wsj import WSJTrainer

trainers = {
    "wsjmix": WSJTrainer,
    "smswsj": WSJTrainer,
    "whamr": WSJTrainer,
    "librimix": WSJTrainer,
    "libricss": LibriCSSTrainer,
    "fuss": FUSSTrainer,
}
