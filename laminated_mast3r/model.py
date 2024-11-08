from typing import Any
from mast3r.model import AsymmetricMASt3R


class LaminatedMast3rModel:
    def __init__(self, model):
        self._model = model

    @classmethod
    def default(cls, device="cuda"):
        weights_path = (
            "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        )
        return cls(model=AsymmetricMASt3R.from_pretrained(weights_path).to(device))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._model(*args, **kwds)


if __name__ == "__main__":
    m: LaminatedMast3rModel = LaminatedMast3rModel.default(device="cuda")
