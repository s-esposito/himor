import os
import time
from dataclasses import dataclass

import torch
import tyro
from loguru import logger as guru

from flow3d.renderer import Renderer

torch.set_float32_matmul_precision("high")


@dataclass
class RenderConfig:
    ckpt_path: str
    port: int = 8890


def main(cfg: RenderConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg.ckpt_path
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir="./",
        port=cfg.port,
    )

    guru.info(f"Starting rendering from {renderer.global_step=}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))

