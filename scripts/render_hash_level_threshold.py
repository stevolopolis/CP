import typing as t
import os
from loguru import logger


def get_command(
    hash_level_threshold: t.Optional[int] = None
) -> str:
    base = (
        """ns-render dataset \
        --load-config /home/bird/Desktop/research/nerfstudio/outputs/default/instant-ngp/2023-11-10_092431/config.yml \
        --split train \
        --rendered_output_names rgb """
    )
    base += f"--output-path /home/bird/Desktop/research/nerfstudio/outputs/default/instant-ngp/2023-11-10_092431/renders/th_{hash_level_threshold} "
    if hash_level_threshold is not None:
        base += f"--hash_level_threshold {hash_level_threshold} "

    return base


if __name__ == "__main__":
    # default
    # command = get_command()
    # logger.info(f"Executing {command}")
    # os.system(command)

    # th
    for th in range(2, 16, 2):
        command = get_command(th)
        logger.info(f"Executing {command}")
        os.system(command)
