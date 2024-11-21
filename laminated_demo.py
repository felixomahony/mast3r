from laminated_mast3r.model import LaminatedMast3rModel
from dust3r.demo import get_reconstructed_scene
from laminated_mast3r.reconstruction import get_reconstructed_scene_laminated
from laminated_mast3r.utils import *

import click
import functools
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import cv2
import mlflow
from PIL import Image
import time


def load_config(config_path):
    if config_path is None:
        # Default config file path
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        print(f"Config file not found at {config_path}. Using base config.")
    return {}


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    required=False,
    help="Config file",
    default="./experiments/church/laminated/config.yaml",
)
@click.option(
    "--name",
    "-n",
    type=click.STRING,
    required=False,
    help="Name of the run",
    default=None,
)
@click.option(
    "--export_path",
    "-e",
    type=click.Path(),
    required=False,
    help="Export path",
    default=None,
)
@click.option(
    "--device",
    "-d",
    type=click.STRING,
    required=False,
    help="Device to use",
    default=None,
)
@click.option(
    "--image_folder",
    "-d",
    type=click.Path(),
    required=False,
    help="Image folder",
    default=None,
)
@click.option(
    "--exclude_imgs",
    "-e",
    type=click.STRING,
    required=False,
    help="Images to exclude",
    default=None,
)
@click.option(
    "--primary_image",
    "-p",
    type=click.STRING,
    required=False,
    help="Primary image",
    default=None,
)
@click.option(
    "--image_order",
    "-o",
    type=click.STRING,
    required=False,
    help="Image order",
    default=None,
)
@click.option(
    "--experiment",
    "-e",
    type=click.STRING,
    required=False,
)
@click.option(
    "--fast",
    "-f",
    is_flag=True,
    help="Use fast features",
)
def main(
    config,
    name,
    export_path,
    device,
    image_folder,
    exclude_imgs,
    primary_image,
    image_order,
    experiment,
    fast,
):

    config = load_config(config)
    name = name or config.get("name", "base")
    export_path = export_path or config.get("export_path", "./output")
    device = device or config.get("device", "cpu")
    image_folder = image_folder or config.get("image_folder", "./data/")
    exclude_imgs = exclude_imgs or config.get("exclude_imgs", None)
    primary_image = primary_image or config.get("primary_image", None)
    image_order = image_order or config.get("image_order", None)
    experiment = experiment or config.get("experiment", "none")
    fast = fast or config.get("fast", False)

    # initialise logging
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=name)

    # make directory for meshes
    os.makedirs(export_path, exist_ok=True)

    # get list of images
    image_list = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
    ]

    # exclude on the basis of exclude_imgs
    if exclude_imgs is not None:
        exclude_imgs = exclude_imgs.split(",")
        exclude_imgs = [img.replace(" ", "") for img in exclude_imgs]
        image_list = [
            img
            for img in image_list
            if img.split("/")[-1].replace(" ", "") not in exclude_imgs
        ]

    # set primary image
    assert not (
        primary_image and image_order
    ), "Only one of primary_image and image_order can be set"
    if primary_image is not None:
        primary_image = primary_image.replace(" ", "")
        image_list_names = [img.split("/")[-1].replace(" ", "") for img in image_list]
        if primary_image not in image_list_names:
            raise ValueError("Primary image not in image list")
        primary_image_idx = image_list_names.index(primary_image)
        image_list = [image_list[primary_image_idx]] + [
            image_list[i] for i in range(len(image_list)) if i != primary_image_idx
        ]
        print("Primary image set to: ", image_list[0])
    elif image_order is not None:
        image_order = image_order.split(",")
        image_order = [img.replace(" ", "") for img in image_order]
        image_list_names = [img.split("/")[-1].replace(" ", "") for img in image_list]
        # check image order is a subset of image_list_names
        assert set(image_order).issubset(
            set(image_list_names)
        ), f"Image order contains invalid names: {set(image_order) - set(image_list_names)}"
        image_list = [image_list[image_list_names.index(img)] for img in image_order]
        print("Image order set to: ", image_list)
    else:
        print("No image order given, using first image as primary: ", image_list[0])

    num_images = len(image_list)

    # set some default values
    scenegraph_type = "complete"
    winsize = 1
    refid = 0

    reconstruction_params = {
        "filelist": image_list,
        "return_attention": True,
        "subsample": 8,
    }

    model = LaminatedMast3rModel.default(device=device)
    recon_fun = functools.partial(
        get_reconstructed_scene_laminated, model, device, False, 512, fast=fast
    )

    start_time = time.time()
    out_data, corres = recon_fun(**reconstruction_params)
    end_time = time.time()

    out_data = [
        [out_data[i] for i in range(j * num_images, (j + 1) * num_images)]
        for j in range(num_images)
    ]

    print("Logging data to mlflow")
    # now we want to export data
    log_images(image_list)
    log_depth(num_images, out_data)
    log_obj(num_images, out_data, export_path)
    log_attn(out_data, image_list)
    log_features(out_data, image_list)
    log_corres(image_list, corres)

    # log time metric
    mlflow.log_metric("inference_time", end_time - start_time)

    mlflow.end_run()


if __name__ == "__main__":
    main()
