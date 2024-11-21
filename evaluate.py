from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs, make_ns
from dust3r.utils.image import load_images


from pathlib import Path
import numpy as np
import mlflow
import time
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

DATA_PATH = Path("/Users/cambridge/Documents/Data/Caterpillar/")
TMP_PATH = "./log"
WEIGHTS_PATH = "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DEVICE = "cpu"

IM_SIZE = 512
N_RUNS = 100
MAX_N_IMGS = 3
FAST = True


def get_cam2w_gt(data_path):
    cam2w = data_path / "Caterpillar_COLMAP_SfM.log"
    with open(cam2w, "r") as f:
        lines = f.readlines()
    mats = []
    for i in range(len(lines) // 5):
        mat = np.array(
            [
                [float(lines[5 * i + 1].split(" ")[j]) for j in range(4)],
                [float(lines[5 * i + 2].split(" ")[j]) for j in range(4)],
                [float(lines[5 * i + 3].split(" ")[j]) for j in range(4)],
                [float(lines[5 * i + 4].split(" ")[j]) for j in range(4)],
            ]
        )
        mats.append(mat)
    return np.array(mats)


def norm_position(pos):
    pos_norm = pos - pos.mean(axis=0)
    trace = np.trace(pos_norm @ pos_norm.T)
    pos_norm /= np.sqrt(trace)
    return pos_norm


def calc_ate(cam2w_gt, cam2w_pred):
    pos_gt = cam2w_gt[:, :3, 3]
    pos_gt = norm_position(pos_gt)

    pos_pred = cam2w_pred[:, :3, 3]
    pos_pred = norm_position(pos_pred)

    trajectories_gt = pos_gt[:, None] - pos_gt[None, :]
    dists_gt = np.linalg.norm(trajectories_gt, axis=-1)

    trajectories_pred = pos_pred[:, None] - pos_pred[None, :]
    dists_pred = np.linalg.norm(trajectories_pred, axis=-1)

    dists_error = np.abs(dists_gt - dists_pred)

    mean_dists_error = np.mean(
        dists_error[
            (
                np.ones(dists_error.shape, dtype=int)
                - np.eye(dists_error.shape[0], dtype=int)
            ).astype(bool)
        ]
    )

    return mean_dists_error


def calc_rra(cam2w_gt, cam2w_pred):
    rot_gt = cam2w_gt[:, :3, :3]
    rot_pred = cam2w_pred[:, :3, :3]

    delta_rot_gt = rot_gt[:, None] @ np.transpose(rot_gt, (0, 2, 1))[None]
    trace_gt = np.trace(delta_rot_gt, axis1=-2, axis2=-1)

    delta_rot_pred = rot_pred[:, None] @ np.transpose(rot_pred, (0, 2, 1))[None]
    trace_pred = np.trace(delta_rot_pred, axis1=-2, axis2=-1)

    ang_gt = np.arccos((trace_gt - 1) / 2)
    ang_pred = np.arccos((trace_pred - 1) / 2)

    ang_error = np.abs(ang_gt - ang_pred)

    mean_ang_error = np.nanmean(
        ang_error[
            (
                np.ones(ang_error.shape, dtype=int)
                - np.eye(ang_error.shape[0], dtype=int)
            ).astype(bool)
        ]
    )

    return mean_ang_error


if __name__ == "__main__":
    model = AsymmetricMASt3R.from_pretrained(WEIGHTS_PATH).to(DEVICE)
    cam2w_gt = get_cam2w_gt(DATA_PATH)

    winsize = 1
    refid = 0
    scenegraph_type = "complete"
    win_cyclic = False
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append("noncyclic")
    scene_graph = "-".join(scene_graph_params)

    # initialise logging
    mlflow.set_experiment("evaluation_caterpillar_")
    mlflow.start_run(run_name="trial")

    full_results = []
    for i in range(N_RUNS):
        num_images = np.random.randint(3, MAX_N_IMGS + 1)
        img_ids = np.random.randint(1, 384, num_images)
        img_paths = [str(DATA_PATH / f"imgs/{(i+1):06d}.jpg") for i in img_ids]

        imgs = load_images(img_paths, size=IM_SIZE, verbose=False)
        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            prefilter=None,
            symmetrize=True,
            # laminate=laminate,
        )
        relevant_cam2w_gt = cam2w_gt[img_ids]

        base_start_time = time.time()
        scene_base = sparse_global_alignment(
            img_paths, pairs, TMP_PATH, model, device=DEVICE, fast_features=FAST
        )
        base_inference_time = time.time() - base_start_time
        cam2w_pred_base = np.array(scene_base.cam2w.detach().cpu())

        ate_base = calc_ate(relevant_cam2w_gt, cam2w_pred_base)
        rra_base = calc_rra(relevant_cam2w_gt, cam2w_pred_base)

        lam_start_time = time.time()
        scene_lam = sparse_global_alignment(
            img_paths,
            pairs,
            TMP_PATH,
            model,
            device=DEVICE,
            laminate=True,
            fast_features=FAST,
        )
        lam_inference_time = time.time() - lam_start_time
        cam2w_pred_lam = np.array(scene_lam.cam2w.detach().cpu())

        ate_lam = calc_ate(relevant_cam2w_gt, cam2w_pred_lam)
        rra_lam = calc_rra(relevant_cam2w_gt, cam2w_pred_lam)

        # log results
        mlflow.log_metric("ate_base", ate_base)
        mlflow.log_metric("rra_base", rra_base)
        mlflow.log_metric("ate_lam", ate_lam)
        mlflow.log_metric("rra_lam", rra_lam)
        mlflow.log_metric("base_inference_time", base_inference_time)
        mlflow.log_metric("lam_inference_time", lam_inference_time)

        results_dict = {
            "img_ids": img_ids,
            "ate_base": ate_base,
            "rra_base": rra_base,
            "ate_lam": ate_lam,
            "rra_lam": rra_lam,
            "base_inference_time": base_inference_time,
            "lam_inference_time": lam_inference_time,
        }
        full_results.append(results_dict)

        if len(full_results) % 1 == 0:
            ates_base = [res["ate_base"] for res in full_results]
            rras_base = [res["rra_base"] for res in full_results]

            ates_lam = [res["ate_lam"] for res in full_results]
            rras_lam = [res["rra_lam"] for res in full_results]

            base_inference_times = [res["base_inference_time"] for res in full_results]
            lam_inference_times = [res["lam_inference_time"] for res in full_results]

            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(1, 3, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])

            sns.scatterplot(x=ates_base, y=ates_lam, ax=ax1)
            ax1.plot(
                [0, max(np.max(ates_base), np.max(ates_lam))],
                [0, max(np.max(ates_base), np.max(ates_lam))],
                "k--",
            )
            ax1.set_xlabel("ATE Base")
            ax1.set_ylabel("ATE Laminated")

            sns.scatterplot(x=rras_base, y=rras_lam, ax=ax2)
            ax2.set_xlabel("RRA Base")
            ax2.set_ylabel("RRA Laminated")
            ax2.plot(
                [0, max(np.max(ates_base), np.max(ates_lam))],
                [0, max(np.max(ates_base), np.max(ates_lam))],
                "k--",
            )

            sns.histplot(
                x=base_inference_times, ax=ax3, color="b", alpha=0.5, label="Base"
            )
            sns.histplot(
                x=lam_inference_times, ax=ax3, color="r", alpha=0.5, label="Laminated"
            )
            ax3.legend()
            ax3.set_xlabel("Inference Time")

            # log with mlflow
            mlflow.log_figure(fig, "results.png")
    mlflow.end_run()
