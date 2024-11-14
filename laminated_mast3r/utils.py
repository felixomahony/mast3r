import numpy as np
import cv2
from PIL import Image
import mlflow
import os
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch


def log_images(img_list):
    for i in range(len(img_list)):
        img = Image.open(img_list[i])
        # resize image to 512x288
        img = img.resize((512, 288))
        mlflow.log_image(img, f"image/{i}.png")


def log_depth(num_images, out_data):
    for i in range(num_images):
        for j in range(num_images):
            key = "pts3d"
            depth = out_data[i][j][key][0, :, :, -1].detach().cpu().numpy()
            conf = out_data[i][j]["conf"].detach().cpu().numpy().squeeze()
            pc_10_depth = np.percentile(depth, 10)
            pc_90_depth = np.percentile(depth, 90)
            depth = (depth - pc_10_depth) / (pc_90_depth - pc_10_depth)
            depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)

            # map image to cmap with cv2
            depth_im = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            depth_im = Image.fromarray(depth_im)
            mlflow.log_image(
                depth_im, f"depth/{i}/{'target' if j == 0 else 'reference_'+str(j)}.png"
            )

            # map confidence to cmap with cv2
            conf_lower, conf_upper = 0, 15
            conf = np.clip(conf, conf_lower, conf_upper)
            conf = (conf - conf_lower) / (conf_upper - conf_lower) * 255
            conf = conf.astype(np.uint8)

            conf_im = cv2.applyColorMap(conf, cv2.COLORMAP_WINTER)
            conf_im = cv2.cvtColor(conf_im, cv2.COLOR_BGR2RGB)
            conf_im = Image.fromarray(conf_im)
            mlflow.log_image(
                conf_im,
                f"depth/{i}/{'target' if j == 0 else 'reference_'+str(j)}_conf.png",
            )


def log_obj(num_images, out_data, export_path):
    export_path = Path(export_path)
    for i in range(num_images):
        os.makedirs(export_path / str(i), exist_ok=True)
        for j in range(num_images):
            # save point cloud
            pts3d = out_data[i][j]["pts3d"][0].detach().cpu().numpy()

            # save as .obj file
            with open(
                os.path.join(export_path / str(i), f"point_cloud_{j}.obj"), "w"
            ) as f:
                for pt in pts3d.reshape(-1, 3):
                    f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
                for k in range(pts3d.shape[0] - 1):
                    for l in range(pts3d.shape[1] - 1):
                        f.write(
                            f"f {k*pts3d.shape[1]+l+1} {k*pts3d.shape[1]+l+2} {(k+1)*pts3d.shape[1]+l+2}\n"
                        )
                        f.write(
                            f"f {k*pts3d.shape[1]+l+1} {(k+1)*pts3d.shape[1]+l+2} {(k+1)*pts3d.shape[1]+l+1}\n"
                        )


def log_attn(out_data, img_list):
    patch_size = 16
    size = np.array([288, 512])
    patches = size // patch_size
    numpatches = np.prod(patches)
    num_images = len(img_list)
    num_references = num_images - 1

    X, Y = np.meshgrid(np.arange(patches[1]), np.arange(patches[0] * num_references))
    X, Y = X.flatten(), Y.flatten()

    orig_images = [Image.open(img) for img in img_list]

    # resize images to 512x288
    orig_images = [img.resize((512, 288)) for img in orig_images]

    for i in range(len(orig_images)):
        relevant_data = out_data[i][0]
        attn = relevant_data["attn"].detach().cpu().numpy().squeeze()

        for head in range(attn.shape[0]):
            attn_head = attn[head].reshape(*patches, -1)
            attn_head = np.argmax(attn_head, axis=-1)

            # create canvas
            masks_size = (size[0] * num_references, size[1] * 2, 3)
            masks = np.ones(masks_size, dtype=np.uint8) * 255

            # add image to masks
            orig_image_location_y = masks_size[0] // 2 - size[0] // 2
            masks[
                orig_image_location_y : orig_image_location_y + size[0], : size[1]
            ] = np.array(orig_images[i])

            for j, img_idx in enumerate(
                np.concatenate(
                    (np.arange(i + 1, len(orig_images)), np.arange(0, i - 1))
                )
            ):
                masks[j * size[0] : (j + 1) * size[0], size[1] : size[1] * 2, :] = (
                    np.array(orig_images[img_idx])
                )

            for x_tgt in range(attn_head.shape[1]):
                for y_tgt in range(attn_head.shape[0]):
                    col = (
                        255 * x_tgt // attn_head.shape[1],
                        255 * y_tgt // attn_head.shape[0],
                        255,
                    )

                    cv2.circle(
                        masks,
                        (
                            x_tgt * patch_size + patch_size // 2,
                            y_tgt * patch_size
                            + patch_size // 2
                            + orig_image_location_y,
                        ),
                        2,
                        col,
                        -1,
                    )
                    cv2.circle(
                        masks,
                        (
                            X[attn_head[y_tgt, x_tgt]] * patch_size
                            + int(patch_size * np.random.rand())
                            + size[1],
                            Y[attn_head[y_tgt, x_tgt]] * patch_size
                            + int(patch_size * np.random.rand()),
                        ),
                        2,
                        col,
                        -1,
                    )

            mlflow.log_image(masks, f"attn/{i}/attn_head_{head}.png")


def log_features(out_data, img_list):
    num_images = len(img_list)
    num_references = num_images - 1
    im_shape = (288, 512)

    imgs = [Image.open(img).resize(im_shape[::-1]) for img in img_list]

    for i in range(num_images):
        all_features = np.stack(
            [
                out_data[i][j]["desc"].detach().cpu().numpy().squeeze()
                for j in range(num_images)
            ]
        )
        all_features = all_features.reshape(-1, all_features.shape[-1])
        pca = PCA(n_components=3)
        pca.fit(all_features)

        final_image = np.zeros(
            (im_shape[0] * num_images, im_shape[1] * 2, 3), dtype=np.uint8
        )

        for j in range(num_images):
            features = out_data[i][j]["desc"].detach().cpu().numpy().squeeze()
            shape = features.shape
            features = pca.transform(features.reshape(-1, features.shape[-1]))
            features = features.reshape(*shape[:-1], -1)
            features += 1
            features /= 2
            features = (features * 255).astype(np.uint8)

            final_image[j * im_shape[0] : (j + 1) * im_shape[0], : im_shape[1]] = (
                np.array(imgs[(j + i) % num_images])
            )
            final_image[j * im_shape[0] : (j + 1) * im_shape[0], im_shape[1] :] = (
                features
            )

        mlflow.log_image(final_image, f"features/{i}/features.png")


def log_corres(img_list, corres):
    num_images = len(img_list)
    num_references = num_images - 1
    im_shape = np.array((512, 288))

    imgs = [Image.open(img).resize(tuple(im_shape)) for img in img_list]

    color_00 = torch.tensor([1, 0, 0])
    color_01 = torch.tensor([0, 1, 0])
    color_10 = torch.tensor([0, 0, 1])
    color_11 = torch.tensor([1, 1, 0])

    ctr = 0
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i >= j:
                continue

            corres_i = corres[ctr]
            corres_im_0 = corres_i[0]
            corres_im_1 = corres_i[1]

            im_0 = np.array(imgs[i])
            im_1 = np.array(imgs[j])

            corres_im_0_frac = (
                corres_im_0 - torch.min(corres_im_0, axis=0).values
            ).float()
            corres_im_0_frac /= torch.max(corres_im_0_frac, axis=0).values

            colors = (
                color_00
                * (1 - corres_im_0_frac[:, 0, None])
                * (1 - corres_im_0_frac[:, 1, None])
                + color_01
                * (1 - corres_im_0_frac[:, 0, None])
                * corres_im_0_frac[:, 1, None]
                + color_10
                * corres_im_0_frac[:, 0, None]
                * (1 - corres_im_0_frac[:, 1, None])
                + color_11 * corres_im_0_frac[:, 0, None] * corres_im_0_frac[:, 1, None]
            )

            plt.subplot(1, 2, 1)
            plt.imshow(im_0)
            plt.scatter(corres_im_0[:, 0], corres_im_0[:, 1], c=colors, s=1)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(im_1)
            plt.scatter(corres_im_1[:, 0], corres_im_1[:, 1], c=colors, s=1)
            plt.axis("off")

            plt.tight_layout()

            mlflow.log_figure(plt.gcf(), f"corres/{i}_{j}.png")

            plt.close()

            ctr += 1
