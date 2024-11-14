from mast3r.demo import *
from mast3r.cloud_opt.sparse_ga import symmetric_inference
from mast3r.cloud_opt.sparse_ga import extract_correspondences

import torch


def get_reconstructed_scene_laminated(
    model,
    device,
    silent,
    image_size,
    filelist,
    return_attention=False,
    subsample=8,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)

    pairs = make_ns(imgs)[0]
    # pairs.append(pairs[-1][::-1])
    res = symmetric_inference(
        model._model, *pairs, device=device, return_attention=return_attention
    )

    xij = [r["pts3d"][0] for r in res]
    cij = [r["conf"][0] for r in res]
    dij = [r["desc"][0] for r in res]
    qij = [r["desc_conf"][0] for r in res]

    total_corres = []

    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i >= j:
                continue
            del_ij = j - i
            idxes = [
                i * len(imgs),
                i * len(imgs) + del_ij,
                j * len(imgs),
                j * len(imgs) + ((-del_ij) % len(imgs)),
            ]
            corres = extract_correspondences(
                [dij[ij].detach() for ij in idxes],
                [qij[ij].detach() for ij in idxes],
                device=device,
                subsample=subsample,
            )

            total_corres.append(corres)

    return res, total_corres
