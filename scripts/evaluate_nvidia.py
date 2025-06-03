import argparse
import json
import os.path as osp
from glob import glob
from itertools import product
import os

import cv2
import imageio.v3 as iio
import numpy as np
import roma
import torch
from tqdm import tqdm

from flow3d.metrics import mLPIPS, mPSNR, mSSIM, CLIP


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the data directory that contains all the sequences.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    help="Path to the result directory that contains the results."
)
args = parser.parse_args()


def generate_masks(mask, val_names, threshold=0.95, kernel_size=(3, 3), erode_iterations=5):
    prefixes = list(set(int(name.split("_")[0]) for name in val_names))

    # Initialize masks for each unique prefix
    aggregated_masks = {prefix: np.zeros_like(mask[0], dtype=np.float64)for prefix in prefixes}

    # Aggregate masks based on the unique prefixs
    for i, name in enumerate(val_names):
        prefix = int(name.split("_")[0])
        aggregated_masks[prefix] += mask[i]
    
    processed_masks = {}
    kernel = np.ones(kernel_size, dtype=np.uint8)

    for prefix, agg_mask in aggregated_masks.items():
        binary_mask = (agg_mask > agg_mask.max() * threshold).astype(np.float64)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=erode_iterations)
        processed_masks[prefix] = eroded_mask

    # Update original masks using the processed masks
    for i, name in enumerate(val_names):
        prefix = int(name.split("_")[0])
        mask[i] = 255.0 * processed_masks[prefix]
    
    return mask


def load_data_dict(data_dir, train_names, val_names):
    val_imgs = np.array(
        [iio.imread(osp.join(data_dir, "rgb/2x", f"{name}.png")) for name in val_names]
    )
    val_times = np.array(
        [int(t.split("_")[1]) for t in val_names]
    )
    return {
        "val_imgs": val_imgs,
        "val_times": val_times
    }


def load_result_dict(result_dir, val_names):
    try:
        pred_val_imgs = [iio.imread(osp.join(result_dir,"rgb", f"{name}.png")) for name in val_names]
    except:
        pred_val_imgs = None
        masks = None
    masks = np.array([pred_val_img[..., 3] for pred_val_img in pred_val_imgs])
    pred_val_imgs = np.array([pred_val_img[..., :-1] for pred_val_img in pred_val_imgs])
    masks = generate_masks(masks,val_names)

    return {
        "pred_val_imgs": pred_val_imgs,
        "masks": masks
    }

def evaluate_nv(data_dict, result_dict):
    device = "cuda"
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)
    clip_metric = CLIP().to(device)
    clipt_metric = CLIP().to(device)

    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3].to(device)
    pred_val_imgs = torch.from_numpy(result_dict["pred_val_imgs"]).to(device)
    val_times = torch.from_numpy(data_dict["val_times"]).to(device)
    val_masks = torch.from_numpy(result_dict["masks"]).to(device)

    for i in tqdm(range(len(val_imgs))):
        val_img = val_imgs[i] / 255.0
        pred_val_img = pred_val_imgs[i] / 255.0
        val_mask = val_masks[i] / 255.0
        if i + 5 < len(val_imgs):
            if val_times[i+5] - val_times[i] == 5:
                next_pred_val_img = pred_val_imgs[i+5] / 255.0
                next_val_mask = val_masks[i+5] / 255.0
                clipt_metric.update(pred_val_img[None]*val_mask[None][...,None], next_pred_val_img[None]*val_mask[None][...,None])
        clip_metric.update(val_img[None]*val_mask[None][...,None], pred_val_img[None]*val_mask[None][...,None])
        psnr_metric.update(val_img, pred_val_img, val_mask)
        ssim_metric.update(val_img[None], pred_val_img[None], val_mask[None])
        lpips_metric.update(val_img[None], pred_val_img[None], val_mask[None])
    
    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    clip = clip_metric.compute().item()
    clipt = clipt_metric.compute().item()
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mSSIM: {mssim:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    print(f"NV mCLIP-I: {clip:.4f}")
    print(f"NV mCLIP-T: {clipt:.4f}")
    return mpsnr, mssim, mlpips, clip, clipt


if __name__ == "__main__":
    seq_name = args.data_dir.split("/")[-2]

    print("=========================================")
    print(f"Evaluating {seq_name}")
    print("=========================================")
    data_dir = osp.join(args.data_dir, seq_name)
    if not osp.exists(data_dir):
        data_dir = args.data_dir
    if not osp.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} not found.")
    result_dir = osp.join(args.result_dir, seq_name, "results/")
    if not osp.exists(result_dir):
        result_dir = osp.join(args.result_dir, "results/")
    if not osp.exists(result_dir):
        raise ValueError(f"Result directory {result_dir} not found.")
    
    with open(osp.join(data_dir, "splits/train.json")) as f:
        train_names = json.load(f)["frame_names"]
    with open(osp.join(data_dir, "splits/val.json")) as f:
        val_names = json.load(f)["frame_names"]
    
    data_dict = load_data_dict(data_dir, train_names, val_names)
    result_dict = load_result_dict(result_dir, val_names)
    if len(data_dict["val_imgs"]) > 0:
        if result_dict["pred_val_imgs"] is None:
            print("No NV results found.")
        mpsnr, mssim, mlpips, clip, clipt= evaluate_nv(data_dict, result_dict)
