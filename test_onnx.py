import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort  # type: ignore
from PIL import Image, ImageDraw
from torchvision import datasets  # type: ignore


# 默认使用 CIFAR-10 类别名，可用 --labels 传入自定义标签文件
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _infer_shape(onnx_shape, batch_size):
    shape = []
    for dim in onnx_shape:
        if dim is None:
            shape.append(batch_size)
        elif isinstance(dim, str):
            shape.append(batch_size if dim.lower() == "batch" else batch_size)
        else:
            shape.append(int(dim))
    return shape


def _softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _preprocess_image(image_path, h, w, mean, std):
    img = Image.open(image_path).convert("RGB")
    # 用于推理的尺寸
    img_resized = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    # 用于可视化：保留原图的像素感，可按模型输入尺寸放大
    vis_img = img.copy().resize((w, h), Image.NEAREST)
    return arr, vis_img


def _maybe_load_labels(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _visualize(img: Image.Image, text: str, out_path: str):
    draw = ImageDraw.Draw(img)
    pad = 6
    draw.rectangle([0, 0, img.width, 30 + pad * 2], fill=(0, 0, 0, 128))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    img.save(out_path)
    print(f"Saved visualization to {out_path}")


def _save_cifar10_sample(root: str, index: int, train: bool, out_path: str):
    # 从本地 CIFAR-10 数据集中取一张图片并保存，避免额外下载
    dataset = datasets.CIFAR10(root=root, train=train, download=False)
    img, label = dataset[index]
    img.save(out_path)
    return out_path, label


def _save_cifar10_batch(root: str, start: int, count: int, train: bool, out_dir: str):
    dataset = datasets.CIFAR10(root=root, train=train, download=False)
    out_paths = []
    labels = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        idx = start + i
        if idx >= len(dataset):
            break
        img, label = dataset[idx]
        out_path = Path(out_dir) / f"cifar10_{'train' if train else 'test'}_{idx}.png"
        img.save(out_path)
        out_paths.append(str(out_path))
        labels.append(label)
    return out_paths, labels


def _collect_images(directory: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for entry in sorted(Path(directory).iterdir()):
        if entry.is_file() and entry.suffix.lower() in exts:
            files.append(str(entry))
    return files


def run_inference(
    onnx_path,
    batch_size=1,
    provider=None,
    image_path=None,
    image_paths: Optional[List[str]] = None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    labels: Optional[List[str]] = None,
    topk=5,
    out_viz: Optional[str] = None,
):
    providers = None if provider is None else [provider]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp = sess.get_inputs()[0]
    input_shape = _infer_shape(inp.shape, batch_size)

    if len(input_shape) < 4:
        raise ValueError(f"Unexpected input shape: {input_shape}")
    _, c, h, w = input_shape
    if c != 3:
        raise ValueError(f"Expected 3-channel input, got {c}")

    vis_imgs = []
    arr_list = []
    if image_paths:
        for img_p in image_paths:
            arr, vis_img = _preprocess_image(
                img_p, h, w, np.array(mean), np.array(std)
            )
            arr_list.append(arr)
            vis_imgs.append((img_p, vis_img))
        x = np.stack(arr_list, axis=0).astype(np.float32)
        batch_size = len(arr_list)
    elif image_path:
        arr, vis_img = _preprocess_image(
            image_path, h, w, np.array(mean), np.array(std)
        )
        x = np.expand_dims(arr, axis=0).astype(np.float32)
        vis_imgs.append((image_path, vis_img))
    else:
        x = np.random.randn(*input_shape).astype(np.float32)
        vis_imgs = []

    start = time.time()
    outputs = sess.run(None, {inp.name: x})
    elapsed = (time.time() - start) * 1000

    print(f"ONNX loaded from: {onnx_path}")
    print(f"Providers: {sess.get_providers()}")
    print(f"Input name/shape: {inp.name} / {input_shape}")
    for i, out in enumerate(outputs):
        print(f"Output[{i}] shape: {list(out.shape)}")
    print(f"Inference time: {elapsed:.2f} ms for batch {batch_size}")

    if vis_imgs:
        logits_batch = outputs[0]
        for i, (img_p, vis_img) in enumerate(vis_imgs):
            logits = logits_batch[i]
            probs = _softmax(logits)
            topk_idx = np.argsort(probs)[::-1][:topk]
            print(f"Top-{topk} predictions for {img_p}:")
            for idx in topk_idx:
                label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
                print(f"  {label}: {probs[idx]*100:.2f}%")
            if out_viz:
                out_base = Path(out_viz)
                if image_paths and len(vis_imgs) > 1:
                    out_base.mkdir(parents=True, exist_ok=True)
                    out_path = out_base / (Path(img_p).stem + "_pred.png")
                else:
                    out_base.parent.mkdir(parents=True, exist_ok=True)
                    out_path = out_base
                best_idx = topk_idx[0]
                best_label = (
                    labels[best_idx]
                    if labels and best_idx < len(labels)
                    else f"class_{best_idx}"
                )
                _visualize(vis_img, f"{best_label}: {probs[best_idx]*100:.2f}%", str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Test ONNX classifier export")
    parser.add_argument(
        "--onnx", type=str, default="dino_classifier.onnx", help="Path to ONNX model"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size for dummy input"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to an image for real inference"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory of images for batch inference (jpg/png/webp/bmp)",
    )
    parser.add_argument(
        "--cifar10-sample",
        action="store_true",
        help="Use a local CIFAR-10 sample (requires CIFAR-10 already downloaded to --cifar10-root)",
    )
    parser.add_argument(
        "--cifar10-batch",
        type=int,
        default=0,
        help="Use a batch of CIFAR-10 samples (count), saved to --cifar10-outdir",
    )
    parser.add_argument(
        "--cifar10-root", type=str, default="./data", help="CIFAR-10 root folder (same as training)"
    )
    parser.add_argument("--cifar10-index", type=int, default=0, help="Index of CIFAR-10 sample to use")
    parser.add_argument(
        "--cifar10-outdir",
        type=str,
        default="cifar10_samples",
        help="Where to save CIFAR-10 samples when using --cifar10-batch",
    )
    parser.add_argument(
        "--cifar10-train",
        action="store_true",
        help="Use train split for CIFAR-10 sample (default: test split)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Save visualization with top-1 label"
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="How many predictions to print"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional labels file, one label per line",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=(0.485, 0.456, 0.406),
        help="Normalization mean (R G B)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=(0.229, 0.224, 0.225),
        help="Normalization std (R G B)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CUDAExecutionProvider",
        help="Specific ORT provider, e.g., CUDAExecutionProvider; defaults to ORT priority order",
    )
    args = parser.parse_args()
    labels = _maybe_load_labels(args.labels) if args.labels else CIFAR10_LABELS
    image_path = args.image
    image_paths = None

    if image_path is None and args.cifar10_sample:
        image_path, gt_idx = _save_cifar10_sample(
            args.cifar10_root,
            args.cifar10_index,
            args.cifar10_train,
            out_path="cifar10_sample.png",
        )
        gt_label = (
            CIFAR10_LABELS[gt_idx] if gt_idx < len(CIFAR10_LABELS) else f"class_{gt_idx}"
        )
        split = "train" if args.cifar10_train else "test"
        print(f"Using CIFAR-10 {split} sample idx={args.cifar10_index}: {image_path} (GT: {gt_label})")

    if args.cifar10_batch > 0:
        image_paths, gt_labels = _save_cifar10_batch(
            args.cifar10_root,
            args.cifar10_index,
            args.cifar10_batch,
            args.cifar10_train,
            args.cifar10_outdir,
        )
        split = "train" if args.cifar10_train else "test"
        print(
            f"Using CIFAR-10 {split} samples [{args.cifar10_index}:{args.cifar10_index + len(image_paths)}),"
            f" saved to {args.cifar10_outdir}"
        )
        image_path = None

    if args.image_dir:
        image_paths = _collect_images(args.image_dir)
        if not image_paths:
            raise ValueError(f"No images found in {args.image_dir}")
        print(f"Using {len(image_paths)} images from directory: {args.image_dir}")
        image_path = None

    if args.out and image_paths and len(image_paths) > 1:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    run_inference(
        args.onnx,
        batch_size=args.batch,
        provider=args.provider,
        image_path=image_path,
        image_paths=image_paths,
        mean=args.mean,
        std=args.std,
        labels=labels,
        topk=args.topk,
        out_viz=args.out,
    )


if __name__ == "__main__":
    main()
