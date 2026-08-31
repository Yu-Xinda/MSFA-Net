from pathlib import Path
from argparse import ArgumentParser

from train import DATASET_NAMES, FIXED_EXPERIMENT, prepare_dataset


IMGSZ = 640
BATCH = 16
SPLIT = "val"
CONF = 0.25
PRED_DIR_NAME = f"predictions_{SPLIT}"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("weights", nargs="?", type=Path)
    parser.add_argument("--save-dir", type=Path, default=Path("output-n"))
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="neu_det")
    parser.add_argument("--device", default=FIXED_EXPERIMENT["device"])
    args = parser.parse_args()
    try:
        mode_features(args.mode)
    except ValueError as e:
        parser.error(str(e))
    return args


def default_weights(args):
    return args.save_dir / run_name_from_args(args) / "weights" / "best.pt"


def f1_score(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def table(headers, rows):
    rows = [[str(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(x)) for w, x in zip(widths, row)]

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt.format(*headers), sep, *[fmt.format(*row) for row in rows]])


def class_items(names):
    if isinstance(names, dict):
        return [(int(k), v) for k, v in sorted(names.items())]
    return list(enumerate(names))


def dataset_split_source(data_yaml, split):
    values = {}
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            continue
        values[key.strip()] = value.strip().strip("'\"")

    if split not in values:
        raise KeyError(f"Split '{split}' not found in dataset yaml: {data_yaml}")

    dataset_root = Path(values.get("path") or data_yaml.parent)
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    source = Path(values[split])
    if not source.is_absolute():
        source = dataset_root / source
    return source


def save_predictions(model, weights, data_yaml, device):
    source = dataset_split_source(data_yaml, SPLIT)
    if not source.exists():
        raise FileNotFoundError(f"Prediction source not found for split '{SPLIT}': {source}")

    for _ in model.predict(
        source=str(source),
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONF,
        save=True,
        save_txt=True,
        save_conf=True,
        show_boxes=True,
        show_labels=True,
        show_conf=True,
        project=str(weights.parent),
        name=PRED_DIR_NAME,
        device=device,
        exist_ok=True,
        verbose=False,
        stream=True,
    ):
        pass

    return weights.parent / PRED_DIR_NAME


def main():
    args = parse_args()

    data_yaml = prepare_dataset(args.dataset)

    weights = (args.weights or default_weights(args)).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weight file not found: {weights}")

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=SPLIT, imgsz=IMGSZ, batch=BATCH, device=args.device, plots=False)

    box = metrics.box
    params = sum(p.numel() for p in model.model.parameters())
    gflops = get_flops(model.model, imgsz=IMGSZ)
    if not gflops:
        gflops = get_flops_with_torch_profiler(model.model, imgsz=IMGSZ)
    if not gflops:
        print("Warning: GFLOPs calculation returned 0.00; install ultralytics-thop in the evaluation environment.")
    speed = getattr(metrics, "speed", {})
    ms_per_image = speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)
    fps = 1000 / ms_per_image if ms_per_image else 0.0

    total_rows = [
        [
            "all",
            f"{box.mp:.4f}",
            f"{box.mr:.4f}",
            f"{f1_score(box.mp, box.mr):.4f}",
            f"{box.map50:.4f}",
            f"{box.map:.4f}",
            f"{params}",
            f"{gflops:.2f}",
            f"{fps:.2f}",
        ]
    ]

    names = metrics.names
    result_idx = {int(cls_id): i for i, cls_id in enumerate(box.ap_class_index)}
    class_rows = []
    for cls_id, name in class_items(names):
        i = result_idx.get(int(cls_id))
        p, r, map50, map5095 = box.class_result(i) if i is not None else (0.0, 0.0, 0.0, 0.0)
        class_rows.append(
            [
                name,
                f"{map50:.4f}",
                f"{map5095:.4f}",
                f"{p:.4f}",
                f"{r:.4f}",
                f"{f1_score(p, r):.4f}",
            ]
        )

    text = "\n\n".join(
        [
            f"weights: {weights}",
            f"data: {data_yaml}",
            f"split: {SPLIT}",
            "Total Metrics",
            table(["Class", "Prec", "Recall", "F1", "mAP50", "mAP50:95", "Params", "GFLOPs", "FPS"], total_rows),
            "Class Metrics",
            table(["Class", "mAP50", "mAP50:95", "Prec", "Recall", "F1"], class_rows),
        ]
    )

    save_path = weights.parent / "results.txt"
    save_path.write_text(text + "\n", encoding="utf-8")

    pred_dir = save_predictions(model, weights, data_yaml, args.device)
    with save_path.open("a", encoding="utf-8") as f:
        f.write(f"\npredictions: {pred_dir}\n")

    print(f"Results saved to {save_path}")
    print(f"Predictions saved to {pred_dir}")


if __name__ == "__main__":
    main()
