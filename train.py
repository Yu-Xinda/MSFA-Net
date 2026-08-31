from pathlib import Path
from argparse import ArgumentParser
import random
import shutil
import xml.etree.ElementTree as ET

import yaml


DATASETS = {
    "neu_det": {
        "root": Path("data/NEU-DET"),
        "yaml": Path("data/NEU-DET/yolo/neu_det.yaml"),
        "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        "prepare": True,
    },
    "gc10_det": {
        "root": Path("data/GC10-DET"),
        "yaml": Path("data/GC10-DET/yolo/gc10_det.yaml"),
        "classes": [
            "punching_hole",
            "welding_line",
            "crescent_gap",
            "water_spot",
            "oil_spot",
            "silk_spot",
            "inclusion",
            "rolled_pit",
            "crease",
            "waist folding",
        ],
        "prepare": False,
    },
    "ctdd": {
        "root": Path("data/CTDD"),
        "yaml": Path("data/CTDD/ctdd.yaml"),
        "classes": ["defect"],
        "prepare": False,
    },
}
DATASET_NAMES = tuple(DATASETS)
MODE_TOKENS = ("dahl", "fafm", "hfcd")
MODES = ("dahl_fafm_hfcd",)
PRETRAIN_MODES = ("pretrained", "scratch")
MODEL_CFG_DIR = Path(__file__).resolve().parent / "ultralytics" / "cfg" / "models" / "11"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
FIXED_EXPERIMENT = {
    "model_size": "n",
    "mode": "dahl_fafm_hfcd",
    "pretrain_mode": "scratch",
    "device": "0",
    "dahl_gain": 0.65,
    "dahl_gamma": 1.25,
    "dahl_max_gain": 1.50,
    "hfcd_gain": 0.05,
    "hfcd_beta": 1.00,
    "hfcd_warmup_epochs": 10.0,
}


def model_cfg_path(model_size, cfg_name):
    size_cfg = MODEL_CFG_DIR / f"yolo11{model_size}-{cfg_name}.yaml"
    generic_cfg = MODEL_CFG_DIR / f"yolo11-{cfg_name}.yaml"
    # Ultralytics will load the generic YAML from a virtual size-specific path and preserve the scale suffix.
    return size_cfg if size_cfg.exists() or generic_cfg.exists() else generic_cfg


def mode_features(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown MODE '{mode}'. Valid modes: {', '.join(MODES)}")
    features = mode.split("_")
    unknown = sorted(set(features) - set(MODE_TOKENS))
    if unknown:
        raise ValueError(f"Unknown MODE token(s): {', '.join(unknown)}")
    return set(features)


def arch_cfg_name(features):
    if "fafm" in features:
        return "fafm"
    return None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "neu_det.yaml")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--print-run-name", action="store_true", help="print the resolved run directory name and exit")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    for name in (
        "save_dir",
        "dataset",
        "data_root",
        "data_yaml",
    ):
        if getattr(args, name, None) is None and name in config:
            setattr(args, name, config[name])

    for name, value in FIXED_EXPERIMENT.items():
        setattr(args, name, value)

    args.save_dir = Path(args.save_dir)
    args.data_root = Path(args.data_root) if args.data_root else None
    args.data_yaml = Path(args.data_yaml) if args.data_yaml else None
    if args.dataset not in DATASET_NAMES:
        parser.error(f"dataset must be one of: {', '.join(DATASET_NAMES)}")
    if args.mode not in MODES:
        parser.error(f"mode must be: {MODES[0]}")
    if args.pretrain_mode not in PRETRAIN_MODES:
        parser.error(f"pretrain-mode must be one of: {', '.join(PRETRAIN_MODES)}")
    try:
        mode_features(args.mode)
    except ValueError as e:
        parser.error(str(e))
    return args


def parse_xml(xml_file, classes):
    root = ET.parse(xml_file).getroot()
    w, h = float(root.findtext("size/width")), float(root.findtext("size/height"))
    filename = root.findtext("filename") or f"{xml_file.stem}.jpg"
    if not Path(filename).suffix:
        filename = f"{filename}.jpg"
    labels = []

    for obj in root.findall("object"):
        cls = obj.findtext("name")
        box = obj.find("bndbox")
        if cls not in classes or box is None:
            continue

        xmin = max(0.0, min(float(box.findtext("xmin")), w))
        ymin = max(0.0, min(float(box.findtext("ymin")), h))
        xmax = max(0.0, min(float(box.findtext("xmax")), w))
        ymax = max(0.0, min(float(box.findtext("ymax")), h))
        if xmax <= xmin or ymax <= ymin:
            continue

        x = ((xmin + xmax) / 2) / w
        y = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        labels.append(f"{classes.index(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    return filename, labels


def link_or_copy(src, dst):
    if dst.exists():
        return
    if dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_dataset(dataset="neu_det", val_ratio=0.2, seed=42, data_root=None, data_yaml=None):
    dataset_cfg = DATASETS[dataset]
    data_yaml = Path(data_yaml or dataset_cfg["yaml"])
    if not dataset_cfg["prepare"]:
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
        return data_yaml

    data_root = Path(data_root or dataset_cfg["root"])
    out_dir = data_yaml.parent
    classes = dataset_cfg["classes"]
    xml_files = sorted((data_root / "ANNOTATIONS").glob("*.xml"))
    random.Random(seed).shuffle(xml_files)
    split_idx = int(len(xml_files) * (1 - val_ratio))
    splits = {"train": xml_files[:split_idx], "val": xml_files[split_idx:]}

    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, files in splits.items():
        image_list = []
        for xml_file in files:
            filename, labels = parse_xml(xml_file, classes)
            src_img = data_root / "IMAGES" / filename
            dst_img = out_dir / "images" / split / filename
            dst_label = out_dir / "labels" / split / f"{Path(filename).stem}.txt"

            link_or_copy(src_img, dst_img)
            dst_label.write_text("\n".join(labels) + ("\n" if labels else ""))
            image_list.append(str(dst_img))

        (out_dir / f"{split}.txt").write_text("\n".join(image_list) + "\n")

    data_yaml.write_text(
        f"path: {out_dir}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(classes))
    )
    return data_yaml


def build_model(model_size, mode, pretrain_mode):
    from ultralytics import YOLO

    cfg_name = arch_cfg_name(mode_features(mode))
    if cfg_name:
        return YOLO(str(model_cfg_path(model_size, cfg_name)))
    if pretrain_mode == "scratch":
        return YOLO(str(MODEL_CFG_DIR / f"yolo11{model_size}.yaml"))
    return YOLO(f"yolo11{model_size}.pt")


def contribution_overrides(args):
    features = mode_features(args.mode)
    return {
        "dahl_loss": "dahl" in features,
        "dahl_gain": args.dahl_gain,
        "dahl_gamma": args.dahl_gamma,
        "dahl_max_gain": args.dahl_max_gain,
        "hfcd_loss": "hfcd" in features,
        "hfcd_gain": args.hfcd_gain,
        "hfcd_beta": args.hfcd_beta,
        "hfcd_warmup_epochs": args.hfcd_warmup_epochs,
    }


def hparam_token(value):
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def run_name(
    dataset,
    model_size,
    mode,
    pretrain_mode,
    dahl_gain=None,
    dahl_gamma=None,
    dahl_max_gain=None,
    hfcd_gain=None,
    hfcd_beta=None,
    hfcd_warmup_epochs=None,
):
    features = mode_features(mode)
    name = f"{dataset}_yolo11{model_size}_{mode}"
    if "dahl" in features and None not in (dahl_gain, dahl_gamma, dahl_max_gain):
        name = "_".join(
            [
                name,
                f"dg{hparam_token(dahl_gain)}",
                f"dgm{hparam_token(dahl_gamma)}",
                f"dmax{hparam_token(dahl_max_gain)}",
            ]
        )
    if "hfcd" in features and None not in (hfcd_gain, hfcd_beta, hfcd_warmup_epochs):
        name = "_".join(
            [
                name,
                f"hg{hparam_token(hfcd_gain)}",
                f"hb{hparam_token(hfcd_beta)}",
                f"hw{hparam_token(hfcd_warmup_epochs)}",
            ]
        )
    return f"{name}_scratch" if pretrain_mode == "scratch" else name


def run_name_from_args(args):
    if getattr(args, "config", None) is not None:
        return Path(args.config).stem
    return run_name(
        args.dataset,
        args.model_size,
        args.mode,
        args.pretrain_mode,
        dahl_gain=args.dahl_gain,
        dahl_gamma=args.dahl_gamma,
        dahl_max_gain=args.dahl_max_gain,
        hfcd_gain=args.hfcd_gain,
        hfcd_beta=args.hfcd_beta,
        hfcd_warmup_epochs=args.hfcd_warmup_epochs,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.print_run_name:
        print(run_name_from_args(args))
        raise SystemExit(0)
    data_yaml = prepare_dataset(args.dataset, data_root=args.data_root, data_yaml=args.data_yaml)
    model = build_model(args.model_size, args.mode, args.pretrain_mode)
    name = args.config.stem
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        project=str(args.save_dir.resolve()),
        name=name,
        device=args.device,
        exist_ok=True,
        **contribution_overrides(args),
    )
    if args.pretrain_mode == "scratch":
        train_kwargs["pretrained"] = False
    elif arch_cfg_name(mode_features(args.mode)):
        train_kwargs["pretrained"] = f"yolo11{args.model_size}.pt"
    model.train(**train_kwargs)
