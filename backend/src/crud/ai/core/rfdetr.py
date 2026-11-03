import argparse
from pathlib import Path

from rfdetr import (
    RFDETRLarge,
    RFDETRMedium,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegSmall,
    RFDETRSmall,
)

DETECT = {
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}
SEGMENT = {
    "small": RFDETRSegSmall,
    "medium": RFDETRSegMedium,
    "large": RFDETRSegLarge,
}
TOOLS = {"bbox": DETECT, "polygon": SEGMENT, "brush": SEGMENT}


def train(
    dataset_dir: str | Path,
    output_dir: str | Path = "backend/data/models/rfdetr",
    size: str = "small",
    tool: str = "bbox",
    epochs: int = 100,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
) -> None:
    dataset = Path(dataset_dir)
    annotations = dataset / "train" / "_annotations.coco.json"
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO 학습 라벨이 없습니다: {annotations}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    models = TOOLS.get(tool)
    if models is None:
        raise ValueError(f"라벨 도구는 {', '.join(TOOLS)} 중에서 고르세요.")
    model_cls = models.get(size)
    if model_cls is None:
        raise ValueError(f"모델 크기는 {', '.join(models)} 중에서 고르세요.")
    model = model_cls()
    model.train(
        dataset_dir=str(dataset),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--size", choices=tuple(DETECT), default="small")
    parser.add_argument("--tool", choices=tuple(TOOLS), default="bbox")
    parser.add_argument("--output-dir", default="backend/data/models/rfdetr")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        size=args.size,
        tool=args.tool,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
