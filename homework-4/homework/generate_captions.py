import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
    image_file_for,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def generate_caption(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100
) -> list:
    """
    Generate caption for a specific view.
    """
    caps = []
    track = extract_track_info(info_path)
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)

    img_file = image_file_for(info_path, view_index)
    caps.append({"caption": f"The track is {track}.", "image_file": img_file})

    if len(karts) == 0:
        return caps

    ego = next(k for k in karts if k["is_center_kart"])
    ego_name = ego["kart_name"]
    # 1. Ego car
    caps.append({"caption": f"{ego_name} is the ego car.", "image_file": img_file})
    caps.append(
        {
            "caption": f"There are {len(karts)} karts in the scene.",
            "image_file": img_file,
        }
    )

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    for k in karts:
        kart_name = k["kart_name"]
        if kart_name != ego_name:
            center = k["center"]
            horiz = "left" if center[0] < ego["center"][0] else "right"
            vert = "in front of" if center[1] < ego["center"][1] else "behind"
            caps.append(
                {
                    "caption": f"{kart_name} is {horiz} of the ego car.",
                    "image_file": img_file,
                }
            )
            caps.append(
                {
                    "caption": f"{kart_name} is {vert} the ego car.",
                    "image_file": img_file,
                }
            )

    return caps


def expand_dataset():
    output = Path(DATA_DIR, "train", "train_captions.json")
    files = sorted((DATA_DIR / "train").glob("*_info.json"))
    rows = []
    for info_path in files:
        for view in range(10):
            rows.extend(generate_caption(str(info_path), view))

    with open(output, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} captions {output}")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": expand_dataset})


if __name__ == "__main__":
    main()
