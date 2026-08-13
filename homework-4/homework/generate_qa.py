import glob
import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400

DATA_DIR = Path(__file__).parent.parent / "data"


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str,
    info_path: str,
    font_scale: float = 0.5,
    thickness: int = 1,
    min_box_size: int = 5,
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (
            y2_scaled - y1_scaled
        ) < min_box_size:
            continue

        if (
            x2_scaled < 0
            or x1_scaled > img_width
            or y2_scaled < 0
            or y1_scaled > img_height
        ):
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle(
            [(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)],
            outline=color,
            width=thickness,
        )

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
    min_box_size: int = 5,
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    result = []
    with open(info_path, "r", encoding="utf-8") as file:
        info = json.load(file)

    dets = info["detections"][view_index]
    karts = info["karts"]

    cx, cy = img_width / 2, img_height / 2
    for det in dets:
        class_id, track_id, x1, y1, x2, y2 = det
        # skip detections that are not karts
        if class_id != 1:
            continue

        sx1 = x1 * (img_width / ORIGINAL_WIDTH)
        sx2 = x2 * (img_width / ORIGINAL_WIDTH)
        sy1 = y1 * (img_height / ORIGINAL_HEIGHT)
        sy2 = y2 * (img_height / ORIGINAL_HEIGHT)

        # skip detections that are outside of the boundaries
        if sx1 >= img_width or sy1 >= img_height or sx2 <= 0 or sy2 <= 0:
            continue

        width = sx2 - sx1
        height = sy2 - sy1

        # skip detections that are not visible
        if width < min_box_size or height < min_box_size:
            continue

        center_x = sx1 + width / 2
        center_y = sy1 + height / 2

        kart = {
            "instance_id": track_id,
            "kart_name": karts[track_id],
            "center": (center_x, center_y),
            "is_center_kart": False,
        }

        result.append(kart)

    if result:
        ego = min(
            result,
            key=lambda k: (k["center"][0] - cx) ** 2 + (k["center"][1] - cy) ** 2,
        )
        ego["is_center_kart"] = True

    return result


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path, "r", encoding="utf-8") as file:
        info = json.load(file)

    return info["track"]


def generate_qa_pairs(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100
) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa = []
    track = extract_track_info(info_path)
    karts = extract_kart_objects(
        info_path, view_index, img_width, img_height, min_box_size=5
    )
    image_file = image_file_for(info_path, view_index)

    # 3. Track information questions
    # What track is this?
    qa.append(
        {"question": "What track is this?", "answer": track, "image_file": image_file}
    )

    if len(karts) == 0:
        return qa

    ego = next(k for k in karts if k["is_center_kart"])
    ego_name = ego["kart_name"]

    # 1. Ego car question
    # What kart is the ego car?
    qa.append(
        {
            "question": "What kart is the ego car?",
            "answer": ego_name,
            "image_file": image_file,
        }
    )

    # 2. Total karts question
    # How many karts are there in the scenario?
    qa.append(
        {
            "question": "How many karts are there in the scenario?",
            "answer": str(len(karts)),
            "image_file": image_file,
        }
    )

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?
    karts_left = 0
    karts_right = 0
    karts_front = 0
    karts_back = 0
    for k in karts:
        kart_name = k["kart_name"]
        center = k["center"]
        if kart_name != ego_name:
            horiz = "left" if center[0] < ego["center"][0] else "right"
            vert = "front" if center[1] < ego["center"][1] else "back"

            # update tallies
            if horiz == "left":
                karts_left += 1
            else:
                karts_right += 1
            if vert == "front":
                karts_front += 1
            else:
                karts_back += 1

            qa.append(
                {
                    "question": f"Is {kart_name} to the left or right of the ego car?",
                    "answer": horiz,
                    "image_file": image_file,
                }
            )
            qa.append(
                {
                    "question": f"Is {kart_name} in front of or behind the ego car?",
                    "answer": vert,
                    "image_file": image_file,
                }
            )
            qa.append(
                {
                    "question": f"Where is {kart_name} relative to the ego car?",
                    "answer": f"{vert} and {horiz}",
                    "image_file": image_file,
                }
            )

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    if karts_left > 0:
        qa.append(
            {
                "question": "How many karts are to the left of the ego car?",
                "answer": str(karts_left),
                "image_file": image_file,
            }
        )
    if karts_right > 0:
        qa.append(
            {
                "question": "How many karts are to the right of the ego car?",
                "answer": str(karts_right),
                "image_file": image_file,
            }
        )
    if karts_front > 0:
        qa.append(
            {
                "question": "How many karts are in front of the ego car?",
                "answer": str(karts_front),
                "image_file": image_file,
            }
        )
    if karts_back > 0:
        qa.append(
            {
                "question": "How many karts are behind the ego car?",
                "answer": str(karts_back),
                "image_file": image_file,
            }
        )

    return qa


def expand_dataset():
    output = Path(DATA_DIR, "train", "train_qa_pairs.json")
    files = sorted((DATA_DIR / "train").glob("*_info.json"))
    rows = []
    for info_path in files:
        for view in range(10):
            rows.extend(generate_qa_pairs(str(info_path), view))

    with open(output, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} QA pairs to {output}")


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def image_file_for(info_path, view_index) -> str:
    _info_path = Path(info_path)
    base_name = _info_path.stem.replace("_info", "")
    return f"{_info_path.parent.name}/{base_name}_{view_index:02d}_im.jpg"

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate": expand_dataset})


if __name__ == "__main__":
    main()
