import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

emojis = {
    "grinning": "😀",
    "crying": "😢",
    "angry": "😠",
    "sunglasses": "😎",
    "neutral": "😐",
    "sleeping": "😴",
    "heart_eyes": "😍",
    "joy": "😂",
    "wink": "😉",
    "blush": "😊",
    "thinking": "🤔",
    "pensive": "😔",
    "scream": "😱",
    "zipper_mouth": "🤐",
    "astonished": "😲",
    "kissing": "😗",
    "disappointed": "😞",
    "worried": "😟",
    "confused": "😕",
    "frowning": "😦",
    "sad": "😢",
    "crying-a-lot": "😭",
    "confounded": "😖",
    "persevere": "😣",
    "tired_face": "😫",
    "weary": "😩",
    "triumph": "😤",
    "astonished": "😲",
    "flushed": "😳",
    "relieved": "😌",
    "satisfied": "😌",
    "frowning": "😦",
    "fearful": "😨",
    "grinning_cat": "😺",
    "grinning_cat_with_smiling_eyes": "😸",
    "cat_with_tears_of_joy": "😹",
    "smiling_cat_with_heart_eyes": "😻",
}


def find_emoji_font():
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux (Ubuntu)
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttc",  # Linux (Arch)
        "/usr/share/fonts/noto/NotoColorEmoji.ttf",  # Linux (generic)
        "/usr/share/fonts/TTF/NotoColorEmoji.ttf",  # Linux (Arch alternative)
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # MacOS
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows
    ]

    for path in font_paths:
        try:
            for size in [32, 28, 24, 20, 16]:
                try:
                    font = ImageFont.truetype(path, size)
                    print(f"Font loaded successfully: {path} with size {size}")
                    return font
                except OSError as e:
                    if "invalid pixel size" in str(e):
                        continue
                    else:
                        raise e
        except Exception as e:
            print(f"Error loading font {path}: {e}")
            continue

    print("Using default font")
    return ImageFont.load_default()


def create_emoji_dataset(size=20, save_path="emoji_dataset.pkl"):
    emoji_font = find_emoji_font()
    emoji_arrays = {}

    print(f"Creating {size}x{size} emoji dataset...")

    for name, char in emojis.items():
        large_size = size * 4
        large_img = Image.new("L", (large_size, large_size), color=255)
        draw = ImageDraw.Draw(large_img)
        draw.text((large_size / 2, large_size / 2), char, font=emoji_font, fill=0, anchor="mm")

        inverted_img = ImageOps.invert(large_img)
        bbox = inverted_img.getbbox()

        if bbox:
            cropped_img = large_img.crop(bbox)
            final_img = cropped_img.resize((size, size), Image.Resampling.LANCZOS)
        else:
            print(f"Warning: Could not find rendered pixels for emoji '{name}'. Creating a blank image.")
            final_img = Image.new("L", (size, size), color=255)

        arr = np.array(final_img) / 255.0
        emoji_arrays[name] = arr

        print(f"Created {name}: {char}")

    dataset = {
        "emojis": emoji_arrays,
        "size": size,
        "names": list(emojis.keys()),
        "characters": list(emojis.values()),
    }

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {save_path}")
    return dataset


def load_emoji_dataset(load_path="assets/emoji_dataset.pkl"):
    if not os.path.exists(load_path):
        print(f"Dataset file {load_path} not found. Creating new dataset...")
        return create_emoji_dataset(save_path=load_path, size=32)

    with open(load_path, "rb") as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded from {load_path}")
    return dataset


def plot_emojis(dataset):
    emoji_arrays = dataset["emojis"]
    names = dataset["names"]

    n_emojis = len(names)
    cols = 5
    rows = (n_emojis + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axs = axs.flatten()

    for i, name in enumerate(names):
        ax = axs[i]
        ax.imshow(emoji_arrays[name], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

    for i in range(n_emojis, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


def get_emoji_data(dataset, name):
    return dataset["emojis"].get(name)


def get_all_emoji_data(dataset):
    return list(dataset["emojis"].values())


def get_emoji_names(dataset):
    return dataset["names"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and display an emoji dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of the dataset by deleting the existing .pkl file.",
    )
    args = parser.parse_args()

    dataset_path = "assets/emoji_dataset.pkl"

    if args.force and os.path.exists(dataset_path):
        print(f"Force flag set. Deleting existing dataset at {dataset_path}...")
        os.remove(dataset_path)

    dataset = load_emoji_dataset(dataset_path)

    plot_emojis(dataset)

    print("\nDataset Info:")
    print(f"Size: {dataset['size']}x{dataset['size']} pixels")
    print(f"Number of emojis: {len(dataset['names'])}")
    print(f"Emoji names: {dataset['names']}")
