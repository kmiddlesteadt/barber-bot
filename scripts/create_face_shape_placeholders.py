from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "src" / "training" / "face_shapes"
FACE_SHAPES = ["Oval", "Square", "Round", "Heart", "Diamond"]
IMAGES_PER_CLASS = 12
SIZE = 512


SKIN_TONES = [
    (246, 206, 172),
    (232, 184, 137),
    (205, 142, 93),
    (156, 95, 62),
    (108, 68, 48),
]

BACKGROUNDS = [
    (238, 242, 255),
    (240, 253, 244),
    (255, 247, 237),
    (245, 243, 255),
    (236, 253, 245),
]

HAIR_COLORS = [
    (30, 24, 22),
    (62, 43, 31),
    (93, 64, 42),
    (152, 104, 53),
]


def face_points(shape, cx, cy, width, height):
    points = []
    for i in range(96):
        t = 2 * math.pi * i / 96
        x_scale = 1.0
        y_scale = 1.0

        if shape == "Oval":
            x_scale = 0.82 + 0.08 * math.cos(t)
        elif shape == "Square":
            x_scale = 0.92 if abs(math.sin(t)) < 0.72 else 0.78
            y_scale = 0.95 if math.sin(t) > -0.55 else 0.82
        elif shape == "Round":
            x_scale = 0.95
            y_scale = 0.88
        elif shape == "Heart":
            x_scale = 0.98 - 0.30 * max(math.sin(t), 0)
            y_scale = 0.94 + 0.05 * max(math.sin(t), 0)
        elif shape == "Diamond":
            x_scale = 0.58 + 0.38 * abs(math.sin(t))
            y_scale = 0.98

        # Make the chin slightly narrower for all but round faces.
        if math.sin(t) < -0.35 and shape != "Round":
            x_scale *= 0.78 + 0.18 * (math.sin(t) + 1)

        x = cx + math.cos(t) * width * x_scale / 2
        y = cy + math.sin(t) * height * y_scale / 2
        points.append((x, y))
    return points


def draw_face(draw, shape, rng):
    cx = SIZE / 2 + rng.randint(-12, 12)
    cy = SIZE / 2 + rng.randint(-8, 14)
    width = rng.randint(240, 278)
    height = rng.randint(318, 354)
    skin = rng.choice(SKIN_TONES)
    hair = rng.choice(HAIR_COLORS)

    pts = face_points(shape, cx, cy, width, height)
    draw.polygon(pts, fill=skin, outline=(70, 60, 55))

    hair_top = cy - height * 0.47
    hair_box = [
        cx - width * 0.46,
        hair_top - rng.randint(20, 40),
        cx + width * 0.46,
        cy - height * 0.18,
    ]
    draw.rounded_rectangle(hair_box, radius=58, fill=hair)

    if shape == "Heart":
        draw.polygon(
            [
                (cx - width * 0.17, cy - height * 0.43),
                (cx, cy - height * 0.33),
                (cx + width * 0.17, cy - height * 0.43),
            ],
            fill=skin,
        )

    eye_y = cy - height * 0.07
    eye_gap = width * 0.18
    for side in [-1, 1]:
        ex = cx + side * eye_gap
        draw.ellipse([ex - 14, eye_y - 7, ex + 14, eye_y + 7], fill=(255, 255, 255))
        draw.ellipse([ex - 5, eye_y - 5, ex + 5, eye_y + 5], fill=(35, 35, 35))
        draw.arc([ex - 20, eye_y - 26, ex + 20, eye_y - 6], 190, 350, fill=hair, width=3)

    nose_y = cy + height * 0.09
    draw.line([(cx, eye_y + 24), (cx - 10, nose_y), (cx + 8, nose_y + 6)], fill=(120, 85, 68), width=3)

    mouth_y = cy + height * 0.24
    draw.arc([cx - 46, mouth_y - 22, cx + 46, mouth_y + 20], 18, 162, fill=(128, 46, 58), width=4)

    # Subtle class cue: outline the jaw and cheek area a little more strongly.
    if shape in {"Square", "Diamond"}:
        draw.line(pts[52:76], fill=(70, 60, 55), width=4)


def add_background(draw, rng):
    for _ in range(24):
        x = rng.randint(0, SIZE)
        y = rng.randint(0, SIZE)
        radius = rng.randint(12, 42)
        color = (255, 255, 255, rng.randint(30, 75))
        layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        yield layer


def create_image(shape, index):
    rng = random.Random(f"{shape}-{index}")
    base = Image.new("RGBA", (SIZE, SIZE), rng.choice(BACKGROUNDS) + (255,))
    for layer in add_background(ImageDraw.Draw(base), rng):
        base = Image.alpha_composite(base, layer)

    draw = ImageDraw.Draw(base)
    draw_face(draw, shape, rng)
    base = base.filter(ImageFilter.SMOOTH_MORE)
    return base.convert("RGB")


def main():
    for shape in FACE_SHAPES:
        shape_dir = OUTPUT_DIR / shape
        shape_dir.mkdir(parents=True, exist_ok=True)

        for index in range(1, IMAGES_PER_CLASS + 1):
            path = shape_dir / f"{shape.lower()}_{index:02d}.jpg"
            if path.exists():
                continue
            image = create_image(shape, index)
            image.save(path, quality=92)

    print(f"Created placeholder face-shape images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
