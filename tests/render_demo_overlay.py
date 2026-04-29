"""Render OASIS demo frames with minimal walk/camera overlay at 10 fps.

Reconstructs the per-frame action vector by replaying the same load+
mask+stretch+halve+walk-insert pipeline used in oasis_inference.py.
Hardcoded paths assume this runs on the remote (via run-test.sh).
"""
import os, subprocess
import torch
from PIL import Image, ImageDraw, ImageFont

EXT_COND_DIM = 25
ACTIONS_PATH = "/tmp/sample_actions_0.one_hot_actions.pt"
FRAMES_DIR = "/tmp/oasis_video_12step_T2"
OUT_FRAMES_DIR = "/tmp/oasis_video_12step_T2/_overlay"
OUT_MP4 = "/tmp/oasis_video_12step_T2/overlay_10fps.mp4"

N_VIDEO_FRAMES = 125
WALK_LEN = 20
INSERT_AT = 60
FPS = 10


def build_action_stream():
    raw = torch.load(ACTIONS_PATH, weights_only=True).float()
    keep = torch.zeros(EXT_COND_DIM)
    keep[11] = 1.0
    keep[15] = 1.0
    keep[16] = 1.0
    raw = raw * keep
    stretched = raw.repeat_interleave(2, dim=0)
    stretched[:, 15] *= 0.5
    stretched[:, 16] *= 0.5
    walk_block = torch.zeros(WALK_LEN, EXT_COND_DIM)
    walk_block[:, 11] = 1.0
    ins = INSERT_AT - 1
    seq = torch.cat([stretched[:ins], walk_block, stretched[ins:]], dim=0)
    out = torch.zeros(N_VIDEO_FRAMES, EXT_COND_DIM)
    n_use = min(seq.shape[0], N_VIDEO_FRAMES - 1)
    out[1:1 + n_use] = seq[:n_use]
    return out


def load_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _key_box(draw, x0, y0, size, label, active, font):
    bg = (40, 200, 80, 230) if active else (0, 0, 0, 0)
    draw.rounded_rectangle((x0, y0, x0 + size, y0 + size), radius=5,
                           fill=bg, outline=(255, 255, 255, 200), width=1)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (size - tw) // 2 - bbox[0], y0 + (size - th) // 2 - bbox[1]),
              label, fill=(255, 255, 255, 255), font=font)


def _arrow_box(draw, x0, y0, size, direction, active):
    bg = (40, 200, 80, 230) if active else (0, 0, 0, 0)
    draw.rounded_rectangle((x0, y0, x0 + size, y0 + size), radius=5,
                           fill=bg, outline=(255, 255, 255, 200), width=1)
    cx = x0 + size // 2
    cy = y0 + size // 2
    s = size // 3
    if direction == "up":
        pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
    elif direction == "down":
        pts = [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)]
    elif direction == "left":
        pts = [(cx - s, cy), (cx + s, cy - s), (cx + s, cy + s)]
    else:  # right
        pts = [(cx + s, cy), (cx - s, cy - s), (cx - s, cy + s)]
    draw.polygon(pts, fill=(255, 255, 255, 240))


def draw_overlay(img, action, frame_idx, total):
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    walk = float(action[11]) > 0.5
    cam_x = float(action[15])  # -=up, +=down
    cam_y = float(action[16])  # -=left, +=right

    pad = 10
    gap = 4
    key = 28
    grid_w = 3 * key + 2 * gap
    grid_h = 2 * key + gap
    gx = pad
    gy = H - pad - grid_h
    font_k = load_font(15)
    _key_box(draw, gx + (key + gap), gy, key, "W", walk, font_k)
    _key_box(draw, gx, gy + (key + gap), key, "A", False, font_k)
    _key_box(draw, gx + (key + gap), gy + (key + gap), key, "S", False, font_k)
    _key_box(draw, gx + 2 * (key + gap), gy + (key + gap), key, "D", False, font_k)

    arrow = 24
    cluster_w = 3 * arrow + 2 * gap
    cluster_h = 3 * arrow + 2 * gap
    ax = W - pad - cluster_w
    ay = H - pad - cluster_h
    thr = 0.05
    up = cam_x < -thr
    down = cam_x > thr
    left = cam_y < -thr
    right = cam_y > thr
    _arrow_box(draw, ax + (arrow + gap), ay, arrow, "up", up)
    _arrow_box(draw, ax, ay + (arrow + gap), arrow, "left", left)
    _arrow_box(draw, ax + 2 * (arrow + gap), ay + (arrow + gap), arrow, "right", right)
    _arrow_box(draw, ax + (arrow + gap), ay + 2 * (arrow + gap), arrow, "down", down)

    font_s = load_font(12)
    label = "frame %d/%d (tt-lang, 10fps, quiet box)" % (frame_idx, total - 1)
    bbox = draw.textbbox((0, 0), label, font=font_s)
    lw = bbox[2] - bbox[0]
    lh = bbox[3] - bbox[1]
    lx = pad
    ly = gy - 4 - lh - 4
    draw.text((lx, ly - bbox[1]), label, fill=(255, 255, 255, 240), font=font_s,
              stroke_width=2, stroke_fill=(0, 0, 0, 200))
    return img


def main():
    os.makedirs(OUT_FRAMES_DIR, exist_ok=True)
    actions = build_action_stream()
    print("rendering %d frames..." % N_VIDEO_FRAMES)
    for i in range(N_VIDEO_FRAMES):
        src = os.path.join(FRAMES_DIR, "frame_%04d.png" % i)
        if not os.path.exists(src):
            print("missing: %s (stopping)" % src)
            break
        img = Image.open(src).convert("RGBA")
        img = draw_overlay(img, actions[i], i, N_VIDEO_FRAMES)
        img.convert("RGB").save(os.path.join(OUT_FRAMES_DIR, "frame_%04d.png" % i))

    cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
           "-i", os.path.join(OUT_FRAMES_DIR, "frame_%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT_MP4]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg failed:\n%s" % r.stderr[-400:])
    else:
        print("wrote %s" % OUT_MP4)


if __name__ == "__main__":
    main()
