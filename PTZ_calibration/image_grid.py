#!/usr/bin/env python3
"""
Stitch all images in the current folder that start with "frame" into a grid.

Usage:
  python3 stitch_frames_grid.py               # runs in the folder where the script lives
  python3 stitch_frames_grid.py --folder /path/to/folder --output out.png
  python3 stitch_frames_grid.py --folder . --thumb 320x240 --rows 3

Options:
  --folder PATH     Folder to scan (default: this script's folder)
  --output NAME     Output filename (default: stitched.png)
  --rows N          Force number of rows
  --cols N          Force number of cols
  --thumb WxH       Resize each image to WxH before stitching (optional)
"""

import os
import math
import re
import argparse
from PIL import Image

def natural_key(s):
    # Split strings into list of ints and text for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def collect_images(folder):
    imgs = []
    for fname in os.listdir(folder):
        if not fname.startswith("frame"):
            continue
        lower = fname.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            imgs.append(fname)
    imgs.sort(key=natural_key)
    return imgs

def parse_size(s):
    if not s:
        return None
    try:
        w,h = s.lower().split('x')
        return (int(w), int(h))
    except Exception:
        raise argparse.ArgumentTypeError("thumb must be WIDTHxHEIGHT, e.g. 320x240")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.path.dirname(__file__) or ".", help="Folder to scan for frame* images")
    parser.add_argument("--output", default="stitched.png", help="Output filename")
    parser.add_argument("--rows", type=int, default=0, help="Force number of rows")
    parser.add_argument("--cols", type=int, default=0, help="Force number of cols")
    parser.add_argument("--thumb", type=parse_size, default=None, help="Resize each image to WxH before stitching (e.g. 320x240)")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    imgs = collect_images(folder)
    if not imgs:
        print("No images starting with 'frame' found in", folder)
        return

    # Open images
    opened = []
    for fname in imgs:
        path = os.path.join(folder, fname)
        try:
            im = Image.open(path).convert("RGB")
            opened.append(im)
        except Exception as e:
            print("Skipping", fname, ":", e)

    if not opened:
        print("No valid images to stitch.")
        return

    # Optional resize (thumbnail) or normalize to smallest size
    if args.thumb:
        target_size = args.thumb
    else:
        # use smallest width/height among images to avoid upscaling
        widths = [im.width for im in opened]
        heights = [im.height for im in opened]
        target_size = (min(widths), min(heights))

    resized = [im.resize(target_size, Image.LANCZOS) for im in opened]

    n = len(resized)
    # determine grid shape
    if args.rows > 0 and args.cols > 0:
        rows, cols = args.rows, args.cols
    elif args.rows > 0:
        rows = args.rows
        cols = math.ceil(n / rows)
    elif args.cols > 0:
        cols = args.cols
        rows = math.ceil(n / cols)
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

    w, h = target_size
    canvas_w = cols * w
    canvas_h = rows * h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0,0,0))
    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x = c * w
        y = r * h
        canvas.paste(im, (x, y))

    outpath = os.path.join(folder, args.output)
    canvas.save(outpath)
    print("Saved stitched image to", outpath)

if __name__ == "__main__":
    main()