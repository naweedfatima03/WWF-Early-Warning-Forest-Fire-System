import os
import argparse
from PIL import Image
import fnmatch

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
GRID_PREFIX = 'stitched_curves_'
GRID_COLS = 2
GRID_ROWS = 2
IMAGES_PER_GRID = GRID_COLS * GRID_ROWS
PATTERN = '*_curve.png'

def is_curve_file(name):
    return fnmatch.fnmatch(name.lower(), PATTERN) and not name.startswith(GRID_PREFIX)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def make_grid_image(image_paths, cols=GRID_COLS, rows=GRID_ROWS, bg_color=(0,0,0)):
    imgs = []
    widths = []
    heights = []
    for p in image_paths:
        im = Image.open(p).convert('RGB')
        imgs.append(im)
        w,h = im.size
        widths.append(w)
        heights.append(h)

    cell_w = max(widths) if widths else 0
    cell_h = max(heights) if heights else 0

    grid_w = cell_w * cols
    grid_h = cell_h * rows
    grid = Image.new('RGB', (grid_w, grid_h), color=bg_color)

    for idx, im in enumerate(imgs):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = row * cell_h
        iw, ih = im.size
        paste_x = x0 + (cell_w - iw) // 2
        paste_y = y0 + (cell_h - ih) // 2
        grid.paste(im, (paste_x, paste_y))

    for im in imgs:
        im.close()

    return grid

def process_directory(dirpath, bg_color=(0,0,0), ext='png'):
    names = sorted([n for n in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath,n))])
    img_names = [n for n in names if is_curve_file(n)]
    if not img_names:
        return

    full_paths = [os.path.join(dirpath, n) for n in img_names]

    for idx, chunk in enumerate(chunked(full_paths, IMAGES_PER_GRID), start=1):
        grid = make_grid_image(chunk, cols=GRID_COLS, rows=GRID_ROWS, bg_color=bg_color)
        outname = f"{GRID_PREFIX}{idx:03d}.{ext}"
        outpath = os.path.join(dirpath, outname)
        # avoid overwriting existing file by incrementing suffix if necessary
        i = 1
        base, _ = os.path.splitext(outname)
        while os.path.exists(outpath):
            outpath = os.path.join(dirpath, f"{base}_{i:02d}.{ext}")
            i += 1
        grid.save(outpath)
        grid.close()

def walk_and_stitch(root, bg_color=(0,0,0), ext='png'):
    for dirpath, dirnames, filenames in os.walk(root):
        process_directory(dirpath, bg_color=bg_color, ext=ext)

def parse_color(s):
    parts = [int(x) for x in s.split(',')]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError("bg-color must be 'v' or 'r,g,b'")

def main():
    p = argparse.ArgumentParser(description="Stitch *_curve.png files into 4x4 grids (does not delete raw files).")
    p.add_argument('root', nargs='?', default='./train', help='root train directory (default: ./train)')
    p.add_argument('--bg-color', type=parse_color, default='0', help="background color for empty cells, e.g. '0' or '255,255,255'")
    p.add_argument('--ext', default='png', help='output image extension (png, jpg, etc.)')
    args = p.parse_args()
    walk_and_stitch(args.root, bg_color=args.bg_color, ext=args.ext)

if __name__ == '__main__':
    main()