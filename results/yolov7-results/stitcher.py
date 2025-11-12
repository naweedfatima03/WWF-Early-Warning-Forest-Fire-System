import os
import argparse
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
GRID_PREFIX = 'stitched_'
GRID_COLS = 4
GRID_ROWS = 4
IMAGES_PER_GRID = GRID_COLS * GRID_ROWS

def is_image_file(name):
    _, ext = os.path.splitext(name.lower())
    return ext in IMAGE_EXTS and not name.startswith(GRID_PREFIX)

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def make_grid_image(image_paths, cols=GRID_COLS, rows=GRID_ROWS, bg_color=(0,0,0)):
    # Load images and compute cell size (max width/height among them)
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

    # Paste each image centered in its cell
    for idx, im in enumerate(imgs):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = row * cell_h

        iw, ih = im.size
        # center in cell
        paste_x = x0 + (cell_w - iw) // 2
        paste_y = y0 + (cell_h - ih) // 2
        grid.paste(im, (paste_x, paste_y))

    # close images
    for im in imgs:
        im.close()

    return grid

def process_directory(dirpath, remove_raw=True, bg_color=(0,0,0), ext='png'):
    names = sorted([n for n in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath,n))])
    img_names = [n for n in names if is_image_file(n)]
    if not img_names:
        return

    full_paths = [os.path.join(dirpath, n) for n in img_names]

    # Create grids in chunks of 16
    saved_grids = []
    for idx, chunk in enumerate(chunked(full_paths, IMAGES_PER_GRID), start=1):
        # If last chunk has fewer than required images, that's fine — blanks remain
        grid = make_grid_image(chunk, cols=GRID_COLS, rows=GRID_ROWS, bg_color=bg_color)
        # find an available filename
        i = 1
        while True:
            fname = f"{GRID_PREFIX}{idx:03d}.{ext}" if len(list(chunked(full_paths, IMAGES_PER_GRID))) > 1 else f"{GRID_PREFIX}{i:03d}.{ext}"
            outpath = os.path.join(dirpath, fname)
            if not os.path.exists(outpath):
                break
            i += 1
        grid.save(outpath)
        grid.close()
        saved_grids.append(outpath)

    # remove raw images used
    if remove_raw:
        for p in full_paths:
            try:
                os.remove(p)
            except Exception:
                pass

def walk_and_stitch(root, remove_raw=True, bg_color=(0,0,0), ext='png'):
    for dirpath, dirnames, filenames in os.walk(root):
        process_directory(dirpath, remove_raw=remove_raw, bg_color=bg_color, ext=ext)

def parse_color(s):
    parts = [int(x) for x in s.split(',')]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError("bg-color must be grayscale like '255' or 'r,g,b'")

def main():
    p = argparse.ArgumentParser(description="Stitch images into 4x4 grids and remove raw images.")
    p.add_argument('root', nargs='?', default='./detect', help='root detect directory (default: ./detect)')
    p.add_argument('--no-remove', dest='remove', action='store_false', help='do not delete raw images after stitching')
    p.add_argument('--bg-color', type=parse_color, default='0', help="background color for empty cells, e.g. '0' or '255' or '255,255,255'")
    p.add_argument('--ext', default='png', help='output image extension (png, jpg, etc.)')
    args = p.parse_args()
    walk_and_stitch(args.root, remove_raw=args.remove, bg_color=args.bg_color, ext=args.ext)

if __name__ == '__main__':
    main()