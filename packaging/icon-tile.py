"""Render an Adobe-style app tile (a rounded square with two letters, as the Creative Cloud
application tiles look) into every icon file a craft app reads, for the owner's personal local
builds only: the tiles imitate Adobe's and Dassault's marks, so they never go into a published
build (docs/comfy/upstream.md has the brand checklist).

Usage: python -I icon-tile.py <repo-root> <name> <letters> <tile-hex> <letter-hex> [<app-id>]

  photocraft  Ps  #001E36 #31A8FF   (Photoshop)
  vectorcraft Ai  #330000 #FF9A00   (Illustrator)
  pdfcraft    Ac  #B30B00 #FFFFFF   (Acrobat)
  solidcraft  SW  #DA291C #FFFFFF   (SolidWorks)

Writes, under <repo-root>/assets/app-icon/: <name>.svg and <name>-small.svg (text-based masters),
<name>-1024.png (Apple's padded grid), <name>-macos-512.png, <name>.icns, <name>.ico (16-256 px),
hicolor/<s>x<s>/apps/<app-id>.png for 16..512 and hicolor/scalable/apps/<app-id>.svg. app-id
defaults to ai.storyteller.<name>. Geometry as packaging/icon-px.py: a 512-unit tile with rx=112,
cropped 22 units a side for Windows and Linux, padded to 636 units for macOS.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\segoeuib.ttf"
SCALE = 8  # render the 512 tile at 4096 px, then downsample


def rgba(hexcode):
    h = hexcode.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


def tile(letters, bg, fg, size_units=512, margin_units=0):
    """The tile at SCALE x, centred in a square of `size_units` with `margin_units` around it."""
    px = size_units * SCALE
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = margin_units * SCALE
    tile_px = px - 2 * m
    r = int(112 * tile_px / 512)
    d.rounded_rectangle([m, m, m + tile_px - 1, m + tile_px - 1], radius=r, fill=bg)
    # Two letters at 56 % of the tile, a little less for wide pairs such as "SW".
    font_px = int(tile_px * (0.42 if len(letters) > 1 and letters.isupper() else 0.56))
    font = ImageFont.truetype(FONT, font_px)
    bbox = d.textbbox((0, 0), letters, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = m + (tile_px - tw) / 2 - bbox[0]
    y = m + (tile_px - th) / 2 - bbox[1] - tile_px * 0.01
    d.text((x, y), letters, font=font, fill=fg)
    return img


def save(img, path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.resize((size, size), Image.LANCZOS).save(path, "PNG", optimize=True)


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    root, name, letters, bg_hex, fg_hex = sys.argv[1:6]
    app_id = sys.argv[6] if len(sys.argv) > 6 else f"ai.storyteller.{name}"
    bg, fg = rgba(bg_hex), rgba(fg_hex)
    out = os.path.join(root, "assets", "app-icon")
    full = tile(letters, bg, fg)
    c = 22 * SCALE
    tight = full.crop((c, c, full.width - c, full.height - c))
    mac = tile(letters, bg, fg, 636, 62)

    save(mac, os.path.join(out, f"{name}-1024.png"), 1024)
    save(mac, os.path.join(out, f"{name}-macos-512.png"), 512)
    for s in (16, 24, 32, 48, 64, 128, 256, 512):
        save(tight, os.path.join(out, "hicolor", f"{s}x{s}", "apps", f"{app_id}.png"), s)
    tight.resize((256, 256), Image.LANCZOS).save(
        os.path.join(out, f"{name}.ico"), format="ICO", sizes=[(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 128, 256)]
    )
    mac.resize((1024, 1024), Image.LANCZOS).save(
        os.path.join(out, f"{name}.icns"), format="ICNS", sizes=[(s, s) for s in (16, 32, 64, 128, 256, 512, 1024)]
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">\n'
        f'  <!-- {name} app icon for local builds: an Adobe-style tile with "{letters}"; rendered by\n'
        '       packaging/icon-tile.py (Segoe UI Bold). Not for a published build. -->\n'
        f'  <rect width="512" height="512" rx="112" fill="{bg_hex}"/>\n'
        '  <text x="256" y="356" text-anchor="middle" font-family="Segoe UI, Helvetica Neue, Arial, sans-serif"\n'
        f'        font-weight="700" font-size="287" fill="{fg_hex}">{letters}</text>\n'
        '</svg>\n'
    )
    for f_name in (f"{name}.svg", f"{name}-small.svg"):
        with open(os.path.join(out, f_name), "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
    scalable = os.path.join(out, "hicolor", "scalable", "apps")
    os.makedirs(scalable, exist_ok=True)
    with open(os.path.join(scalable, f"{app_id}.svg"), "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    print(f"{name}: {letters} tile written to {out}")


if __name__ == "__main__":
    main()
