
# tools/cli_count.py
import argparse, os, sys, glob, json, csv
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.pipeline import analyze_bgr

def process_one(path, outdir):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"cannot read: {path}")
    res = analyze_bgr(img)
    base = os.path.splitext(os.path.basename(path))[0]
    overlay_path = os.path.join(outdir, f"{base}_overlay.jpg")
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(overlay_path, res["overlay_bgr"])
    return {
        "file": os.path.basename(path),
        "count": res["count"],
        "centers": res["centers"],
        "overlay": overlay_path,
    }

def main():
    ap = argparse.ArgumentParser(description="Batch count grafts and save overlays/CSV")
    ap.add_argument("input", help="Image file or a glob, e.g. 'data/*.jpg'")
    ap.add_argument("-o", "--outdir", default="outputs", help="Output directory")
    ap.add_argument("--csv", default="counts.csv", help="CSV filename inside outdir")
    args = ap.parse_args()

    files = []
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = glob.glob(args.input)
        files = [f for f in files if os.path.isfile(f)]

    if not files:
        print("No input images found.")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for f in files:
        res = process_one(f, args.outdir)
        rows.append([res["file"], res["count"], res["overlay"]])
        print(f"{res['file']}: {res['count']}")

    # write CSV
    csv_path = os.path.join(args.outdir, args.csv)
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "count", "overlay"])
        writer.writerows(rows)
    print("Saved CSV:", csv_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
