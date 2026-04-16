import xml.etree.ElementTree as ET
from pathlib import Path

ANNOTATIONS_DIR = Path("data/annotations")
OUTPUT_DIR = Path("data/labels")

CLASS_MAP = {
    "helmet": 0,
    "head":   1
}

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_MAP:
            continue

        xmin = int(obj.findtext("bndbox/xmin"))
        ymin = int(obj.findtext("bndbox/ymin"))
        xmax = int(obj.findtext("bndbox/xmax"))
        ymax = int(obj.findtext("bndbox/ymax"))

        cls_id = CLASS_MAP[name]
        x, y, w, h = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    out_file = output_dir / (xml_file.stem + ".txt")
    out_file.write_text("\n".join(lines))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = list(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"Converting {len(xml_files)} annotations...")

    for xml_file in xml_files:
        convert_annotation(xml_file, OUTPUT_DIR)

    print(f"Done. Labels saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()