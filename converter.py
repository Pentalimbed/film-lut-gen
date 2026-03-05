import json
import os

import numpy as np

ref_prefix = r"ref/ektachrome/100ec"
out_path = r"data/ektachrome-100ec"


def main():
    os.makedirs(out_path, exist_ok=True)

    for info in ["char", "sen", "dye-spec"]:
        with open(f"{ref_prefix}-{info}.json") as fp:
            cdict = json.load(fp)

            for curve in cdict:
                points = [[p["x"], p["y"]] for p in curve["points"]]
                points.sort()
                points = np.array(points).T
                if info == "info":
                    points = np.maximum(points, 0)
                np.save(os.path.join(out_path, f'{info}-{curve["label"]}'), points)


if __name__ == "__main__":
    main()
