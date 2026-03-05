from passes import dict_to_pass, PassBase

import argparse
import json
from typing import List, Dict

import numpy as np
import colour
from tqdm.auto import tqdm


def split_passes(pass_dicts: List[Dict]):
    passes = []
    current_group = []

    for pass_dict in pass_dicts:
        film_pass = dict_to_pass(pass_dict)
        if film_pass.is_global:
            if len(current_group) > 0:
                passes.append(current_group)
                current_group = []
            passes.append(film_pass)
        else:
            current_group.append(film_pass)
    if len(current_group) > 0:
        passes.append(current_group)

    return passes


def main():
    parser = argparse.ArgumentParser(prog="film_lut_gen.py")
    parser.add_argument("config_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config = json.load(fp)

        passes = split_passes(config["passes"])

        domain = np.array(config["input"]["domain"])
        size = config["input"]["size"]
        table = colour.LUT3D.linear_table(size, domain)

        # def job(indices):
        #     v = table[*indices]
        #     for pass_dict in config["passes"]:
        #         v = dict_to_pass(pass_dict).forward(v)
        #     assert isinstance(v, np.ndarray) and v.shape == (3,)
        #     table[*indices] = v

        # Parallel(n_jobs=-1)(delayed(job)(indices) for indices in np.ndindex(table.shape[:-1]))

        for item in tqdm(passes):
            if issubclass(type(item), PassBase):
                table = item.forward(table)
            else:
                for indices in np.ndindex(table.shape[:-1]):
                    v = table[*indices]
                    for film_pass in item:
                        v = film_pass.forward(v)
                    assert isinstance(v, np.ndarray) and v.shape == (3,)
                    table[*indices] = v

        lut = colour.LUT3D(table, domain=domain)
        colour.write_LUT(lut, args.output_path)


if __name__ == "__main__":
    main()
