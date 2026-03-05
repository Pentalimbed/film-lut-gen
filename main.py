from passes import dict_to_pass

import os
import argparse
import json

import numpy as np
import colour
from tqdm.auto import tqdm
from joblib import delayed, Parallel
from tqdm.contrib.concurrent import process_map


def main():
    parser = argparse.ArgumentParser(prog="film_lut_gen.py")
    parser.add_argument("config_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config = json.load(fp)

        domain = np.array(config["input"]["domain"])
        size = config["input"]["size"]
        table = colour.LUT3D.linear_table(size, domain)

        def job(indices):
            v = table[*indices]
            for pass_dict in config["passes"]:
                v = dict_to_pass(pass_dict).forward(v)
            assert isinstance(v, np.ndarray) and v.shape == (3,)
            table[*indices] = v

        # Parallel(n_jobs=-1)(delayed(job)(indices) for indices in np.ndindex(table.shape[:-1]))

        for indices in tqdm(
            np.ndindex(table.shape[:-1]),
            desc="Generating LUT...",
            total=np.prod(table.shape[:-1], dtype=int),
        ):
            v = table[*indices]
            for pass_dict in config["passes"]:
                v = dict_to_pass(pass_dict).forward(v)
            assert isinstance(v, np.ndarray) and v.shape == (3,)
            table[*indices] = v

        lut = colour.LUT3D(table, domain=domain)
        colour.write_LUT(lut, args.output_path)


if __name__ == "__main__":
    main()
