import os
from functools import reduce
import glob2
import pandas as pd


def clean_grbs(main_dir):

    # grab all filepaths for LCs in magnitude
    glob_path = reduce(os.path.join, [main_dir, "*_flux", "*_converted_flux.txt"])
    filepaths = glob2.glob(glob_path)

    num_dupes = 0
    num_successful = 0
    num_points = 0
    num_dupe_points = 0
    unphysical_times = {"grb": [], "filepath": []}
    for filepath in filepaths:
        grb = os.path.split(filepath)[-1].rstrip("_flux.txt")
        df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0).copy()

        # get rid of duplicate times! this keeps the first instance of the duplicate
        duplicates = df.duplicated("time_sec")
        # df = df[~duplicates]
        num_points += len(duplicates)
        num_dupe_points += sum(duplicates)
        if sum(duplicates) > 0:
            num_dupes += 1
            # print(f"[{grb}] Cleaning {sum(duplicates)} duplicates. ({sum(duplicates)}/{len(duplicates)})")

        lt0 = df["time_sec"] <= 0
        if sum(lt0) > 0:
            print(f"[{grb}] Seeing negative times, which means incorrect inputted time(s) :(")
            unphysical_times["grb"].append(grb)
            unphysical_times["filepath"].append(filepath)
            continue

        num_successful += 1
        # df.to_csv(os.path.join(os.path.split(filepath)[0], f"{grb}_converted_flux_cleaned.txt"), sep="\t", index=None)

    unphysical_df = pd.DataFrame.from_dict(unphysical_times)
    unphysical_path = os.path.join(main_dir, "unphysical_times.txt")
    unphysical_df.to_csv(unphysical_path, sep="\t", index=None)

    print(
        "=" * 30,
        # "\nStats\nFiles with Duplicate Times:",
        # num_dupes,
        "\nFiles with unphysical times:",
        len(unphysical_times["grb"]),
        "\nTotal GRBs:",
        len(filepaths),
        "\nNum. Successful:",
        num_successful,
        "\nTotal Points:",
        num_points,
        "\nTotal Discarded Points:",
        num_dupe_points,
        f"({100*num_dupe_points/num_points:.1f}%)",
    )
