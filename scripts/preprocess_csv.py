"""
Creates LMDB files with extracted graph features from provided *.csv files
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import yaml
from datasets import Dataset

import lmdb
from tqdm import tqdm

from ocpmodels.preprocessing import SequenceToGraphs


def write_images_to_lmdb(mp_arg):
    a2g, db_path, samples, sampled_ids, idx, pid, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Set CPU affinity and limit CPU usage.
    # p = psutil.Process()
    # p.nice(19)
    # p.cpu_affinity([p._pid % psutil.cpu_count()])
    # p.cpu_percent(100)

    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for i, frame_log in enumerate(samples):
        sid = frame_log.get(args.idx, None)
        data_object = a2g.convert(frame_log, sid)

        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        idx += 1
        sampled_ids.append(str(sid) + "\n")
        pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def main(args: argparse.Namespace) -> None:
    # load datasets config
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    # add output_path
    args.out_path = dataset_config["output"]
    args.data_path = dataset_config["input"]

    xyz_logs = glob.glob(os.path.join(args.data_path, "*.csv"))
    if not xyz_logs:
        raise RuntimeError("No *.csv files found. Did you uncompress?")
    hg_dataset = Dataset.from_csv(xyz_logs)

    # Initialize feature extractor.
    a2g = SequenceToGraphs(
        r_labels=not args.test_data,
        labels=[label.strip() for label in args.labels.split(",")] if args.labels else None,
        sequence_field=args.sequence_field,
        dense_fields=[dense_field.strip() for dense_field in args.dense_fields.split(",")] if args.dense_fields else None,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_txt_files = [hg_dataset.shard(args.num_workers, index=i) for i in range(args.num_workers)]

    # Extract features
    sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            chunked_txt_files[i],
            sampled_ids[i],
            idx[i],
            i,
            args,
        )
        for i in range(args.num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    sampled_ids, idx = list(op[0]), list(op[1])

    # Log sampled image, trajectory trace
    # print the number of sampled ids
    print("There are {} sampled ids logged in {}".format(
        sum([len(ids) for ids in sampled_ids]),
        os.path.join(args.out_path, "data_log"))
    )
    for j, i in enumerate(range(args.num_workers)):
        ids_log = open(
            os.path.join(args.out_path, "data_log.%04d.txt" % i), "w"
        )
        ids_log.writelines(sampled_ids[j])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to dir containing *.csv files",
    )
    parser.add_argument(
        "--dataset_config",
        help="path to the dataset config file",
    )
    parser.add_argument(
        "--labels",
        help="labels",
    )
    parser.add_argument(
        "--idx",
        help="idx",
    )
    parser.add_argument(
        "--sequence_field",
        help="unique sequence field",
    )
    parser.add_argument(
        "--dense_fields",
        help="dense fields",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Is data being processed test data?",
    )
    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()
    main(args)
