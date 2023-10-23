import argparse
import glob
import multiprocessing as mp
import os
import pickle
import yaml

import lmdb
import numpy as np
from tqdm import tqdm


from ocpmodels.preprocessing import SequenceToGraphs


def write_images_to_lmdb(mp_arg):
    a2g, db_path, samples, sampled_ids, idx, pid, args = mp_arg
    datasets = args.datasets
    feature_selection_configs = args.feature_selection_configs
    field_configs = args.field_configs
    num_datasets = len(datasets)

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )

    error = []
    # loop over all samples in the main dataset
    for sample in samples:
        if_error = False
        with open(sample, 'rb') as f:
            frame_log = pickle.load(f)
        # for each field, if it is in the selected features config dictionary,
        # then use the selected features, otherwise use all features
        for key in field_configs[0]:
            frame_log[field_configs[0][key]] = frame_log.pop(key)[feature_selection_configs[0][key]] \
                if key in feature_selection_configs[0] else frame_log.pop(key)
        frame_name = os.path.splitext(os.path.basename(sample))[0]
        if a2g.r_labels:
            for label in a2g.labels:
                assert label in frame_log
        # concatenate the other fields from the other datasets
        for i in range(1, num_datasets):
            try:
                with open(os.path.join(datasets[i],  frame_name + ".pkl"), 'rb') as f_inner:
                    frame_log_inner = pickle.load(f_inner)
            except:
                if_error = True
                break

            # for each field, if it is in the selected features config dictionary,
            # then use the selected features, otherwise use all features
            for key in field_configs[i]:
                frame_log[field_configs[i][key]] = frame_log_inner.pop(key)[feature_selection_configs[i][key]] \
                    if key in feature_selection_configs[i] else frame_log_inner.pop(key)

        if if_error:
            error.append(frame_name + "\n")
            continue
        sid = frame_log.get(args.idx, frame_name)
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

    return sampled_ids, idx, error


def main(args: argparse.Namespace) -> None:
    # load datasets config
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    # add output_path
    args.out_path = dataset_config["output"]

    dataset_config = dataset_config["input"]
    num_datasets = len(dataset_config)
    # load datasets
    datasets = [dataset_config[i]["src"] for i in range(num_datasets)]
    args.datasets = datasets
    # load feature selection configuration
    feature_selection_configs = []
    for i in range(num_datasets):
        feature_selection_config = os.path.join(
            dataset_config[i]["src"],
            dataset_config[i].get("feature_selection", 'None')
        )
        if os.path.exists(feature_selection_config):
            with open(feature_selection_config, 'rb') as f:
                selected_features = pickle.load(f)
        else:
            selected_features = {}
        feature_selection_configs.append(selected_features)
    args.feature_selection_configs = feature_selection_configs
    # load field configuration
    field_configs = []
    for i in range(num_datasets):
        field_configs.append(dataset_config[i].get("fields", {}))
    args.field_configs = field_configs

    xyz_logs = glob.glob(os.path.join(dataset_config[0]["src"], "*.pkl"))
    if not xyz_logs:
        raise RuntimeError("No *.pkl files found. Did you uncompress?")
    if args.num_workers > len(xyz_logs):
        args.num_workers = len(xyz_logs)

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
    chunked_txt_files = np.array_split(xyz_logs, args.num_workers)

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
    sampled_ids, idx, errors = list(op[0]), list(op[1]), list(op[2])

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

    # flatten the errors list to a single list
    errors = [item for sublist in errors for item in sublist]
    # log error
    if errors:
        # print the number of errors
        print("There are {} errors logged in {}".format(
            len(errors), os.path.join(args.out_path, "error.txt"))
        )
        with open(os.path.join(args.out_path, "error.txt"), "w") as f:
            f.writelines(errors)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idx",
        help="idx",
    )
    parser.add_argument(
        "--labels",
        help="labels. Make sure to put them, if any, in the first dataset",
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
        "--dataset_config",
        help="path to the dataset config file",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
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
