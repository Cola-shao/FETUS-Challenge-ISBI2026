import os
import json
import h5py
import random
import argparse
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="Generate train/valid JSON for semi-supervised FETUS 2026 challenge")

    parser.add_argument("--root", type=str, default="data", help="Dataset root path")
    parser.add_argument("--n_image_per_view", type=int, default=20, help="Number of validation samples per view")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    return parser.parse_args()



def stratified_sample_by_view(filenames, views, n_image_per_view=30):
    view_groups = defaultdict(list)

    for name, view in zip(filenames, views):
        view_groups[view].append(name)

    sampled = []
    for view, files in view_groups.items():
        if len(files) < n_image_per_view:
            print("[Warning] View {} has {} samples, fewer than {}. Keeping all of them.".format(view, len(files), n_image_per_view))
            sampled.extend(files)
        else:
            sampled.extend(random.sample(files, n_image_per_view))

    return sampled


if __name__ == "__main__":
    args = get_args()

    dataset_root_path = args.root
    images_dir_path = os.path.join(dataset_root_path, 'images')
    labels_dir_path = os.path.join(dataset_root_path, 'labels')

    # all image filenames
    all_image_filenames = [name for name in os.listdir(images_dir_path) if name.endswith('.h5')]

    # labeled image filenames
    all_labeled_filenames = [name.replace('_label', '') for name in os.listdir(labels_dir_path) if name.endswith('.h5')]

    # unlabeled image filenames
    all_unlabeled_filenames = [name for name in all_image_filenames if name not in all_labeled_filenames]

    # labeled views
    all_labeled_views = []
    for filename in all_labeled_filenames:
        with h5py.File(os.path.join(images_dir_path, filename), 'r') as f:
            all_labeled_views.append(int(f['view'][:]))

    random.seed(args.seed)

    valid_filename_list = stratified_sample_by_view(all_labeled_filenames, all_labeled_views, args.n_image_per_view)
    train_filename_list = [name for name in all_labeled_filenames if name not in valid_filename_list]

    # validation set
    valid_dataset_list = []
    for label_filenames in valid_filename_list:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        with h5py.File(image_h5_file_path, 'r') as f:
            view_id = int(f['view'][:])

        label_h5_file_path = os.path.abspath(os.path.join(labels_dir_path, label_filenames.replace('.h5', '_label.h5')))

        valid_dataset_list.append({
            'image': image_h5_file_path,
            'label': label_h5_file_path,
            'view_id': view_id
        })

    # training set with labeled
    train_labeled_dataset_list = []
    for label_filenames in train_filename_list:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        with h5py.File(image_h5_file_path, 'r') as f:
            view_id = int(f['view'][:])

        label_h5_file_path = os.path.abspath(os.path.join(labels_dir_path, label_filenames.replace('.h5', '_label.h5')))

        train_labeled_dataset_list.append({
            'image': image_h5_file_path,
            'label': label_h5_file_path,
            'view_id': view_id
        })

    # training set with unlabeled
    train_unlabeled_dataset_list = []
    for label_filenames in all_unlabeled_filenames:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        with h5py.File(image_h5_file_path, 'r') as f:
            view_id = int(f['view'][:])

        train_unlabeled_dataset_list.append({
            'image': image_h5_file_path,
            'label': None,
            'view_id': view_id
        })

    # save JSON
    with open(os.path.join(dataset_root_path, 'train_labeled.json'), 'w') as f:
        json.dump(train_labeled_dataset_list, f, indent=4)

    with open(os.path.join(dataset_root_path, 'train_unlabeled.json'), 'w') as f:
        json.dump(train_unlabeled_dataset_list, f, indent=4)

    with open(os.path.join(dataset_root_path, 'valid.json'), 'w') as f:
        json.dump(valid_dataset_list, f, indent=4)





