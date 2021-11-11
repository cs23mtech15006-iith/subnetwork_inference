import torch

from src.datasets.image_loaders import get_image_loader
from src.evaluation.utils import load_corrupted_dataset, rotate_load_dataset, cross_load_dataset
from src.evaluation.metrics import get_all_classification_stats


def evaluate_laplace(model, dataset, data_dir, corruption=None, rotation=None, target_dataset=None,
                    batch_size=256, λ=1., workers=4, cuda=True, n_test_data=None, test_subset_idx=-1,
                    scaling=None, res_path=None, eval_on_test_subset=False):

    if corruption == 0:
        corruption = None
    if rotation == 0:
        rotation = None

    # TODO: delete these
    assert not (corruption is not None and rotation is not None) # dont corrupt and rotate at the same time
    assert not ((corruption is not None or rotation is not None) and target_dataset is not None) # dont enable corruption or rotation if you have a target dataset

    if corruption is None and rotation is None:
        val_loader = get_image_loader(dataset, batch_size, cuda=cuda, workers=workers, distributed=False,
                                    data_dir=data_dir, n_val_data=n_test_data, val_subset_idx=test_subset_idx)[2]
    elif corruption is not None:
        val_loader = load_corrupted_dataset(dataset, severity=corruption, data_dir=data_dir, batch_size=batch_size,
                                            cuda=cuda, workers=workers, n_data=n_test_data, subset_idx=test_subset_idx)
    elif rotation is not None:
        val_loader = rotate_load_dataset(dataset, rotation, data_dir=data_dir, batch_size=batch_size, cuda=cuda,
                                        workers=workers, n_data=n_test_data, subset_idx=test_subset_idx)

    source_logprobs, targets = model.predict(val_loader, λ=λ, scaling=scaling)

    target_logprobs = None
    if target_dataset is not None:
        _, target_loader = cross_load_dataset(dataset, target_dataset, data_dir=data_dir,
                                            batch_size=batch_size, workers=workers, n_data=n_test_data)
        target_logprobs, _ = model.predict(target_loader, λ=λ, scaling=scaling)

    # if we predicted on an indexed subset, store the predictions to disk for later aggregation
    if not eval_on_test_subset and (n_test_data is not None and test_subset_idx != -1):
        import pickle
        with open(res_path, 'wb') as file:
            res_dict = {'source_logprobs': source_logprobs, 'targets': targets, 'target_logprobs': target_logprobs}
            pickle.dump(res_dict, file)
        return None

    return get_all_classification_stats(source_logprobs, targets, target_logprobs)


def aggregate_results(res_path, aggregate_indices):
    """ Aggregate results across data subsets """

    import pickle
    from pathlib import Path

    # load predictions
    all_source_logprobs = []
    all_targets = []
    all_target_logprobs = []
    for idx in range(aggregate_indices):
        filename = res_path.absolute().as_posix().replace(".csv", f"_{idx}test_subset_idx.csv")
        if not Path(filename).exists():
            print(f"{filename} doesn't exist -- skipping.")
            continue
        with open(filename, "rb") as file:
            res_dict = pickle.load(file)
        all_source_logprobs.append(res_dict["source_logprobs"])
        all_targets.append(res_dict["targets"])
        all_target_logprobs.append(res_dict["target_logprobs"])

    # aggregate predictions and compute metrics
    source_logprobs = torch.cat(all_source_logprobs, dim=0)
    targets = torch.cat(all_targets)
    target_logprobs = torch.cat(all_target_logprobs, dim=0) if all_target_logprobs[0] is not None else None

    return get_all_classification_stats(source_logprobs, targets, target_logprobs)
