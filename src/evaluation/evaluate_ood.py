"""
Evaluation for OOD dataset
"""

from src.datasets.image_loaders import get_image_loader
from src.evaluation.utils import cross_load_dataset
from src.evaluation.metrics import get_all_classification_stats

def evaluate_laplace_ood(model, dataset, data_dir, target_dataset,
                    batch_size=256, λ=1., workers=4, cuda=True, n_test_data=None, test_subset_idx=-1,
                    scaling=None, res_path=None, eval_on_test_subset=False):


    val_loader = get_image_loader(dataset, batch_size, cuda=cuda, workers=workers, distributed=False,
                                    data_dir=data_dir, n_val_data=n_test_data, val_subset_idx=test_subset_idx)[2]

    source_logprobs, targets = model.predict(val_loader, λ=λ, scaling=scaling)

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
