import numpy as np
import matplotlib.pyplot as plt

from env import WaterChlorinationEnv
from evaluation import evaluate
import matplotlib.pyplot as plt

def visualize_results(results, model_names: list[str], save_fig_filname: None | str = None) -> None:
    """
    Visualizes comparison of evaluation metrics from a list of result dictionaries.

    Parameters:
    - results (List[Dict[str, Any]]): List of dictionaries containing evaluation metrics.
    """

    # Normalize data into scalar values
    scalar_results = []
    for result in results:
        scalar_result = {}
        for key, value in result.items():
            if isinstance(value, (list, np.ndarray)):
                scalar_result[key] = float(value[0]) if len(value) > 0 else 0.0
            else:
                scalar_result[key] = float(value)
        scalar_results.append(scalar_result)

    # Extract all unique keys
    keys = list(scalar_results[0].keys())
    num_metrics = len(keys)
    num_runs = len(scalar_results)

    # Prepare data for plotting
    values_per_metric = {key: [res[key] for res in scalar_results] for key in keys}

    # Plotting
    fig, axs = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics), constrained_layout=True)

    if num_metrics == 1:
        axs = [axs]  # ensure axs is always iterable

    for ax, key in zip(axs, keys):
        bars = ax.bar(range(num_runs), values_per_metric[key])
        ax.set_title(f"Metric: {key}")
        ax.set_xlabel("Run Index")
        ax.set_ylabel("Value")
        ax.set_xticks(range(num_runs))
        ax.set_xticklabels(model_names)
        ax.grid(True)

        # Add value labels above the bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.5f}", ha='center', va='bottom', fontsize=10)

    if save_fig_filname:
        plt.savefig(save_fig_filname)
        print(f"Results saved to {save_fig_filname}")
    plt.show()


# compare models:
def compare_models(models: list, env: WaterChlorinationEnv, save_results_to: None | str = None,
                   extend_eval: bool = False) -> None:
    """
    Compare multiple models on the same environment.

    Parameters
    ----------
    save_results_to : str, optional
        If provided, the results will be saved to this file.
        If None, the results will not be saved.
    models : list
        List of ChlorinationControlPolicy instances to compare.
    env : WaterChlorinationEnv
        Environment in which the policies are evaluated.
    """
    results = []
    model_names = [model.__class__.__name__ for model in models]
    for model in models:
        print(f"Evaluating policy: {model.__class__.__name__}")
        if extend_eval:
            results.append(evaluate(model, env, extend_eval=True))
        else:
            results.append(evaluate(model, env))

    visualize_results(results, model_names, save_fig_filname=save_results_to)
