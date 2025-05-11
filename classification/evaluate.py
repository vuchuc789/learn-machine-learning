import ml_edu.experiment
import ml_edu.results


def compare_train_test(
    experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]
):
    print("Comparing metrics between train and test:")
    for metric, test_value in test_metrics.items():
        print("------")
        print(f"Train {metric}: {experiment.get_final_metric_value(metric):.4f}")
        print(f"Test {metric}:  {test_value:.4f}")
