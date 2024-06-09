import pickle
import mlflow
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    print(os.listdir('/home/src/'))
    mlflow.set_tracking_uri("http://52.70.97.143:5000")
    mlflow.set_experiment("yellow_taxi_duration")
    dv, lr = data
    pickle.dump(dv, open("dv.pickle", "wb"))

    with mlflow.start_run():
        mlflow.log_artifact("/home/src/dv.pickle")
        mlflow.sklearn.log_model(lr, "Linear Regression")





