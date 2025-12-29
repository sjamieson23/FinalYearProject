from pathlib import Path

from dateutil.utils import today
from google.cloud import storage
from googleapiclient import discovery
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }


def uploadDataToBucket(path_str):
    client = storage.Client()
    bucket = client.get_bucket("model-storage-data")
    folder_name = today().strftime("%Y-%m-%d") + "_test1"
    base = Path(path_str)
    for p in base.rglob("*"):
        if p.is_file():
            # Object name mirrors relative path under the provided directory
            rel = p.relative_to(base).as_posix()  # forward slashes
            blob = bucket.blob(f"{folder_name}/{base.as_posix()}/{rel}")
            blob.upload_from_filename(str(p))


def terminateVM():
    service = discovery.build('compute', 'v1')

    project = 'final-year-project-477110'
    zone = 'us-west1-a'
    instance_name = 'single-agent-training'

    request = service.instances().stop(project=project, zone=zone, instance=instance_name)
    response = request.execute()
    print("Stop request submitted:", response)
