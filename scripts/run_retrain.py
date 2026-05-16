from src.data.download import download_data
from src.data.preprocessing import run_pipeline
from src.models.train import train

def run_training_pipeline():
    data = download_data()
    processed = run_pipeline(data)
    model = train(processed)
    return model