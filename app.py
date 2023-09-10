import os

from beam import App, Runtime, Image, Volume
from transformers import pipeline


app = App(
    name="sentiment-analysis",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        image=Image(
            python_version="python3.9",
            python_packages=["transformers", "torch"],
        ),
    ),
    volumes=[Volume(name="cached_models", path="./cached_models")],
)

# Cache models in a Beam storage volume
os.environ["TRANSFORMERS_CACHE"] = "./cached_models"


# This function runs once when the container boots
def load_models():
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )

    return model


@app.rest_api(loader=load_models)
def predict(**inputs):
    # Retrieve cached model from loader
    model = inputs["context"]

    # Inference
    result = model(inputs["text"], truncation=True, top_k=1)
    prediction = {i["label"]: i["score"] for i in result}

    print(prediction)

    return {"prediction": prediction}
