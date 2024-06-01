from train_model_up import train_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# model = train_model()

app = FastAPI(
    title="Mlflow Experiment Management",
    description="""Usage, Download the ASL Dataset https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet
extract Test_Alphabet and Train_Alphabet into your working folder
make sure to run mlflow ui --port(any port number you want) to see the experiments
If there are no experiments, the bottom part of the app will not be visible.
Streamlit and MLFlow should be running in different ports, for example 
(streamlit run streamlit_app.py --port 1350)
OR
(mlflow ui --port 1000)

todo list here is to test fast.api implementation and add new functions such as Model Registry and Adding Descriptions to each run."""

)


class TrainRequest(BaseModel):
    image_height: int
    batch_size: int
    num_epochs: int
    num_classes: int
    learning_rate: float
    seed: int



@app.post("/train_model")
async def train_model_endpoint(request: TrainRequest):
    try:
        image_height = request.image_height
        batch_size = request.batch_size
        num_epochs = request.num_epochs
        num_classes = request.num_classes
        learning_rate = request.learning_rate
        seed = request.seed

        # your existing training code here
        try:
            train_model(image_height=image_height, batch_size=batch_size, num_epochs=num_epochs,
                                num_classes=num_classes, learning_rate=learning_rate, seed=seed)
        except Exception as e:
            print(e)

        return {"message": "Model training completed successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
