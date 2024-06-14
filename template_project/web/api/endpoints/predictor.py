import base64
import os
from io import StringIO

import PIL
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from pydantic import BaseModel, EmailStr
from starlette.responses import JSONResponse

from template_project.predictor_config import predictor_config
from template_project.predictor_model.DiffusionPredictor import DiffusionPredictor

from .utils import BackgroundTaskManager, EmailSender

IMG_DIR = os.path.join(os.path.abspath("./"), "template_project/data/saved_images")
EMAIL_RESPONSE = "Email with the prediction will be sent to {email}"

predictor_model = DiffusionPredictor(
    predictor_config.predictor_model_config,
)

email_sender = EmailSender(
    sender_email=os.environ["SENDER_EMAIL"],
    sender_password=os.environ["SENDER_PASSW"],
)


router = APIRouter()
task_manager = BackgroundTaskManager()


class Prediction(BaseModel):
    """
    A model representing a prediction result.

    Attributes:
    -----------
    name : str
        The name associated with the prediction.
    prediction : int
        The predicted value.
    """

    name: str
    prediction: int


class Input(BaseModel):
    """
    A model representing an input for the service.

    Attributes:
    -----------
    task_id : str
        The task id set by a user. Useful to track the task status.
    data : dict | None
        A dict containing prompts for image generation.
    email: EmailStr | None
        An optional email address to send the prediction to.
    """

    task_id: str
    data: dict | None = None
    email: EmailStr | None = None


def prepare_input(input: Input) -> dict | None:
    """
    A function used to prepare the input for prediction.
    """
    data = input.data
    return data


async def process_input_file(content: bytes) -> dict:
    """
    A function used to unpack prompts from a file into an input dict.
    """
    file_content = content.decode("utf-8")
    file_like_object = StringIO(file_content)
    prompt_list = file_like_object.readlines()
    data = dict(prompt=prompt_list)
    return data


def save_and_return_path(
    prediction: PIL.Image.Image,
    image_name: str,
) -> str:
    """
    A function used to save generated images and to return the saving path.
    """
    img_path = os.path.join(IMG_DIR, f"{image_name}.png")
    prediction.save(img_path)
    return img_path


def prepare_single_image_for_response(
    img_path: str,
    img_name: str,
) -> dict:
    """
    A function used to read an image from disk and put into a response dict.
    """
    with open(img_path, "rb") as img_file:
        encoded_data = base64.b64encode(img_file.read()).decode("utf-8")
    response_dict = {"filename": f"{img_name}.png", "data": encoded_data}
    return response_dict


def prepare_multiple_images_for_response(
    img_path_list: list,
    img_name_list: list,
) -> list:
    """
    A function used to read a list of images from disk and put into a response dict.
    """
    response_list = list()
    for img_path, img_name in zip(img_path_list, img_name_list):
        response_dict = prepare_single_image_for_response(
            img_path=img_path,
            img_name=img_name,
        )
        response_list.append(response_dict)
    return response_list


def make_prediction(input: Input) -> list:
    """
    A function used to make generations with a model.
    """
    prepared_input = prepare_input(input)
    prediction = predictor_model.predict(prepared_input)
    return prediction


def make_prediction_background(input: Input) -> None:
    """
    A function used to make generations with a model in a background task.
    """
    task_id = input.task_id
    task_manager.add_task(task_id)
    prediction = make_prediction(input=input)
    img_name_list = [f"{task_id}_{i}" for i in range(len(prediction))]
    img_path_list = [
        save_and_return_path(prediction=img_pred, image_name=img_name)
        for img_pred, img_name in zip(prediction, img_name_list)
    ]
    email_sender.send_mult_imgs_via_email(
        img_path_list=img_path_list,
        recipient_email=input.email,
    )
    task_manager.mark_task_completed(task_id)


def run_prediction_process(input: Input) -> JSONResponse:
    """
    A function doing end-to-end generations with a model.
    """
    task_id = input.task_id
    prediction = make_prediction(input)
    img_name_list = [f"{task_id}_{i}" for i in range(len(prediction))]
    img_path_list = [
        save_and_return_path(prediction=img_pred, image_name=img_name)
        for img_pred, img_name in zip(prediction, img_name_list)
    ]
    response = prepare_multiple_images_for_response(
        img_path_list=img_path_list,
        img_name_list=img_name_list,
    )
    return JSONResponse(content=response)


async def run_prediction_process_background(
    input: Input,
    background_tasks: BackgroundTasks,
) -> Prediction | dict:
    """
    A function doing end-to-end generations with a model in a background task.
    """
    background_tasks.add_task(make_prediction_background, input=input)
    response_message = EMAIL_RESPONSE.format(email=input.email)
    return {"message": response_message}


@router.post("/predict")
async def predict(input: Input, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    A generation endpoint processing JSON requests.
    """
    try:
        if input.email is not None:
            response = await run_prediction_process_background(
                input=input,
                background_tasks=background_tasks,
            )
            return response
        response = run_prediction_process(input=input)
        return response
    except Exception as e:
        return {"error": str(e)}


@router.post("/predict_file")
async def predict_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task_id: str = Form(...),
    email: str | None = Form(None),
) -> JSONResponse:
    """
    A generation endpoint processing requests containing files.
    """
    try:
        content = await file.read()
        data = await process_input_file(content)
        input = Input(
            task_id=task_id,
            email=email,
            data=data,
        )
        if email is not None:
            response = await run_prediction_process_background(
                input=input,
                background_tasks=background_tasks,
            )
            return response
        response = run_prediction_process(input=input)
        return response
    except Exception as e:
        return {"error": str(e)}


@router.get("/task-status/{task_id}")
def get_task_status(task_id: str) -> dict:
    return {"task_id": task_id, "status": task_manager.get_task_status(task_id)}
