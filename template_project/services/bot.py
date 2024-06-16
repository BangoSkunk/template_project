import asyncio
import base64
import json
import logging
import os

import aiohttp
from aiogram import Bot, Dispatcher, types

bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()


SERVER_URL = "http://127.0.0.1:8000/api/predict"


async def send_post_request(message_text, user_id):
    async with aiohttp.ClientSession() as session:

        async with session.post(
            SERVER_URL,
            json={"task_id": str(user_id), "data": {"prompt": message_text}},
        ) as response:
            if response.status == 200:
                response_json = await response.text()
                response_list = json.loads(response_json)
                image_list = list()
                for response_dict in response_list:
                    encoded_image = response_dict.get("data", 0)
                    if encoded_image == 0:
                        raise Exception(
                            f"Didn't get an image, instead got the following response: {response}",
                        )
                    image_bytes = base64.b64decode(encoded_image.encode("utf-8"))
                    image_file = types.BufferedInputFile(
                        image_bytes,
                        filename=f"{user_id}.jpg",
                    )
                    image_list.append(image_file)
                return image_list
            else:
                raise Exception(
                    f"Server returned {response.status, response.text} status code",
                )


@dp.message()
async def process_text(message: types.Message):
    user_id = message.from_user.id
    text = message.text

    try:
        image_list = await send_post_request(text, user_id)
        for image in image_list:
            await bot.send_photo(chat_id=message.chat.id, photo=image)

    except Exception as e:
        logging.exception(f"Error processing message: {e}")
        await message.reply("An error occurred while processing your request.")


async def main():
    logging.basicConfig(level=logging.DEBUG)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
