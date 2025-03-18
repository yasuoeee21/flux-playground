import base64
from mistralai import Mistral
from PIL import Image
from typing import Union
import io

class ContentBuilder:
    def __init__(self):
        self.content = []

    def add_text(self, text):
        item = {
            "type": "text",
            "text": text
        }
        self.content.append(item)

    def add_image(self, path_or_image: Union[str, Image.Image] = None):
        if type(path_or_image) == str:
            base64_image = self.encode_image(path_or_image)
        else:
            base64_image = self.encode_pil_image(path_or_image)
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        self.content.append(item)

    def finish(self):
        return self.content

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def encode_pil_image(self, pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


class Agent:
    def __init__(self, token, model='pixtral-large-2411'):
        self.client = Mistral(
            api_key=token,
        )
        self.model = model

        self.clear()

    def clear(self):
        self.conversation = []

    def chat(self, content, temperature=0.2):
        self.conversation.append({"role": "user", "content": content})

        response = self.client.chat.complete(
            model=self.model,
            messages=self.conversation,
            temperature=temperature
        )
        #print(response)

        assistant_data = response.choices[0].message
        self.conversation.append(assistant_data)

        # 返回响应中的文本部分
        return assistant_data.content