import os
import sys
from PIL import Image
from tqdm import tqdm
from llm.api import Agent, ContentBuilder
import json

root_dir = '/amax/hchuz/architectural_heritage/data/control/img'
#target_dir = '/amax/hchuz/architectural_heritage/data/分类的texture/all'
jsonl_file = '/amax/hchuz/architectural_heritage/data/control/meta.jsonl'
json_file = '/amax/hchuz/architectural_heritage/data/control/meta.json'

#prefixs = ['[Stone, Granite, Concrete]', '[Plaster, Stucco, Concrete]', '[Red Brick, Brick, Concrete]']

def update_jsonl_to_json(jsonl_file, json_file):
    json_lines = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            json_lines.append(json.loads(line)) 
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    for line in json_lines:
        if line['img_path'] not in json_dict:
            json_dict[line['img_path']] = {'caption':line['caption']}
    with open(json_file, 'w') as f:
        json.dump(json_dict, f, indent=2)

update_jsonl_to_json(jsonl_file, json_file)

api_key = '3PxBu2J1g8rrLMvfo3QGsjOwNDXVmXZC'
model = "pixtral-large-2411"
agent = Agent(token=api_key, model=model)


with open(json_file, 'r') as f:
    json_dict = json.load(f)
#textures = [it for it in os.listdir(root_dir) if 'texture' in it and 'test' not in it]
#assert len(prefixs) == len(textures)
#for texture in textures:
img_root = root_dir
img_names = os.listdir(img_root)
for img_name in tqdm(img_names):
    if 'jpg' not in img_name:
        continue
    img_path = os.path.join(img_root, img_name)
    # with open(os.path.join(img_root, img_name.replace('png', 'txt')), 'w') as f:
    #     f.write(prefix + ' ' + json_dict[img_path]['caption'])
    if img_path in json_dict:
        continue
    img = Image.open(img_path)

    # chat gpt
    contentbuilder = ContentBuilder()
    contentbuilder.add_image(img)
    contentbuilder.add_text('Briefly describe the materials of the building in the image.')
    text = agent.chat(contentbuilder.finish())
    agent.clear()
    with open(jsonl_file, 'a') as f:
        json_line = json.dumps({'img_path': img_path, 'caption': text})  # 将对象转换为 JSON 格式的字符串
        f.write(json_line + '\n')  # 写入文件并换行

    # write to itself
    # with open(os.path.join(img_root, img_name.replace('png', 'txt')), 'w') as f:
    #     f.write('['+texture+']'+ ' an architectural heritage')

    # write to all
    # with open(os.path.join(target_dir, img_name.replace('png', 'txt')), 'w') as f:
    #     f.write('['+texture+']'+ ' an architectural heritage')
    # img.save(os.path.join(target_dir, img_name))

update_jsonl_to_json(jsonl_file, json_file)