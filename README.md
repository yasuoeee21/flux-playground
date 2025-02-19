用flux redux controlnet(canny) fill实现一阶段参考整体风格图与整体布局的生成，与二阶段参考局部风格图与局部mask的生成。

## 环境
### diffusers
```
diffusers 0.32.0.dev0
```
### flux and its tools
```
https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union

https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev

https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
```

### detector
```
https://github.com/IDEA-Research/GroundingDINO

https://github.com/facebookresearch/segment-anything
```

## 运行
```
main.ipynb
```