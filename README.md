# Segment Anything (SAM) 
Segment Anything  takes inspiration from chat based `LLM` where `prompting` is an integral part.

It contains three components.
1. Image Encoder
2. Prompt Encoder
3. Mask Decoder

<img src= "https://learnopencv.com/wp-content/uploads/2023/04/segment-anything-model.png"/>

## Model Architecture
Firstly, the input image is passed through the `image encoder` which produces a one-time `embedding` for the image.

A `prompt decoder` for points, boxes or text. 

- For points, `x` and `y` coordinates along with foreground and background information becomes input to the `encoder`.
- For boxes, `bounding box` coordinates become the input of the `encoder`
- For text, `tokens` become the input.

In case, we provide a `mask` as an input, it directly goes through a `downsampling` stage. The downsampling happens using `2D` convolution layers. Then the model concatenates it with the `image embedding` to get the final vector.

Now, any `vector` that the model gets from the $prompt vector + image embedding$ passes through a `lightweight decoder` that creates the final segmentation `mask`.

## 1. SAM Image Encoder
Image Encoder is one of the most powerful components of `SAM`. It is built upon `MAE pretrained ViT` model.
## 2. Prompt Encoder
In this, `points`, `boxes` and `text` act as sparse inputs and masks act as dense inputs. The creators of SAM represent points and bounding boxes using `positional encodings` and sum it with `learned embeddings`. For text prompts, SAM uses the `text encoder` from CLIP. For `masks` as prompts, after downsampling, the `embedding` is summed element-wise with the input image embedding.

## SAM Weights
As of now, there are three different scales of `ViT` models
1. ViT-B SAM (375 MB)
2. ViT-L SAM (1.25 GB)
3. ViT-H SAM (2.56 GB)

## Inference
We can use any of these scale versions of `SAM ViT` for running inference on video, image etc.

1. Download the weights using `hf hub` link of the official `fbaipublicfiles` link like this 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
In this, we are using `ViT-H SAM` model.
2. Clone the repo using this 
```
git clone https://github.com/facebookresearch/segment-anything.git
```
3. Run setup.py to install the module
### Links
1. https://github.com/facebookresearch/segment-anything [Official Repo]
2. https://arxiv.org/pdf/2304.02643.pdf [Paper]
3. https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/ [Blog]
4. https://huggingface.co/ybelkada/segment-anything [HF Hub Weights]
5. https://www.kaggle.com/code/raghvender/segment-anything-sam-onnx [Kaggle Notebook]