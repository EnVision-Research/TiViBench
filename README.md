<h2 align="center"> TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models</h2>
<div align="center">

_**[Harold H. Chen](https://haroldchen19.github.io/)<sup>1,2*</sup>, [Disen Lan](https://landisen.github.io/)<sup>3*</sup>, [Wen-Jie Shu](https://github.com/EnVision-Research/TiViBench)<sup>2*</sup>, [Qingyang Liu](https://github.com/EnVision-Research/TiViBench)<sup>4</sup>, [Zihan Wang](https://github.com/EnVision-Research/TiViBench)<sup>1</sup>, [Sirui Chen](https://github.com/EnVision-Research/TiViBench)<sup>1</sup>, [Wenkai Cheng](https://github.com/EnVision-Research/TiViBench)<sup>1</sup>, [Kanghao Chen](https://khao123.github.io/)<sup>1,2</sup>, [Hongfei Zhang](https://github.com/EnVision-Research/TiViBench)<sup>1</sup>, [Zixin Zhang](https://github.com/EnVision-Research/TiViBench)<sup>1,2</sup>, [Rongjin Guo](https://github.com/EnVision-Research/TiViBench)<sup>5</sup>,
<br>
[Yu Cheng](https://ych133.github.io/)<sup>6†</sup>, [Ying-Cong Chen](https://www.yingcong.me/)<sup>1,2†</sup>**_
<br><br>
<sup>*</sup>Equal Contribution; <sup>†</sup>Corresponding Author
<br>
<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>FDU, <sup>4</sup>SJTU, <sup>5</sup>CityUHK, <sup>6</sup>CUHK

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

 <a href='https://arxiv.org/abs/2511.13704'><img src='https://img.shields.io/badge/arXiv-2511.13704-b31b1b.svg'></a>
 [![Project Page](https://img.shields.io/badge/TiViBench-Website-green?logo=googlechrome&logoColor=green)](https://haroldchen19.github.io/TiViBench-Page/)
<br>

</div>


### Table of Contents
- [News](#news)
- [Overview](#overview)
- [Evaluation Results](#evaluation_results)
- [Installation](#installation)
- [Inference Suite](#inference_suite)
- [Evaluation Suite](#evaluation_suite)
- [VideoTPO](#videotpo)
- [Citation](#citation)


<a name="news"></a>
## 📌 News
- [11/2025] 🔥 We release VideoTPO inference code on Wan2.1!
- [11/2025] 🔥 We release TiViBench, a hierarchical manner benchmark tailored to visual reasoning for I2V generation models!


## 🧰 TODO

- [x] Release Paper.
- [x] Release VideoTPO inference code.
- [ ] Release data and eval code.


<a name="overview"></a>
## 🌟 Overview

The rapid evolution of video generative models has shifted their focus from producing visually plausible outputs to tackling tasks requiring physical plausibility and logical consistency. However, despite recent breakthroughs such as Veo 3's chain-of-frames reasoning, it remains unclear whether these models can exhibit reasoning capabilities similar to large language models (LLMs). Existing benchmarks predominantly evaluate visual fidelity and temporal coherence, failing to capture higher-order reasoning abilities. To bridge this gap, we propose **TiViBench**, a hierarchical manner benchmark specifically designed to evaluate the reasoning capabilities of image-to-video (I2V) generation models. TiViBench systematically assesses reasoning across four dimensions: i) **Structural Reasoning & Search**, ii) **Spatial & Visual Pattern Reasoning**, iii) **Symbolic & Logical Reasoning**, and iv) **Action Planning & Task Execution**, spanning 24 diverse task scenarios across 3 difficulty levels. Through extensive evaluations, we show that commercial models (*e.g.*, Sora 2, Veo 3.1) demonstrate stronger reasoning potential, while open-source models reveal untapped potential that remains hindered by limited training scale and data diversity. To further unlock this potential, we introduce **VideoTPO**, a simple yet effective test-time strategy inspired by preference optimization. By performing LLM self-analysis on generated candidates to identify strengths and weaknesses, VideoTPO significantly enhances reasoning performance without requiring additional training, data, or reward models. Together, TiViBench and VideoTPO pave the way for evaluating and advancing reasoning in video generation models, setting a foundation for future research in this emerging field.

<table class="center">
    <tr>
    <td><img src="assets/tivibench.png"></td>
    </tr>
    <tr>
    <td><img src="assets/pipeline.png"></td>
    </tr>
</table>


<a name="evaluation_results"></a>
## 📈 Evaluation Results

Pass@1 performance overview on our TiViBench of 3 commercial models and 4 open-source models:
<table class="center">
    <tr>
    <td><img src="assets/radar.png"></td>
    </tr>
</table>



<a name="installation"></a>
## 🚀 Installation

1. Clone this repository and navigate to source folder
```bash
cd TiViBench
```

2. Build Environment 


```Shell
echo "Creating conda environment"
conda create -n TiViBench python=3.10
conda activate TiViBench

echo "Installing dependencies"
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install opencv-python pytesseract scikit-image pillow
pip install dds-cloudapi-sdk==0.5.3 # DINO-X for Eval
```


<a name="inference_suite"></a>
## 📍 Inference Suite

### Prompt Suite

The inference prompts can be found in `~/eval_cache/**_prompt.json` files:
```
├─SS_prompt.json # Structural Reasoning & Search
├─SV_prompt.json # Spatial & Visual Pattern Reasoning
├─SL_prompt.json # Symbolic & Logical Reasoning
├─AT_prompt.json # Action Planning & Task Execution
```

### Image Suite

You can access our image suite on [Google Drive].

**Automatic Download**
```
pip install gdown
```
```
python scripts/image_suit_download.py
```

**Data Format**

The image suite is organized in the following format `~/images/`:
```
├─AT # Action Planning & Task Execution
├─SL # Symbolic & Logical Reasoning
├─SS # Structural Reasoning & Search
├─SV # Spatial & Visual Pattern Reasoning
├──easy_graph_001.png
├──easy_graph_002.png
......
```

The default size of all images is 1280x720. We provide adaptive cropping of images to fit your video model.


### Inference Details

For each image-prompt pair, sample 5 videos with 5 fixed random seeds to ensure the evaluation results are reproducible. To facilitate subsequent evaluation, we strongly recommend that you organize your generation results in the following format:
```
├─AT_easy_game_001
├──AT_easy_game_001-0.mp4
├──AT_easy_game_001-1.mp4
├──AT_easy_game_001-2.mp4
├──AT_easy_game_001-3.mp4
├──AT_easy_game_001-4.mp4
├─AT_easy_game_002
......
├─SV_medium_graph_050
```


<a name="evaluation_suite"></a>
## 🚩 Evaluation Suite

### Data Preparation

Please download the [data] required for evaluations:
```
python scripts/eval_suit_download.py
```

and put them in the folder `./eval_cache`:
```
├─AT
├─SL
├──easy_{type}_001
├───end.png
.....
├─SS
├─SV
```

### Evaluation

**Dimension-by-Dimension**

To perform evaluation on one dimension:
```
python evaluate.py --base_path $VIDEO_FOLDER --dimension $DIMENSION
```
- Dimensions: `AT`, `SL`, `SS`, and `SV`.
- The evaluation result will be saved in `./evaluation_results`.
- Please specify the DINO-X and Gemini API in `./metrics/dinox.py` and `./metrics/gemini.py`.

**All Four Dimensions**

We also provide an overall evaluation for all four dimensions, just run:
```
python evaluate.py --base_path $VIDEO_FOLDER 
```

**Only Pass@1**
```
python evaluate.py --base_path $VIDEO_FOLDER --metric 'pass@1'
```

<a name="videotpo"></a>
## 🚁 VideoTPO

### VideoTPO on Wan2.1:
```bash
cd VideoTPO
```
Build Environment:
```Shell
echo "Creating conda environment"
conda create -n VideoTPO python=3.10
conda activate VideoTPO

echo "Installing dependencies"
pip install textgrad
pip install -r requirements.txt
```
Set Optimizer:
```Shell
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```
### Run
```Shell
python videotpo_wan.py \
    --output_dir /path/to/output \
    --image_path /path/to/image.png \
    --init_prompt "Your reasoning prompt here" \
    --task "Graph Traversal" \
    --seed1 100 \
    --seed2 200

# See details
python inference_videotpo.py --help
```

<a name="citation"></a>
## 📝 Citation
Please consider citing our paper if our benchmark or test-time strategy are useful:
```bib
@article{chen2025tivibench,
  title={TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models},
  author={Chen, Harold Haodong and Lan, Disen and Shu, Wen-Jie and Liu, Qingyang and Wang, Zihan and Chen, Sirui and Cheng, Wenkai and Chen, Kanghao and Zhang, Hongfei and Zhang, Zixin and others},
  journal={arXiv preprint arXiv:2511.13704},
  year={2025}
}
```


## 📪 Contact
For any question, feel free to email `haroldchen328@gmail.com` or `disenlan1002@gmail.com`.
