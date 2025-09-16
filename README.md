KidLLM: An General-purpose Language Model for Children Speech Recognition and Understanding

# Installation

This codebase is tested on Ubuntu 20.04 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

Setup conda environment (recommended).

**Create a conda environment**

```
conda create -y -n kidspeak python=3.10
conda activate kidspeak
```

**Install torch, torchaudio (requires torchaudio >= 2.4.0) and torchvision**

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pin install -r requirement.txt
```

## 1 Dataset Preparation
### 1.1 Ultrasuite Dataset
Download the dataset from the Ultrasuite official website (https://ultrasuite.github.io/download/#uxtd-uxssd-upx). Note that we only need the /core-upx subset, which is about 191G, this step may take servel hours.

```bash
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-upx .
```
### 2.1 ENNI Dataset
Use the instruction from [FASA README](./FASA/README.md) to construct the ENNI dataset.

## 2 Training the KidLLM Model
This command is used to train our KidLLM model. Our model is trained using the DeepSpeed framework. In the `code/dsconfig/openllama_peft_stage_1.json` file, you can modify the training parameters for KidLLM.

```bash
sh code/train.sh 
```

In the `train.sh` command, you can modify the path of the pre-trained model we load, the path of the training data, and the path for saving the checkpoint.

## 3 Evaulate the KidLLM Model
Here is the English translation of your request:

"In the `eval.sh`, you need to update the path to your trained checkpoint. KidLLM will load the pre-trained model and perform evaluation, with the evaluation results saved in a `.txt` file under the `/res` directory."

```bash
cd code
sh eval.sh 
```

## 4 Parsing results
After obtaining the output of KidLLM, we need to parse it against the dataset's ground truth. This part is implemented through `evaluate_responses.py`. The model will automatically evaluate the accuracy of tasks such as disorder detection and save the evaluation results in the `result.json` file.

```bash
cd code/parse
sh parse.sh
```