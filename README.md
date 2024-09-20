 # Embrace the world of large language models!
 
 ## Description
 
This repository stores the source code and data for the paper 'A Prompt-Engineered Large Language Model, Deep Learning Workflow
for Materials Classification' published in *Materials Today*.  
arXiv link: [https://arxiv.org/abs/2401.17788](https://arxiv.org/abs/2401.17788)  
Materials Today link: [DOI: 10.1016/j.mattod.2024.08.028](https://www.sciencedirect.com/science/article/abs/pii/S1369702124002001).

There are five folders here, namely the metallic glasses database source data folder, the large language model folder, the classification model folder, the model interpretation and visualization folder and the supplemetary_data_for_revision folder.  

Here are some steps for setting up the configuration.
 
 ## Step1: Configure Python environment and libraries
 
 All code is recommended to run in a Python virtual environment. 

 If you have not installed Python before, it is recommended to follow the following link for installation: [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/)  

 To create and activate a new conda environment, use the following command:
 
 ```bash
 conda create --name bmg python=3.10
 conda activate bmg
 ```
 Then please use the following code to install the required Python packages:

 ```bash
 pip install -r requirements.txt
 ```
 
 ## Step2: Register Gemini API from Google
If you also want to generate text data through Gemini, please apply for a free API from [Google Dev](https://ai.google.dev/) first.  

Then copy and paste it to the `.env` file in `llm` folder:

```bash
GOOGLE_API_KEY='xxxxx'
```


 ## Step3: Download Huggingface Pre-trained Models
 
 Our classification model is fine tuned from pre-trained models. So if you want to repeat the training process by yourself, at least you need to obtain the model files.

 You can directly load the model according to the [official guide](https://huggingface.co/learn/nlp-course/en/chapter2/3?fw=pt).

 In case you want to download a pre-trained model from Huggingface, use the following command:
 
 ```bash
 from huggingface_hub import snapshot_download
 snapshot_download(repo_id="xxx", local_dir="xxx")
 ```
 
 Replace `repo_id` and `local_dir` with the name of the model you want to download and the folder you want to store.

 repo_id of MatSciBERT: m3rg-iitd/matscibert   
 repo_id of Longformer: allenai/longformer-base-4096  
 repo_id of BERT: bert-base-cased  

 If you just want to do inference with MgBERT, you can use the model weights file in the checkpoint folder:

 ```bash
 cd classification_models/different_BERT/checkpoint
 ```

 and load it with the `inference_template` file in `interpretability_and_visualization` folder.

