import torch
import random
import os
import numpy as np
import pandas as pd

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
# 设置随机数种子
setup_seed(42)


lf_PATH = 'your longformer path'
from transformers import AutoTokenizer, AutoModel, AutoConfig
config = AutoConfig.from_pretrained(lf_PATH)
tokenizer = AutoTokenizer.from_pretrained(lf_PATH)
model = AutoModel.from_pretrained(lf_PATH, config=config)


def find_text(composition):
    file_path = os.path.join('../../llm/description', composition + '.txt')
    with open(file_path, 'r') as file:
        text = file.read()
    return text


df = pd.read_csv('../../original_data/unique_compositions.csv')

labels = {'BMG': 0,
          'Ribbon': 1,
          'NR': 2
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [torch.tensor(labels[label]) for label in df['glass_forming_category']]
        self.texts = [tokenizer(find_text(composition),
                                padding='max_length', 
                                max_length = 1200, 
                                truncation=True,
                                return_tensors="pt") 
                      for composition in df['composition']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y



from torch.utils.data import DataLoader, Subset

# 创建自定义数据集对象
dataset = Dataset(df)

# 将数据按类别划分为不同的子集
class_indices = {
    label: df[df['glass_forming_category'] == category].index.tolist()
    for category, label in labels.items()
}

# 划分训练集和测试集的比例或样本数量
train_ratio = 0.8  # 训练集比例

train_indices = []
test_indices = []

# 对每个类别进行分层抽样
for class_label, indices in class_indices.items():
    # 计算当前类别的样本数量和训练集数量
    class_size = len(indices)
    train_size = int(train_ratio * class_size)
    # 随机从indices里抽样train_size
    random.shuffle(indices)
    train_indices.extend(indices[:train_size])
    test_indices.extend(indices[train_size:])

# 创建训练集和测试集的子集对象
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)



from torch import nn

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer



from torch.optim import Adam
from tqdm import tqdm

def train(model, train_data, test_data, batch_size, learning_rate, epochs):
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, eps = 1e-8)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
      # 进度条函数tqdm
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
                output = model(input_id, mask)

                # 计算损失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
        # 模型更新
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_test = 0
            total_loss_test = 0
        # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for test_input, test_label in test_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                    test_label = test_label.to(device)
                    mask = test_input['attention_mask'].to(device)
                    input_id = test_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, test_label)
                    total_loss_test += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == test_label).sum().item()
                    total_acc_test += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Test Loss: {total_loss_test / len(test_data): .3f} 
              | Test Accuracy: {total_acc_test / len(test_data): .3f}''')


EPOCHS = xx
model = BertClassifier()
LR = xx
batch_size = xx
train(model, train_dataset, test_dataset, batch_size, LR, EPOCHS)
torch.save(model.state_dict(), 'lfbert_classifier_weights.pth')

