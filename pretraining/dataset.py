import torch
import os
from glob import glob
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, file_paths, max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS as padding

        self.max_length = max_length
        self.data = []

        for file_path in file_paths:
            print(f"Processing {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors="pt")["input_ids"].squeeze(0)
            chunks = [tokens[i:i + self.max_length] for i in range(0, len(tokens), self.max_length)]
            padding_length = self.max_length - len(chunks[-1])
            chunks[-1] = torch.cat([chunks[-1], torch.full((padding_length,), self.tokenizer.pad_token_id)])
            for c in chunks:
                assert(c.shape == torch.Size([1024]))
            self.data.extend(chunks)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/train_10M"))
    train_paths = glob(os.path.join(train_dir, "*.train"))

    train_dataset = TextDataset(train_paths)
    train_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./train.pt"))
    
    torch.save(train_dataset, train_dataset_path)

    dev_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/dev"))
    dev_paths = glob(os.path.join(dev_dir, "*.dev"))

    dev_dataset = TextDataset(dev_paths)
    dev_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./dev.pt"))
    
    torch.save(dev_dataset, dev_dataset_path)

    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/test"))
    test_paths = glob(os.path.join(test_dir, "*.test"))

    train_dataset = TextDataset(test_paths)
    train_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./test.pt"))
    
    torch.save(train_dataset, train_dataset_path)
    