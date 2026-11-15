"""只读探针v3：1:1 复刻 chat_cli 调用序列，逐步定位 incompatible 警告发射点。"""
import logging, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())
logging.getLogger("transformers").addHandler(Capture())

def cnt():
    return sum(1 for m in records if "incompatible" in m)

SRC = r"byt5_model"
tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForSeq2SeqLM.from_pretrained(SRC, dtype=torch.float32)
model.cuda()
print(f"[1] 加载完成: {cnt()}")

model.gradient_checkpointing_enable()
print(f"[2] gradient_checkpointing_enable: {cnt()}")

# chat_cli learn 模式启动: model.train()
model.train()
print(f"[3] model.train() (learn模式): {cnt()}")

enc = tok("你好", return_tensors="pt").input_ids.cuda()

# chat_cli 第一轮生成 (291-301)
model.eval()
model.config.use_cache = True
with torch.no_grad():
    out = model.generate(input_ids=enc, max_new_tokens=48, do_sample=False, num_beams=1, repetition_penalty=1.15)
model.config.use_cache = False
print(f"[4] 第一轮 generate 完成({out.shape[-1]} tok): {cnt()}")

# chat_cli 学习路径 (327-328)
model.train()
labels = tok("你好呀", return_tensors="pt").input_ids.cuda()
loss = model(input_ids=enc, labels=labels).loss
loss.backward()
print(f"[5] 学习 forward 完成: {cnt()}")
model.zero_grad(set_to_none=True)

# 第二轮生成
model.eval()
model.config.use_cache = True
with torch.no_grad():
    out2 = model.generate(input_ids=enc, max_new_tokens=48, do_sample=False, num_beams=1, repetition_penalty=1.15)
model.config.use_cache = False
print(f"[6] 第二轮 generate 完成: {cnt()}")

for m in records:
    if "incompatible" in m:
        print("警告原文:", m)
