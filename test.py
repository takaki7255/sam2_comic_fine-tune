import inspect, transformers, os
from transformers import TrainingArguments

print("Transformers version :", transformers.__version__)
print("TrainingArguments in :", inspect.getfile(TrainingArguments))
print("Signature            :", inspect.signature(TrainingArguments.__init__))