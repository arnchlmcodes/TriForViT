import os

# Disable TensorFlow in transformers/peft to avoid protobuf conflict on Windows host
os.environ["USE_TF"] = "0"
