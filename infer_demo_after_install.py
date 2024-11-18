from ml_decoder import inference_ml_decoder
from PIL import Image

model = inference_ml_decoder()
image_path = "ML_Decoder/pics/000000000885.jpg"
pil_img = Image.open(image_path)
preds = model(pil_img)
print(preds)
