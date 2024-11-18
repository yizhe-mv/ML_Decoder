from ml_decoder import MLDecoder
from PIL import Image

model = MLDecoder().cuda().half().eval()
image_path = "ML_Decoder/pics/000000000885.jpg"
pil_img = Image.open(image_path)
preds = model(pil_img)
print(preds)
