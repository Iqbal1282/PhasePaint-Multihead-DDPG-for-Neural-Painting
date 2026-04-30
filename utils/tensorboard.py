import numpy as np
from PIL import Image
from io import BytesIO
import tensorboardX as tb
from tensorboardX.summary import Summary

class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def add_image(self, tag, img, step):
        summary = Summary()
        bio = BytesIO()

        # 1. Handle input types and convert to PIL Image
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, Image.Image):
            pass
        else: 
            # If img is a numpy array or torch tensor
            if hasattr(img, 'detach'): # Handle torch tensors
                img = img.detach().cpu().numpy()
            
            # 2. Fix "Mode F": Scale float (0.0-1.0) to int (0-255)
            if img.dtype == np.float32 or img.dtype == np.float64:
                # Only scale if the max value is <= 1.0 (typical for neural net outputs)
                if img.max() <= 1.01: 
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            img = Image.fromarray(img)

        # 3. Ensure image is RGB if it's currently CMYK or other incompatible modes
        if img.mode != 'RGB' and img.mode != 'L':
            img = img.convert('RGB')

        img.save(bio, format="png")
        image_summary = Summary.Image(encoded_image_string=bio.getvalue())
        summary.value.add(tag=tag, image=image_summary)
        self.summary_writer.add_summary(summary, global_step=step)

    def add_scalar(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)