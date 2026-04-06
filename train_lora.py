import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

# ==================== ???????? ====================
# ???????ùù??
BASE_MODEL_ID = "./local_models/sdxl-inpainting"  # ???????ùù??

# ?????ùù??
DATASET_DIR = "./dataset"  # ??????dataset/train/input ?? dataset/train/target

# ???ùù??
OUTPUT_DIR = "./lora_photo_restoration"

# ???????
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ???????
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
MAX_STEPS = 1000
SAVE_STEPS = 200
LOGGING_STEPS = 50

# LoRA ????
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ==================== ??????? ====================
class PhotoRestorationDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.input_dir = os.path.join(dataset_dir, "input")
        self.target_dir = os.path.join(dataset_dir, "target")
        self.image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        # ??????
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        
        # ?????????????????????????????????????
        # ???????ùù???????????????????????
        mask = Image.new("L", input_image.size, 255)  # ???????
        
        # ???ùù
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
            mask = self.transform(mask)
        
        return {
            "input": input_image,
            "target": target_image,
            "mask": mask,
            "filename": filename
        }

# ==================== ????ùù ====================
def transform(image):
    """???????ùù"""
    # ??????ùù? 512x512??SDXL ???????????
    image = image.resize((512, 512))
    # ????????
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return image

# ==================== ??????? ====================
def train():
    # ?????????
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ?????????
    print("?????????...")
    train_dataset = PhotoRestorationDataset(os.path.join(DATASET_DIR, "train"), transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ??????????
    print("??????????...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True
    )
    pipe = pipe.to(DEVICE)
    
    # ???? LoRA
    print("???? LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["unet"],
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    
    # ??? LoRA ????? UNet
    unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet = unet
    
    # ?????
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    
    # ?????????
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=MAX_STEPS
    )
    
    # ??????
    print("??????...")
    global_step = 0
    for epoch in range(NUM_TRAIN_EPOCHS):
        epoch_loss = 0.0
        
        for batch in tqdm(train_dataloader):
            # ???????
            input_images = batch["input"].to(DEVICE, dtype=DTYPE)
            target_images = batch["target"].to(DEVICE, dtype=DTYPE)
            masks = batch["mask"].to(DEVICE, dtype=DTYPE)
            
            # ????????
            noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            noise = torch.randn_like(target_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (input_images.shape[0],), device=DEVICE)
            
            # ???????????????
            noisy_images = noise_scheduler.add_noise(target_images, noise, timesteps)
            
            # ????
            with torch.no_grad():
                latents = pipe.vae.encode(target_images).latent_dist.sample()
                latents = latents * 0.18215  # ????????
            
            # ???????
            model_pred = pipe.unet(latents, timesteps, encoder_hidden_states=None).sample
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            
            # ?????
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # ??????
            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # ???
                if (global_step + 1) % LOGGING_STEPS == 0:
                    print(f"Step {global_step+1}, Loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}")
                
                # ???????
                if (global_step + 1) % SAVE_STEPS == 0:
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step+1}")
                    unet.save_pretrained(save_path)
                    print(f"???????ùù {save_path}")
                
                global_step += 1
                
                # ???????
                if global_step >= MAX_STEPS:
                    break
        
        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        if global_step >= MAX_STEPS:
            break
    
    # ???????????
    final_save_path = os.path.join(OUTPUT_DIR, "final")
    unet.save_pretrained(final_save_path)
    print(f"???????????ùù {final_save_path}")

# ==================== ??? LoRA ??? ====================
def use_lora_model():
    """????????? LoRA ??????????"""
    # ??????????
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True
    )
    pipe = pipe.to(DEVICE)
    
    # ???? LoRA ???
    from peft import PeftModel
    lora_path = os.path.join(OUTPUT_DIR, "final")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    # ???????
    input_image = Image.open("./old_photos/sample.jpg").convert("RGB")
    mask = Image.new("L", input_image.size, 255)  # ???????
    
    result = pipe(
        prompt="restored old photo, clear, sharp, vibrant colors",
        image=input_image,
        mask_image=mask,
        strength=0.7,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    result.save("./restored_photos/sample_restored.jpg")
    print("????????????????")

# ==================== ?????? ====================
if __name__ == "__main__":
    # ??????
    train()
    
    # ???????????????????
    # use_lora_model()
