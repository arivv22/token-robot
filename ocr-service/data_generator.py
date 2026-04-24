import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
from tqdm import tqdm

class TokenDataGenerator:
    def __init__(self, output_dir='data/training'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample tokens for training
        self.sample_tokens = [
            "22592297522702336675",
            "13420917670475678992",
            "12345678901234567890",
            "98765432109876543210",
            "55556666777788889999",
            "11112222333344445555",
            "99998888777766665555",
            "24681357902468135790",
            "13579246801357924680",
            "86420975318642097531"
        ]
        
    def generate_synthetic_token_image(self, token, noise_level=0.1):
        """Generate synthetic token image with various fonts and noise"""
        
        # Create blank image
        width, height = 800, 600
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use different fonts
        try:
            # Try system fonts
            fonts = [
                ImageFont.truetype("arial.ttf", 40),
                ImageFont.truetype("times.ttf", 40),
                ImageFont.truetype("courier.ttf", 40)
            ]
            font = random.choice(fonts)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Format token with spaces
        formatted_token = ' '.join([token[i:i+4] for i in range(0, len(token), 4)])
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), formatted_token, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), formatted_token, fill='black', font=font)
        
        # Convert to numpy array for noise addition
        img_array = np.array(img)
        
        # Add noise
        if noise_level > 0:
            # Gaussian noise
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = img_array + noise
            img_array = np.clip(img_array, 0, 255)
            
            # Random brightness/contrast
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                img_array = img_array * brightness
                img_array = np.clip(img_array, 0, 255)
        
        # Convert back to PIL
        img = Image.fromarray(img_array.astype(np.uint8))
        
        return img
    
    def add_background_noise(self, img):
        """Add background noise to simulate real screenshots"""
        img_array = np.array(img)
        
        # Add random lines/shapes to simulate UI elements
        h, w = img_array.shape[:2]
        
        # Random horizontal lines
        for _ in range(random.randint(0, 3)):
            y = random.randint(0, h)
            color = random.randint(150, 200)
            cv2.line(img_array, (0, y), (w, y), (color, color, color), 1)
        
        # Random rectangles
        for _ in range(random.randint(0, 2)):
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(20, 100)
            color = random.randint(200, 240)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (color, color, color), -1)
        
        return Image.fromarray(img_array)
    
    def generate_dataset(self, num_samples=100):
        """Generate synthetic training dataset"""
        print(f"🔄 Generating {num_samples} synthetic training images...")
        
        labels = {}
        
        for i in tqdm(range(num_samples)):
            # Random token
            token = random.choice(self.sample_tokens)
            
            # Generate image with varying noise levels
            noise_level = random.uniform(0.05, 0.2)
            img = self.generate_synthetic_token_image(token, noise_level)
            
            # Add background noise
            if random.random() > 0.3:
                img = self.add_background_noise(img)
            
            # Save image
            filename = f"token_{i:04d}.jpg"
            filepath = self.output_dir / filename
            img.save(filepath, 'JPEG', quality=85)
            
            # Store label
            labels[filename] = token
        
        # Save labels file
        labels_path = self.output_dir / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"✅ Dataset generated successfully!")
        print(f"📁 Images saved to: {self.output_dir}")
        print(f"📋 Labels saved to: {labels_path}")
        print(f"📊 Total samples: {len(labels)}")
        
        return labels
    
    def create_real_dataset_from_existing(self, image_paths, labels):
        """Create dataset from existing images"""
        print(f"📁 Creating dataset from {len(image_paths)} existing images...")
        
        labels_dict = {}
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            # Copy image to training directory
            filename = f"real_{i:04d}.jpg"
            filepath = self.output_dir / filename
            
            # Load and save image
            img = Image.open(img_path)
            img.save(filepath, 'JPEG', quality=85)
            
            # Store label
            labels_dict[filename] = label
        
        # Save labels file
        labels_path = self.output_dir / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(labels_dict, f, indent=2)
        
        print(f"✅ Real dataset created successfully!")
        print(f"📁 Images saved to: {self.output_dir}")
        print(f"📋 Labels saved to: {labels_path}")
        
        return labels_dict

def main():
    """Main function to generate training data"""
    generator = TokenDataGenerator()
    
    # Generate synthetic dataset
    generator.generate_dataset(num_samples=200)
    
    # Also create dataset from existing images if available
    existing_images = [
        "/home/arip/Documents/token.jpg",
        "/home/arip/Documents/token2.jpg"
    ]
    
    existing_labels = [
        "22592297522702336675",
        "13420917670475678992"
    ]
    
    # Check if existing images exist
    valid_images = []
    valid_labels = []
    
    for img_path, label in zip(existing_images, existing_labels):
        if os.path.exists(img_path):
            valid_images.append(img_path)
            valid_labels.append(label)
    
    if valid_images:
        generator.create_real_dataset_from_existing(valid_images, valid_labels)

if __name__ == "__main__":
    from tqdm import tqdm
    main()
