import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import EmbeddingModel
import os

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 512

# Transforms must match training (minus augmentations)
val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Model
model = EmbeddingModel(embedding_size=EMBEDDING_SIZE).to(DEVICE)
model.load_state_dict(torch.load("arcface_backbone.pth"))
model.eval()

# --- Helper Function: Get Vector for one image ---
def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_transforms(img).unsqueeze(0).to(DEVICE) # Add batch dim
    with torch.no_grad():
        embedding = model(img_tensor) # Output is already normalized by model
    return embedding

# --- Step A: Registration (Build Database) ---
# In a real app, you load these from a folder and save to a dictionary/JSON
reference_images = {
    "Screw_Type_A": ["./dataset/train/class_A/img1.jpg", "./dataset/train/class_A/img2.jpg"],
    "Screw_Type_B": ["./dataset/train/class_B/img1.jpg"]
}

database = {}

print("Building Database Prototypes...")
for class_name, img_paths in reference_images.items():
    embeddings_list = []
    for path in img_paths:
        if os.path.exists(path):
            emb = get_embedding(path)
            embeddings_list.append(emb)
    
    # Create Prototype: Average all embeddings for this class
    if embeddings_list:
        stacked_emb = torch.cat(embeddings_list, dim=0)
        prototype = torch.mean(stacked_emb, dim=0, keepdim=True)
        # Normalize the prototype again after averaging
        prototype = F.normalize(prototype, p=2, dim=1)
        database[class_name] = prototype

print(f"Database built with {len(database)} classes.")

# --- Step B: Inference (Predicting a new unknown crop) ---
def identify_object(crop_path):
    # 1. Get embedding for the unknown crop
    query_emb = get_embedding(crop_path)
    
    best_score = -1.0
    best_class = "Unknown"
    
    # 2. Compare against all database prototypes
    for name, prototype in database.items():
        # Calculate Cosine Similarity
        # Since vectors are normalized, Cosine Sim = Dot Product
        score = torch.mm(query_emb, prototype.t()).item()
        
        print(f"Score for {name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_class = name
            
    # 3. Thresholding (Optional but recommended)
    threshold = 0.5  # Adjust this based on testing
    if best_score < threshold:
        return "Unknown", best_score
        
    return best_class, best_score

# --- Test it ---
test_image = "./dataset/val/class_A/test_img.jpg" # Replace with your cropped image path
if os.path.exists(test_image):
    predicted_class, confidence = identify_object(test_image)
    print(f"\nFinal Result: {predicted_class} (Confidence: {confidence:.4f})")