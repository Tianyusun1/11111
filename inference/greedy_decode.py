# File: inference/greedy_decode.py (V8.0: Data-Driven Gestalt & Relaxed Constraints)

import torch
import numpy as np
import random
import os

# --- Import KG & Location ---
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Error] Could not import PoetryKnowledgeGraph. Make sure models/kg.py is accessible.")
    PoetryKnowledgeGraph = None

# Import Location Generator
try:
    from models.location import LocationSignalGenerator
except ImportError:
    print("[Error] Could not import LocationSignalGenerator. Make sure models/location.py is accessible.")
    LocationSignalGenerator = None

# [NEW] Import Integrated Visualization Tool
try:
    from data.visualize import draw_integrated_heatmap
except ImportError:
    draw_integrated_heatmap = None
# -----------------

# [V8.0] Relaxed Shape Priors (Data-Driven Approach)
# Since the model now learns physical properties from real images (OpenCV extraction),
# we remove strict 'max' constraints to allow the model to predict extreme shapes (e.g., tall mountains, long rivers).
# We only keep minimal 'min' constraints to prevent invisible objects.
CLASS_SHAPE_PRIORS = {
    2: {'min_w': 0.05, 'min_h': 0.05}, # Mountain
    3: {'min_w': 0.05, 'min_h': 0.02}, # Water
    4: {'min_w': 0.01, 'min_h': 0.02}, # People
    5: {'min_w': 0.02, 'min_h': 0.05}, # Tree
    6: {'min_w': 0.03, 'min_h': 0.03}, # Building
    7: {'min_w': 0.05, 'min_h': 0.02}, # Bridge
    8: {'min_w': 0.01, 'min_h': 0.01}, # Flower
    9: {'min_w': 0.01, 'min_h': 0.01}, # Bird
    10: {'min_w': 0.02, 'min_h': 0.02} # Animal
}

# Class ID Mapping
CLASS_ID_TO_NAME = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements=None, device='cuda', mode='greedy', top_k=3):
    """
    Query-Based decoding with Location Guidance & CVAE Diversity.
    [Updated V8.0] Supports 8-dim output (Coords + Gestalt) and relaxed constraints.
    """
    if PoetryKnowledgeGraph is None:
        return []

    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    
    # 1. Instantiate Components
    pkg = PoetryKnowledgeGraph()
    
    if LocationSignalGenerator is not None:
        location_gen = LocationSignalGenerator(grid_size=8)
    else:
        location_gen = None
    
    # 2. Extract KG Content
    kg_vector = pkg.extract_visual_feature_vector(poem)
    
    # Use torch.as_tensor to avoid warning
    kg_vector_t = torch.as_tensor(kg_vector)
    if kg_vector_t.device != device:
        kg_vector_t = kg_vector_t.cpu()
        
    existing_indices = torch.nonzero(kg_vector_t > 0).squeeze(1)
    raw_class_ids = (existing_indices + 2).tolist()
    
    if not raw_class_ids:
        return []
        
    # KG Quantity Expansion
    if hasattr(pkg, 'expand_ids_with_quantity'):
        kg_class_ids = pkg.expand_ids_with_quantity(raw_class_ids, poem)
    else:
        kg_class_ids = raw_class_ids
        
    if max_elements:
        kg_class_ids = kg_class_ids[:max_elements]
        
    # 3. Prepare Model Inputs
    kg_class_tensor = torch.tensor([kg_class_ids], dtype=torch.long).to(device)
    
    # Build Spatial Matrix
    try:
        kg_spatial_matrix_np = pkg.extract_spatial_matrix(poem, obj_ids=kg_class_ids)
    except TypeError:
        kg_spatial_matrix_np = pkg.extract_spatial_matrix(poem)
        
    kg_spatial_matrix = torch.as_tensor(kg_spatial_matrix_np, dtype=torch.long).unsqueeze(0).to(device)
    
    # === Generate Location Guidance Signals ===
    location_grids_tensor = None
    heatmap_layers = [] # For visualization
    
    if location_gen is not None:
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        grids_list = []
        
        for i, cls_id in enumerate(kg_class_ids):
            # Get spatial relations row/col
            if i < kg_spatial_matrix_np.shape[0]:
                row = kg_spatial_matrix_np[i]
                col = kg_spatial_matrix_np[:, i]
            else:
                row = np.zeros(len(kg_class_ids))
                col = np.zeros(len(kg_class_ids))
            
            # Infer location signal
            signal, current_occupancy = location_gen.infer_stateful_signal(
                i, row, col, current_occupancy, 
                mode=mode, top_k=top_k 
            )
            
            # Jitter
            if mode == 'sample' and random.random() < 0.6:
                shift_val = random.randint(-2, 2)
                signal = torch.roll(signal, shifts=shift_val, dims=1)
            
            grids_list.append(signal)
            
            if draw_integrated_heatmap is not None:
                heatmap_layers.append((signal.cpu().numpy(), int(cls_id)))
            
        location_grids_tensor = torch.stack(grids_list).unsqueeze(0).to(device)

        # Draw Heatmaps (Optional)
        if draw_integrated_heatmap is not None and len(heatmap_layers) > 0:
            safe_poem_name = "".join(x for x in poem if x.isalnum())[:10]
            save_dir = os.path.join("outputs", "heatmaps")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"integrated_{safe_poem_name}.png")
            draw_integrated_heatmap(heatmap_layers, poem, save_path)
    # ==========================================
    
    # 4. Forward Pass
    inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    padding_mask = torch.zeros(kg_class_tensor.shape, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        # Model returns: mu, logvar, dynamic_layout (8 dims), decoder_output
        _, _, pred_boxes, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kg_class_ids=kg_class_tensor, 
            padding_mask=padding_mask, 
            kg_spatial_matrix=kg_spatial_matrix, 
            location_grids=location_grids_tensor
        )
        
    # 5. Format Output (V8.0 Update)
    layout = []
    boxes_flat = pred_boxes[0].cpu().tolist()
    
    for cls_id, box in zip(kg_class_ids, boxes_flat):
        cid = int(cls_id)
        
        # [CRITICAL] Slice first 4 dims for coordinates, keep rest for Gestalt
        # box structure: [cx, cy, w, h, bx, by, rot, flow]
        cx, cy, w, h = box[:4] 
        
        # Minimal Shape Constraints (Sanity Check)
        w = max(w, 0.01)
        h = max(h, 0.01)
        
        # Apply only MIN constraints from priors, ignore MAX to allow data-driven shapes
        if cid in CLASS_SHAPE_PRIORS:
            prior = CLASS_SHAPE_PRIORS[cid]
            if 'min_w' in prior: w = max(w, prior['min_w'])
            if 'min_h' in prior: h = max(h, prior['min_h'])
        
        # [CRITICAL] Preserve 4-dim Gestalt Parameters
        if len(box) >= 8:
            gestalt_params = box[4:] # bx, by, rot, flow
        else:
            gestalt_params = [0.0, 0.0, 0.0, 0.0]

        # Reassemble: [cls, cx, cy, w, h, bx, by, rot, flow]
        full_item = [float(cls_id), cx, cy, w, h] + gestalt_params
        
        layout.append(tuple(full_item))
        
    return layout