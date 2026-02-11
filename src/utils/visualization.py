import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Use non-interactive backend
matplotlib.use('Agg')

def get_latent_representations(model, X_num, X_cat, batch_size=256):
    """Get latent representations from model (encoder output z)."""
    model.eval()
    latents = []
    
    n_samples = len(X_num)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_num = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32).to(device)
            batch_cat = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long).to(device)
            
            z = model.encoder(batch_num, batch_cat)
            latents.append(z.cpu().numpy())
    
    return np.concatenate(latents, axis=0)


def get_pspace_representations(model, X_num, X_cat, batch_size=256):
    """Get P-Space representations from model (Phase 2: z -> projector -> coordinates @ prototypes)."""
    model.eval()
    pspace_list = []
    coords_list = []
    
    n_samples = len(X_num)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_num = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32).to(device)
            batch_cat = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long).to(device)
            
            z = model.encoder(batch_num, batch_cat)
            coordinates = model.projector(z)
            p_space = model.global_prototype_layer(coordinates)
            
            pspace_list.append(p_space.cpu().numpy())
            coords_list.append(coordinates.cpu().numpy())
    
    return np.concatenate(pspace_list, axis=0), np.concatenate(coords_list, axis=0)


def _compute_tsne_embeddings(
    data_vectors, proto_vectors, perplexity=30, random_state=42
):
    """Compute t-SNE embeddings for data + prototypes."""
    n_prototypes = len(proto_vectors)
    all_vectors = np.vstack([data_vectors, proto_vectors])
    
    print(f"  Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        learning_rate='auto',
        init='pca'
    )
    embeddings = tsne.fit_transform(all_vectors)
    
    data_embeddings = embeddings[:-n_prototypes]
    proto_embeddings = embeddings[-n_prototypes:]
    
    return data_embeddings, proto_embeddings


    # Print save message
    print(f"  Saved to {output_path}")

def _overlay_images(ax, embeddings, images, zoom=0.5):
    """Overlay images on the plot at given coordinates."""
    for x, y, img in zip(embeddings[:, 0], embeddings[:, 1], images):
        im = OffsetImage(img, zoom=zoom, cmap='gray')
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=True,
                            pad=0.1, bboxprops=dict(facecolor='white', alpha=0.8))
        ax.add_artist(ab)

def _plot_tsne_embeddings(
    data_embeddings, proto_embeddings, labels, output_path, title, proto_label,
    proto_images=None, sample_images=None, sample_embeddings=None
):
    """Plot pre-computed t-SNE embeddings with given labels."""
    # Create plot
    print(f"  Creating plot: {output_path}")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Handle labels dynamically
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Use tab10/tab20 for multiclass
    cmap_scatter = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20', n_classes)
    
    # Define custom colors
    custom_colors = {
        0: '#2ecc71',   # Green
        1: '#e74c3c',   # Red
        0.0: '#2ecc71',
        1.0: '#e74c3c',
    }
    
    # Plot Data points
    for i, label_val in enumerate(unique_labels):
        if label_val in custom_colors:
            color = custom_colors[label_val]
        else:
            color = cmap_scatter(i)
        
        mask = labels == label_val
        ax.scatter(
            data_embeddings[mask, 0],
            data_embeddings[mask, 1],
            color=color,
            label=f"{label_val}",
            alpha=0.4, # Lower alpha for better visibility of overlays
            edgecolors='none',
            s=20
        )
    
    # Prototypes (Markers hidden if images are present, otherwise stars)
    if proto_images is None:
        ax.scatter(
            proto_embeddings[:, 0],
            proto_embeddings[:, 1],
            c='#9b59b6', # Keep prototypes distinct purple
            marker='*',
            s=400,
            edgecolors='white',
            linewidths=2,
            label=proto_label,
            zorder=100
        )
    else:
        # Just add a dummy invisible point for legend
        ax.scatter(
            [], [], 
            c='#9b59b6', marker='*', s=400, label=proto_label
        )

    # Overlay Prototype Images
    if proto_images is not None:
        # Increase zoom for prototypes (Make them BIG)
        _overlay_images(ax, proto_embeddings, proto_images, zoom=1.5)
    
    # Overlay Random Sample Images
    if sample_images is not None and sample_embeddings is not None:
        _overlay_images(ax, sample_embeddings, sample_images, zoom=0.5)
        
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def create_tsne_visualization(
    model, 
    train_num, train_cat, 
    labels_dict: dict, # Dictionary map: {'Class': labels, 'Alpha': labels, ...}
    output_dir: str = '.',
    n_samples: int = 5000,
    perplexity: int = 30,
    random_state: int = 42,
    file_prefix: str = 'tsne_visualization'
):
    """
    Create t-SNE visualization(s). 
    Computes indices and t-SNE embeddings ONCE, then generates multiple plots 
    colored by different labels provided in labels_dict.
    """
    print(f"Sampling {n_samples} points for visualization...")
    
    np.random.seed(random_state)
    indices = np.random.choice(len(train_num), min(n_samples, len(train_num)), replace=False)
    
    sampled_num = train_num[indices]
    sampled_cat = train_cat[indices]
    
    # Prepare all label sets
    sampled_labels_dict = {k: v[indices] for k, v in labels_dict.items()}
    
    device = next(model.parameters()).device
    
    # --- Phase 1 Visualization ---
    if model.phase == 1:
        print("Phase 1 detected: Visualizing Latent Z Space...")
        z = get_latent_representations(model, sampled_num, sampled_cat)
        
        print("  Initializing temporary prototypes via KMeans for visualization...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=model.n_global_prototypes, random_state=random_state, n_init='auto')
        kmeans.fit(z)
        centers = kmeans.cluster_centers_
        
        # FIND NEAREST NEIGHBORS (Real Data) for each center
        print("  Finding nearest actual data points to cluster centers...")
        from sklearn.metrics import pairwise_distances_argmin_min
        closest_indices, _ = pairwise_distances_argmin_min(centers, z)
        
        # Use the actual data points as prototypes
        prototypes = z[closest_indices]
        
        # Get the actual images for these 'prototypes'
        proto_num = sampled_num[closest_indices]
        # Invert normalization
        proto_imgs = (proto_num * 0.3081 + 0.1307).reshape(-1, 28, 28)
        proto_imgs = np.clip(proto_imgs, 0, 1)
        
        # 1. Compute Embeddings ONCE
        print(f"  Computing t-SNE embeddings for Phase 1 Z-Space...")
        data_emb, proto_emb = _compute_tsne_embeddings(z, prototypes, perplexity, random_state)
        
        # Prepare random sample overlay (same as before)
        print("  Prepared samples for visualization...")
        # Randomly select some data samples to overlay
        n_overlay = 20
        # Exclude the prototype indices to avoid duplicates if possible, usually fine though
        idx_overlay = np.random.choice(len(data_emb), min(n_overlay, len(data_emb)), replace=False)
        sample_embeddings = data_emb[idx_overlay]
        
        # Get their images
        sample_num = sampled_num[idx_overlay]
        sample_imgs = (sample_num * 0.3081 + 0.1307).reshape(-1, 28, 28)
        sample_imgs = np.clip(sample_imgs, 0, 1)

        # 2. Plot for each label set
        for label_name, labels in sampled_labels_dict.items():
            fname = f"{file_prefix}_{label_name.lower()}.png"
            full_path = f"{output_dir}/{fname}"
            full_path = full_path.replace('//', '/')
            
            _plot_tsne_embeddings(
                data_emb, proto_emb, labels,
                output_path=full_path,
                title=f'Phase 1: Latent Space ({label_name})',
                proto_label='Prototypes (Nearest Real Data)',
                proto_images=proto_imgs, # Real images!
                sample_images=sample_imgs,
                sample_embeddings=sample_embeddings
            )
        return

    # --- Phase 2: Dual Visualization ---
    print("Phase 2 detected: Generating seperate plots for Latent Z and P-Space...")
    
    # 1. P-Space Visualization (Projected Space)
    print(">>> Generating P-Space Visualization...")
    p_space, coords = get_pspace_representations(model, sampled_num, sampled_cat)
    
    n_prototypes = model.n_global_prototypes
    identity_coords = torch.eye(n_prototypes, device=device)
    with torch.no_grad():
        pspace_prototypes = model.global_prototype_layer(identity_coords).cpu().numpy()
        
    # Compute P-Space Embeddings ONCE
    print(f"  Computing t-SNE embeddings for Phase 2 P-Space...")
    p_data_emb, p_proto_emb = _compute_tsne_embeddings(p_space, pspace_prototypes, perplexity, random_state)
    
    # Plot P-Space for each label set
    for label_name, labels in sampled_labels_dict.items():
        fname = f"{file_prefix}_pspace_{label_name.lower()}.png"
        full_path = f"{output_dir}/{fname}".replace('//', '/')
        
        _plot_tsne_embeddings(
            p_data_emb, p_proto_emb, labels,
            output_path=full_path,
            title=f'Phase 2: P-Space ({label_name})',
            proto_label='Global Prototype Directions'
        )
    
    
    # 2. Latent Z Visualization (Encoder Output)
    print(">>> Generating Latent Z Visualization...")
    z = get_latent_representations(model, sampled_num, sampled_cat)
    raw_prototypes = model.global_prototype_layer.prototypes.detach().cpu().numpy()
    
    # Compute Z-Space Embeddings ONCE
    print(f"  Computing t-SNE embeddings for Phase 2 Z-Space...")
    z_data_emb, z_proto_emb = _compute_tsne_embeddings(z, raw_prototypes, perplexity, random_state)
    
    # Decode Prototypes for Z-Space
    print("  Decoding prototypes for visualization...")
    device = next(model.parameters()).device
    with torch.no_grad():
        proto_z = model.global_prototype_layer.prototypes
        num_recon, _ = model.decoder(proto_z) # Decode prototypes
        proto_imgs = num_recon.cpu().numpy().reshape(-1, 28, 28)
        
        # Randomly select some data samples to overlay
        n_overlay = 20
        idx_overlay = np.random.choice(len(z_data_emb), n_overlay, replace=False)
        sample_embeddings = z_data_emb[idx_overlay]
        
        # Get their images
        sample_num = sampled_num[idx_overlay]
        # Invert normalization for visualization (approximate)
        # x * std + mean. 0.3081, 0.1307
        sample_imgs = (sample_num * 0.3081 + 0.1307).reshape(-1, 28, 28)
        sample_imgs = np.clip(sample_imgs, 0, 1)
        
        # Invert prototypes too
        proto_imgs = (proto_imgs * 0.3081 + 0.1307)
        proto_imgs = np.clip(proto_imgs, 0, 1)

    # Plot Z-Space for each label set
    for label_name, labels in sampled_labels_dict.items():
        fname = f"{file_prefix}_z_{label_name.lower()}.png"
        full_path = f"{output_dir}/{fname}".replace('//', '/')
        
        _plot_tsne_embeddings(
            z_data_emb, z_proto_emb, labels,
            output_path=full_path,
            title=f'Phase 2: Latent Z Space ({label_name})',
            proto_label='Global Prototypes',
            proto_images=proto_imgs, # Overlay prototype images
            sample_images=sample_imgs, # Overlay sample images
            sample_embeddings=sample_embeddings
        )
