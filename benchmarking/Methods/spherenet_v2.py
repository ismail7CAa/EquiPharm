
# Import the SphereNet model class from DIG library 
from dig.threedgraph.method import SphereNet  

# Initialize a SphereNet model:
def SphereNet_model():
    model = SphereNet(               # SphereNet is a 3D Graph Neural Network that incorporates distances, angles, and torsions between atoms. It’s especially effective for molecular property prediction tasks (like QM9).
        energy_and_force=False,      # If True, the model predicts both energy & forces (for MD); here, only energy/property
        cutoff=5.0,                  # Maximum distance (Å) for considering atom interactions (neighbors)
        num_layers=4,                # Number of message-passing layers (depth of the GNN)
        hidden_channels=128,         # Hidden dimension size of node embeddings
        out_channels=19,             # Output dimension (e.g., 1 for scalar regression like energy)
        int_emb_size=64,             # Size of intermediate embedding for interactions
        basis_emb_size_dist=8,       # Embedding size for distance basis functions
        basis_emb_size_angle=8,      # Embedding size for angular basis functions
        basis_emb_size_torsion=8,    # Embedding size for torsion basis functions
        out_emb_channels=256,        # Output embedding size before final prediction layer
        num_spherical=3,             # Number of spherical harmonics components used (angular resolution)
        num_radial=6,                # Number of radial basis functions used (distance resolution)
        envelope_exponent=5,         # Smooth cutoff envelope exponent for distance weighting
        num_before_skip=1,           # Layers before skip connection in each block
        num_after_skip=2,            # Layers after skip connection
        num_output_layers=3,         # MLP layers after GNN for final output
        use_node_features=True       # If True, initial node (atom) features are included
    )
    return model