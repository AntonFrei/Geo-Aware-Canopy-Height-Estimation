import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from encoder import ENCODER_MAP, transform_coordinates, extract_epsg

class CoordInjectionModelWrapper(nn.Module):
    """
    Wrapper that handles coordinate injection into feature maps before final conv.
    """
    def __init__(self, base_model, coord_encoder_name, device):
        super().__init__()
        self.base_model = base_model
        self.coord_encoder_name = coord_encoder_name
        self.encoder_fn = ENCODER_MAP.get(coord_encoder_name)
        self.device = device
        self.coord_channels = self.encoder_fn.num_output_channels
        
        # Modify the segmentation head to accept coordinate channels
        self._modify_segmentation_head()
        
        # Store coordinates for current batch
        self.current_coords = None
    
    def _modify_segmentation_head(self):
        """Modify the segmentation head to accept coordinate channels."""
        if not hasattr(self.base_model, 'segmentation_head'):
            raise ValueError("Model does not have segmentation_head attribute")
        
        seg_head = self.base_model.segmentation_head
        
        # Handle different types of segmentation heads
        if isinstance(seg_head, nn.Sequential):
            # Find the first Conv2d layer in the sequential
            conv_layer = None
            conv_index = None
            for i, layer in enumerate(seg_head):
                if isinstance(layer, nn.Conv2d):
                    conv_layer = layer
                    conv_index = i
                    break
            
            if conv_layer is None:
                raise ValueError("No Conv2d found in segmentation head")
                
        elif isinstance(seg_head, nn.Conv2d):
            # Single Conv2d layer
            conv_layer = seg_head
            conv_index = None
        else:
            raise ValueError(f"Unsupported segmentation head type: {type(seg_head)}")
        
        # Store original info
        self.original_in_channels = conv_layer.in_channels
        
        print(f"Original segmentation head input channels: {self.original_in_channels}")
        print(f"Adding coordinate channels: {self.coord_channels}")
        print(f"New input channels: {self.original_in_channels + self.coord_channels}")
        
        # Create new conv layer with expanded input channels
        new_conv = nn.Conv2d(
            in_channels=self.original_in_channels + self.coord_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias is not None
        )
        
        # Copy weights for original channels
        with torch.no_grad():
            # Copy original weights
            new_conv.weight[:, :self.original_in_channels] = conv_layer.weight.clone()
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.clone()
            
            # Initialize coordinate channel weights to small values
            nn.init.normal_(new_conv.weight[:, self.original_in_channels:], std=0.01)
        
        # Replace the conv layer in the model
        if isinstance(self.base_model.segmentation_head, nn.Sequential):
            self.base_model.segmentation_head[conv_index] = new_conv
        else:
            self.base_model.segmentation_head = new_conv
            
        print(f"Successfully modified segmentation head!")
        print(f"New segmentation head weight shape: {new_conv.weight.shape}")
    
    def set_coordinates(self, coordinates):
        """Set coordinates for the current batch."""
        self.current_coords = coordinates
    
    def forward(self, x):
        """Forward pass with coordinate injection."""
        # Forward through encoder
        encoder_features = self.base_model.encoder(x)
        
        # Forward through decoder to get features before segmentation head
        decoder_features = self.base_model.decoder(*encoder_features)
        
        # THIS IS WHERE THE MAGIC HAPPENS: Inject coordinates
        if self.current_coords is not None:
            coord_maps = self._create_coordinate_maps(decoder_features.shape, self.current_coords)
            # Concatenate decoder features with coordinate maps
            enhanced_features = torch.cat([decoder_features, coord_maps], dim=1)
            #print(f"Enhanced features shape: {enhanced_features.shape} (decoder: {decoder_features.shape[1]}, coords: {coord_maps.shape[1]})")
        else:
            enhanced_features = decoder_features
            print(f"No coordinates provided, using decoder features: {decoder_features.shape}")
        
        # Apply the modified segmentation head
        output = self.base_model.segmentation_head(enhanced_features)
        
        return output
    
    def _create_coordinate_maps(self, feature_shape, coordinates):
        """Create coordinate feature maps with same spatial dimensions as features."""
        B, _, H, W = feature_shape
        coord_maps = []
        
        for i in range(self.coord_channels):
            # Create a map filled with the i-th coordinate value for each sample in batch
            coord_map = torch.zeros(B, 1, H, W, device=self.device, dtype=torch.float32)
            for b in range(B):
                coord_map[b, 0, :, :] = coordinates[b, i]
            coord_maps.append(coord_map)
        
        coord_tensor = torch.cat(coord_maps, dim=1)
        #print(f"Created coordinate maps shape: {coord_tensor.shape}")
        return coord_tensor

    # Compatibility methods for your existing code
    def train(self, mode=True):
        """Override train mode."""
        super().train(mode)
        self.base_model.train(mode)
        return self
    
    def eval(self):
        """Override eval mode."""
        super().eval()
        self.base_model.eval()
        return self
    
    def to(self, device):
        """Override to() method."""
        super().to(device)
        self.base_model.to(device)
        self.device = device
        return self
    
    def parameters(self):
        """Return all parameters."""
        return self.base_model.parameters()
    
    def state_dict(self):
        """Return state dict."""
        return self.base_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        return self.base_model.load_state_dict(state_dict)