from model.third_party_transformer.modeling_dinov2_config import Dinov2Config
from model.third_party_transformer.modeling_dinov2 import Dinov2Backbone as Dinov2BackboneOurs
from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone
import torch#num_frames=5,
def create_mock_input(batch_size=2,  channels=3, height=256, width=256):
    images = torch.randn(batch_size,  channels, height, width)
    camera_intri = torch.randn(batch_size, 768)
    sample = {'images': images, 'camera_intri': camera_intri}
    return sample


# models = Dinov2Backbone.from_pretrained('facebook/dinov2-base')
sample = create_mock_input()

config = Dinov2Config(768,image_size=518,patch_size=14)
backbone = Dinov2BackboneOurs(config)
#print (backbone(sample['images'], sample['camera_intri']))


print ("finish initialization")
# 

# backbone.load_state_dict(models.state_dict(), strict=False)

print (backbone(sample['images'], sample['camera_intri']))

