from transformers import BeitForImageClassification, BeitFeatureExtractor

# Load the pre-trained BEiT model and feature extractor from Microsoft
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224')

# Save them locally
model.save_pretrained('beit_model')
feature_extractor.save_pretrained('beit_feature_extractor')

print("Model and feature extractor downloaded successfully!")
