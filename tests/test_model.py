import torch
from PIL import Image

from malaria_cnn.inference import predict_image, preprocess_image
from malaria_cnn.model import MalariaCNN, count_parameters


def test_model_outputs_two_logits_for_rgb_batch():
    torch.manual_seed(1)
    model = MalariaCNN().eval()
    output = model(torch.zeros(2, 3, 96, 96))
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert 0 < count_parameters(model) < 1_000_000


def test_preprocessing_and_prediction_shape():
    image = Image.new("RGB", (75, 110), color=(160, 80, 120))
    batch = preprocess_image(image, 96)
    assert batch.shape == (1, 3, 96, 96)
    assert float(batch.min()) >= 0 and float(batch.max()) <= 1
    model = MalariaCNN().eval()
    result = predict_image(model, image, image_size=96)
    assert result["prediction"] in {"Uninfected", "Parasitized"}
    assert 0 <= result["parasitized_score"] <= 1
    assert result["class_scores_are_calibrated_probabilities"] is False
