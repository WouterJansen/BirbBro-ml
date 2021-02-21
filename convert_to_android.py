import torch
import torch.nn as nn
import torchvision as tv

MODEL = "results/model_Transfer_ep=17_acc=0.796875.pt"
NUM_CLASSES = 34
SIZE_IMAGE = 700

if __name__ == '__main__':
    # instantiate the model
    model = tv.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # use the chosen model snapshot
    model.load_state_dict(torch.load(MODEL))
    model.eval()


    # convert to TorchScript format
    input_tensor = torch.rand(1, 3, SIZE_IMAGE, SIZE_IMAGE)
    script_model = torch.jit.trace(model, input_tensor)
    script_model.save("birbronet.pt")