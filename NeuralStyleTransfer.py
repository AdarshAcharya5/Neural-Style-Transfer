import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.vgg
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.chosenconvlayers = {0, 5, 10, 19, 28}
        self.model = models.vgg19(weights=torchvision.models.vgg.VGG19_Weights).features[:29]

    def forward(self, x):
        selectedactivations = []
        for num, layer in enumerate(self.model):
            x = layer(x)
            if num in self.chosenconvlayers:
                selectedactivations.append(x)
        return selectedactivations

def load_image(image_name, image_size=356):
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image=loader(Image.open(image_name))
    return image.to(device)


content_img = load_image("Content.png")
style_img = load_image("Style.png")
generated_img = content_img.clone().requires_grad_(True) #Or noise
model = VGG19().to(device).eval()
lr = 0.001
alpha = 1
beta = 0.1
optimizer = optim.Adam([generated_img], lr=lr)

for step in range(6000):
    generated_features = model(generated_img)
    content_features = model(content_img)
    style_features = model(style_img)
    style_loss, content_loss = 0, 0
    for gen_feature, con_feature, sty_feature in zip(
        generated_features, content_features, style_features
    ):
        channels, height, width = gen_feature.shape

        content_loss += torch.mean((gen_feature-con_feature)**2)

        #gram matrix
        G = gen_feature.view(channels, height*width).mm(gen_feature.view(channels, height*width).t())
        A = sty_feature.view(channels, height*width).mm(sty_feature.view(channels, height*width).t())

        style_loss += torch.mean((G-A)**2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if not step % 500:
        print(f"\n Loss : {total_loss}")
        save_image(generated_img, "generated.png")
