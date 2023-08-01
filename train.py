import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from models import Generator, Discriminator, LocalDiscriminator, GlobalDiscriminator

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Resize([32, 32])
])

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_split, val_split, test_split = 0.9, 0.05, 0.05
train_size = int(len(dataset) * train_split)
val_size = int(len(dataset) * val_split)
test_size = int(len(dataset) * test_split)

train, val, test = torch.utils.data.random_split(dataset, (train_size, val_size, test_size))
print(len(train), len(val), len(test))

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask(N, im_h, im_w, hole_h, hole_w, same_size=True):
    startY, startX = np.random.randint(0, im_h - hole_h, N), np.random.randint(0, im_w - hole_w, N)
    bounds = [(startY[i], startY[i] + hole_h, startX[i], startX[i] + hole_w) for i in range(N)]
    masks = np.zeros((N, 1, im_h, im_w), np.float32)

    for i in range(N):
        masks[i, 0, startY[i]: startY[i] + hole_h + 1, startX[i]: startX[i] + hole_w + 1] = 1

    return torch.tensor(masks), bounds


print(create_mask(2, 3, 3, 1, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, local_d, global_d):
        super(Discriminator, self).__init__()
        self.local_discriminator = local_d
        self.global_discriminator = global_d
        self.fc = nn.Linear(2048, 1)

    def forward(self, X, mask_bounds):
        X_local = self.local_discriminator(X, mask_bounds)
        X_global = self.global_discriminator(X)
        concated = torch.cat((X_local, X_global), dim=1)
        out = self.fc(concated)

        return out


def train_gan(g, d, train, val, g_optimizer, d_optimizer, params, masks_fn):
    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], num_workers=0, pin_memory=0)
    val_loader = torch.utils.data.DataLoader(val, params["val_batch_size"], shuffle=True, pin_memory=True)

    optimizer_g = g_optimizer
    optimizer_d = d_optimizer

    T_c, T_d = params["T_c"], params["T_d"]
    w = params["w"]

    for epoch in range(params["epochs"]):

        for batch, _ in train_loader:
            g.train()
            d.train()

            N = batch.shape[0]
            batch = batch

            masks_g, bounds_g = masks_fn(N)

            masks_g = masks_g

            batch_masked = batch * (1 - masks_g)
            batch_with_masks = torch.cat((batch_masked, masks_g[:, :1]), dim=1)
            fake = g(batch_with_masks)

            loss_mse = (((batch - fake) * masks_g) ** 2).sum() / masks_g.sum()

            if epoch < T_c:

                loss_g = loss_mse
                loss_g.backward()

                optimizer_g.step()
                optimizer_g.zero_grad()

            else:

                inpainted = batch.clone()

                for i in range(len(bounds_g)):
                    y1, y2, x1, x2 = bounds_g[i]
                    inpainted[i, :, y1:y2 + 1, x1:x2 + 1] = masks_g[i, :, y1:y2 + 1, x1:x2 + 1]

                batch_with_masks = torch.cat((inpainted, masks_g[:, :1]), dim=1)
                fake = g(batch_with_masks)

                loss = nn.CrossEntropyLoss()

                inpainted = torch.cat((inpainted, masks_g[:, :1]), dim=1)
                d_fake = d(inpainted.detach(), bounds_g)

                masks_d, bounds_d = masks_fn(N)
                masks_d = masks_d
                real = torch.cat((batch.clone(), masks_d[:, :1]), dim=1)
                d_real = d(real, bounds_d)

                loss_d_fake = loss(d_fake, torch.zeros_like(d_fake))
                loss_d_real = loss(d_real, torch.ones_like(d_real))
                loss_d = (loss_d_fake + loss_d_real) / 2
                loss_d.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()

                if epoch >= T_c + T_d:
                    inpainted = batch.clone()

                    for i in range(len(bounds_g)):
                        y1, y2, x1, x2 = bounds_g[i]
                        inpainted[i, :, y1:y2 + 1, x1:x2 + 1] = fake[i, :, y1:y2 + 1, x1:x2 + 1]

                    inpainted = torch.cat((inpainted, masks_g[:, :1]), dim=1)
                    d_fake = d(inpainted, bounds_g)

                    criterion = nn.MSELoss()
                    loss_d_fake = criterion(d_fake, torch.zeros_like(d_fake))

                    loss_g = loss_mse + w * ((d_fake - 1) ** 2).mean()

                    loss_g.backward()
                    optimizer_g.step()
                    optimizer_g.zero_grad()

            i = 0
            g.eval()
            for data, _ in val_loader:
                if i > 0:
                    break
                i += 1

                N = data.shape[0]
                data = data

                masks_g, bounds_g = masks_fn(N)

                data_masked = data.clone() * (1 - masks_g)
                data_with_masks = torch.cat((data_masked, masks_g[:, :1]), dim=1)
                fake = g(data_with_masks)

                print(torchvision.utils.make_grid(data_masked[:8], nrow=4, normalize=True).shape)
                grid = torchvision.utils.make_grid(data_masked[:8], nrow=4, normalize=True).permute(1, 2, 0).numpy()
                plt.imshow(grid)
                plt.axis("off")
                plt.show()

                for i in range(len(bounds_g)):
                    y1, y2, x1, x2 = bounds_g[i]
                    data[i, :, y1:y2 + 1, x1:x2 + 1] = fake[i, :, y1:y2 + 1, x1:x2 + 1]

                print(torchvision.utils.make_grid(data[:8], nrow=4, normalize=True).shape)
                grid = torchvision.utils.make_grid(data[:8], nrow=4, normalize=True).permute(1, 2, 0).numpy()
                plt.imshow(grid)
                plt.axis("off")
                plt.show()


global_d = GlobalDiscriminator(im_channels=3)
local_d = LocalDiscriminator(im_channels=3, region_size=32)
discriminator = Discriminator(local_d=local_d, global_d=global_d)

generator = Generator(im_channels=3)

train_params = {
    "w": 0.0005,
    "learning_rate_g": 1e-3,
    "learning_rate_d": 1e-4,
    "batch_size": 500,
    "val_batch_size": 1000,
    "T_c": 0,
    "T_d": 0,
}

train_params['epochs'] = 20 + train_params['T_c'] + train_params['T_d']


def gen_masks(N, ch=3):
    masks, bounds = create_mask(N, 32, 32, 5, 5, same_size=False)
    return masks, bounds


optimizer_g = torch.optim.Adam(generator.parameters(), lr=train_params['learning_rate_g'])
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=train_params['learning_rate_d'])

train_gan(generator, discriminator, train, val, optimizer_g, optimizer_d, train_params, gen_masks)
