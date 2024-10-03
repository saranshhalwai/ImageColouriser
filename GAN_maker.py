from torch import nn
from torch.nn import LeakyReLU
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# Data preparation

# Load and reshape grayscale (L) data
l_arr = np.reshape(np.load("data/l/gray_scale.npy"), (25000, 224, 224, 1))  # Grayscale images

# Load AB channel data (for color)
arr2 = np.load("data/ab/ab/ab1.npy")  # AB channels batch 1
arr3 = np.load("data/ab/ab/ab2.npy")  # AB channels batch 2
arr4 = np.load("data/ab/ab/ab3.npy")  # AB channels batch 3

# Concatenate AB channels from different batches along the 0th axis (batch dimension)
ab_arr = np.concatenate([arr2, arr3, arr4], axis=0)



# Normalize both l_arr (grayscale) and ab_arr (color channels) to float32 and range [-1, 1]
l_arr = (l_arr / 127.5).astype(np.float32) - 1  # Normalize grayscale
ab_arr = (ab_arr / 127.5).astype(np.float32) - 1  # Normalize AB channels

# Convert numpy arrays to PyTorch tensors and change shape
l_tensor = torch.from_numpy(l_arr).permute(0,3,1,2)
ab_tensor = torch.from_numpy(ab_arr).permute(0,3,1,2)

# Create TensorDataset (Input: L channel, Target: AB channels)
train_dataset = TensorDataset(l_tensor[:20000], ab_tensor[:20000])  # First 20000 samples for training
test_dataset = TensorDataset(l_tensor[20000:], ab_tensor[20000:])  # Remaining samples for testing

# Create DataLoader objects for training and testing with a smaller batch size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




# EncoderBlock = (convolution -> batch normalization -> leaky ReLU)
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, apply_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1)
        self.normalise = nn.BatchNorm2d(out_ch) if apply_batchnorm else nn.Identity()  # Use Identity if no BatchNorm
        self.relu = LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalise(x)
        return self.relu(x)




# Decoder block = (Transpose convolution-> interpolate(to fix shape) -> concat with skip connection -> batch normalisation -> ReLU)
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, apply_batchnorm=True):
        super().__init__()
        self.apply_batchnorm = apply_batchnorm
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)  # Use appropriate stride/padding to control size
        if self.apply_batchnorm:
            self.normalise = nn.BatchNorm2d(out_ch + skip_ch)
        self.relu = nn.ReLU()

    def forward(self, x, feat):
        x = self.conv(x)

        # Check if shapes match, otherwise interpolate
        if x.shape[2:] != feat.shape[2:]:
            x = F.interpolate(x, size=feat.shape[2:])  # Ensure sizes match

        # Concatenate along the channel dimension (dim=1)
        x = torch.cat([x, feat], dim=1)
        if self.apply_batchnorm:
            x = self.normalise(x)
        return self.relu(x)


# Final code of generator using encoding Encoder and Decoder blocks. Similar to U-net.
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder layers
        self.enc1 = EncoderBlock(1, 64, apply_batchnorm=False)  # First layer, no batchnorm
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 512)
        self.enc6 = EncoderBlock(512, 512, kernel_size=2)
        self.enc7 = EncoderBlock(512, 512, kernel_size=2)
        self.enc8 = EncoderBlock(512, 512, kernel_size=2, apply_batchnorm=False)  # Last encoder layer, no batchnorm

        # Decoder layers
        self.dec1 = DecoderBlock(512, 512, 512)
        self.dec2 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(1024, 512, 512)
        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec5 = DecoderBlock(1024, 256, 256)
        self.dec6 = DecoderBlock(512, 128, 128)
        self.dec7 = DecoderBlock(256, 64, 64, apply_batchnorm=False)  # Last decoder layer, no batchnorm

        self.final = nn.ConvTranspose2d(128, 2, 4, 2, 1)  # Output AB channels
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder forward pass with skip connections
        d1 = self.dec1(e8, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)

        output = self.final(d7)
        output = self.tanh(output)
        return output


# Discriminator code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.enc1=EncoderBlock(2,64)
        self.enc2=EncoderBlock(64,64)
        self.enc3=EncoderBlock(64,128)
        self.enc4=EncoderBlock(128,256)
        self.enc5=EncoderBlock(256,512)
        self.enc6=EncoderBlock(512,512, kernel_size=2)
        self.enc7=EncoderBlock(512,512, kernel_size=2)
        self.enc8=EncoderBlock(512,1, kernel_size=2)

        # Global average pooling to reduce spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)
        x = self.global_pool(x)  # Apply global pooling
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 1)
        return torch.sigmoid(x)  # Apply sigmoid to get probability





# Initialize models
generator = Generator().to("cuda")
discriminator = Discriminator().to("cuda")

# Use BCELoss for adversarial losses
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers with reduced β1
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

def train_gan(generator, discriminator, dataloader, gen_optimizer, disc_optimizer, num_epochs, lambda_l1=100, save_path="gan_model.pth"):
    adversarial_loss = nn.BCELoss()  # Binary cross-entropy for GAN
    l1_loss = nn.L1Loss()  # L1 loss for reconstruction

    for epoch in range(num_epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to("cuda"), y.to("cuda")

            # Train Discriminator
            disc_optimizer.zero_grad()

            # Generate fake images
            generated_images = generator(X)

            # Prepare labels
            real_labels = torch.full((y.size(0), 1), 0.9, device="cuda")  # Smooth real labels
            fake_labels = torch.full((y.size(0), 1), 0.0, device="cuda")  # Fake labels

            # Discriminator real loss
            real_input = y  # Use real AB channels
            real_output = discriminator(real_input)
            real_loss = adversarial_loss(real_output, real_labels)

            # Discriminator fake loss
            fake_input = generated_images.detach()  # Detach to avoid gradient flow to generator
            fake_output = discriminator(fake_input)
            fake_loss = adversarial_loss(fake_output, fake_labels)

            # Total discriminator loss
            disc_loss = (real_loss + fake_loss) / 2

            disc_loss.backward()
            disc_optimizer.step()

            # Train Generator
            gen_optimizer.zero_grad()

            # Generator adversarial loss (try to fool the discriminator)
            fake_output = discriminator(generated_images)
            gen_adv_loss = adversarial_loss(fake_output, real_labels)  # Use real_labels to pretend generated images are real

            # Generator L1 loss (reconstruction loss)
            gen_l1_loss = l1_loss(generated_images, y)  # Compare generated AB channels with real AB channels

            # Total generator loss
            gen_loss = gen_adv_loss + lambda_l1 * gen_l1_loss

            gen_loss.backward()
            gen_optimizer.step()

            if batch % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch}/{len(dataloader)}], "
                      f"Disc Loss: {disc_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}, "
                      f"Adv Loss: {gen_adv_loss.item():.4f}, L1 Loss: {gen_l1_loss.item():.4f}")

        # Prompt for saving the model after each epoch
        save_option = input(f"Do you want to save the model after epoch {epoch + 1}? (yes/no): ").strip().lower()
        if save_option == 'yes':
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
            }, save_path)
            print(f"Model saved at epoch {epoch+1}")

        # Prompt to change learning rate manually
        lr_change = input(f"Do you want to adjust the learning rate of the generator? (current LR: {gen_optimizer.param_groups[0]['lr']}) (yes/no): ").strip().lower()
        if lr_change == 'yes':
            new_lr = float(input("Enter new learning rate for the generator: ").strip())
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to {new_lr} for the generator.")


train_gan(generator, discriminator, train_loader, gen_optimizer, disc_optimizer, num_epochs=100)
