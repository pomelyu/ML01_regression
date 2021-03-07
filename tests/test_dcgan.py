import mlconfig
import torch

import context  # pylint: disable=unused-import
import src.models.dcgan  # pylint: disable=unused-import


def test_dcgan():
    LATENT_SIZE = 64
    BATCH_SIZE = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_generator = mlconfig.config.Config({
        "name": "DCGAN_G",
        "latent_size": 64,
        "out_nc": 3,
        "nd": 16,
        "num_layers": 6,
    })

    generator = config_generator().to(device)

    latent = torch.rand(BATCH_SIZE, LATENT_SIZE, device=device)

    fake = generator(latent)

    assert fake.shape[:2] == (BATCH_SIZE, 3)

    config_discriminator = mlconfig.config.Config({
        "name": "DCGAN_D",
        "in_nc": 3,
        "nd": 16,
        "num_layers": 5,
    })

    discriminator = config_discriminator().to(device)

    logit_fake = discriminator(fake)

    assert logit_fake.shape[:2] == (BATCH_SIZE, 1)

    print(f"image size: {fake.shape[1:]}, logit_size: {logit_fake.shape[1:]}")
