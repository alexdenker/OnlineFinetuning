

from torchvision.datasets import Flowers102
import torchvision.transforms as T
import torchvision.transforms.functional as F

def flowers102(root, split):
    # split = "train", "val", "test"
    image_size = 64
    transform = T.Compose(
        [
            T.Lambda(lambda img: F.center_crop(img, min(*img._size))),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),  # normalize to [-1, 1]
        ]
    )
    dataset = Flowers102(root=root, split=split, transform=transform, download=True)
    return dataset