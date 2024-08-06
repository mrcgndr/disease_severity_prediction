import torch

import disease_pred.models.backbones as backbones


class TestBackbones:
    def test_pretrained_vit(self):
        vit = backbones.PretrainedViT(
            in_channels=3, out_classes=10, output_layer="last_hidden", pretrained_name="google/vit-base-patch16-224"
        )

        assert vit(torch.randn(4, 3, 224, 224)).logits.shape == torch.Size([4, 10])

    def test_vit(self):
        params_vit = {"vit_type": "vit", "output_layer": "pooled"}
        vit = backbones.ViT(params=params_vit)
        assert vit(torch.randn(4, 3, 224, 224)).shape == torch.Size([4, 768])
        del vit

        params_pretrained_vit = {"vit_type": "vit", "output_layer": "pooled", "pretrained": True}
        pretrained_vit = backbones.ViT(params=params_pretrained_vit)
        assert pretrained_vit(torch.randn(4, 3, 224, 224)).shape == torch.Size([4, 768])
        del pretrained_vit

        params_deit = {"vit_type": "deit", "output_layer": "last_hidden"}
        deit = backbones.ViT(params=params_deit)
        assert deit(torch.randn(4, 3, 224, 224)).shape == torch.Size([4, 768])
        del deit

    def test_cnn(self):
        params = {"num_channels": 3}
        cnn = backbones.CNN(params=params)

        assert cnn(torch.randn(4, 3, 224, 224)).shape == torch.Size([4, 3200])

    def test_resnet(self):
        params = {"depth": 18, "num_channels": 3, "out_features": 64}
        resnet = backbones.ResNet(params=params)

        assert resnet(torch.randn(4, 3, 224, 224)).shape == torch.Size([4, 64])

    def test_vgg(self):
        params = {"depth": "11", "num_channels": 3, "out_features": 64}
        vgg = backbones.VGG(params=params)

        assert vgg(torch.randn(4, 3, 224, 224)).shape == torch.Size([4, 64])

    def test_cvae(self):
        params = {"image_size": 224, "num_channels": 3, "latent_dim": 128}
        cvae = backbones.CVAE(params=params)

        result = cvae(torch.randn(4, 3, 224, 224))

        assert result["mean"].shape == torch.Size([4, 128])
        assert result["log_var"].shape == torch.Size([4, 128])
        assert result["out"].shape == torch.Size([4, 3, 224, 224])
