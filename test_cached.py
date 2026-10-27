import unittest

import torch

from train_log.IFNet_HDv3 import CachedIFNet, IFNet, IFNetTwoBlock


class TestCachedInference(unittest.TestCase):
    def test_stream_matches_uncached_and_export(self):
        torch.manual_seed(2026)
        for model_type in (IFNet, IFNetTwoBlock):
            with self.subTest(model=model_type.__name__), torch.no_grad():
                model = model_type().eval()
                frames = [torch.rand(1, 3, 64, 64) for _ in range(3)]
                features = model.encode(frames[0])
                program = torch.export.export(
                    CachedIFNet(model), (frames[0], frames[1], features)
                ).module()
                exported_features = features
                for previous, current in zip(frames, frames[1:]):
                    expected = model(torch.cat((previous, current), 1))[2][-1]
                    image, features = model.forward_cached(previous, current, features)
                    exported_image, exported_features = program(
                        previous, current, exported_features
                    )
                    torch.testing.assert_close(image, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        features, model.encode(current), rtol=0, atol=0
                    )
                    torch.testing.assert_close(exported_image, image, rtol=0, atol=0)
                    torch.testing.assert_close(
                        exported_features, features, rtol=0, atol=0
                    )
                restarted, _ = model.forward_cached(
                    frames[1], frames[2], model.encode(frames[1])
                )
                torch.testing.assert_close(restarted, image, rtol=0, atol=0)

    def test_discarded_call_primes_cache(self):
        torch.manual_seed(2026)
        for model_type in (IFNet, IFNetTwoBlock):
            with self.subTest(model=model_type.__name__), torch.no_grad():
                model = model_type().eval()
                frames = [torch.rand(1, 3, 64, 64) for _ in range(3)]
                cached = CachedIFNet(model)
                seed = torch.zeros(1, 4, 64, 64)
                exported = torch.export.export(
                    cached, (frames[0], frames[1], seed)
                ).module()
                for implementation in (cached, exported):
                    for value in (0.0, 127.0):
                        _, features = implementation(
                            frames[0], frames[0], torch.full_like(seed, value)
                        )
                        torch.testing.assert_close(
                            features, model.encode(frames[0]), rtol=0, atol=0
                        )
                        for previous, current in zip(frames, frames[1:]):
                            expected = model(torch.cat((previous, current), 1))[2][-1]
                            image, features = implementation(
                                previous, current, features
                            )
                            torch.testing.assert_close(image, expected, rtol=0, atol=0)
                            torch.testing.assert_close(
                                features, model.encode(current), rtol=0, atol=0
                            )

    def test_cached_backward(self):
        model = IFNetTwoBlock()
        previous, current = (torch.rand(1, 3, 64, 64) for _ in range(2))
        features = model.encode(previous).detach().requires_grad_()
        image, _ = model.forward_cached(previous, current, features)
        image.mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(model.encode.cnn0.weight.grad)


if __name__ == "__main__":
    unittest.main()
