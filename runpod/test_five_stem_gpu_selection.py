import os
import unittest
from unittest.mock import Mock, patch

from benchmark_control import select_gpu
from global_storage import NoGpuCapacity, mig_regions
from gpu_profile import ADA, MIG24


class Tests(unittest.TestCase):
    def test_only_explicit_none_is_a_capacity_failure(self):
        api = Mock()
        api.call.return_value = {"price": {"secure": 0.69}, "availability": "NONE"}
        with self.assertRaises(NoGpuCapacity):
            mig_regions(api, MIG24)
        for level in (None, "HIGH", "unexpected"):
            api.call.return_value = {"price": {"secure": 0.69}, "availability": level}
            with self.assertRaises(RuntimeError) as caught:
                mig_regions(api, MIG24)
            self.assertNotIsInstance(caught.exception, NoGpuCapacity)

    def test_mig_is_preferred_even_with_fallback_enabled(self):
        with (
            patch.dict(os.environ, {"SPT_ALLOW_4090_FALLBACK": "true"}),
            patch("benchmark_control.mig_regions", return_value=["EUR-IS-2"]) as regions,
        ):
            self.assertEqual(select_gpu(Mock(), "five-stem"), (MIG24, 0.69, ["EUR-IS-2"]))
            self.assertEqual(regions.call_count, 1)

    def test_fallback_only_for_explicit_five_stem_and_capacity_failure(self):
        for profile, allow in (("bounded", "true"), ("five-stem", "false")):
            with (
                patch.dict(os.environ, {"SPT_ALLOW_4090_FALLBACK": allow}),
                patch("benchmark_control.mig_regions", side_effect=NoGpuCapacity()) as regions,
                self.assertRaises(NoGpuCapacity),
            ):
                select_gpu(Mock(), profile)
            self.assertEqual(regions.call_count, 1)
        with (
            patch.dict(os.environ, {"SPT_ALLOW_4090_FALLBACK": "true"}),
            patch(
                "benchmark_control.mig_regions", side_effect=[NoGpuCapacity(), ["US-PA-1"]]
            ) as regions,
        ):
            self.assertEqual(select_gpu(Mock(), "five-stem"), (ADA, 1.0, ["US-PA-1"]))
            self.assertEqual(regions.call_args.kwargs, {"rate_cap": 1.0})

    def test_transport_or_price_error_never_triggers_fallback(self):
        with (
            patch.dict(os.environ, {"SPT_ALLOW_4090_FALLBACK": "true"}),
            patch(
                "benchmark_control.mig_regions", side_effect=RuntimeError("invalid price")
            ) as regions,
            self.assertRaises(RuntimeError),
        ):
            select_gpu(Mock(), "five-stem")
        self.assertEqual(regions.call_count, 1)


if __name__ == "__main__":
    unittest.main()
