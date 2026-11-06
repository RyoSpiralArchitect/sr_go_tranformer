import unittest

from compare_fp32_dispatch import summarize


def sample(time, size, allocations):
    return dict(name='BenchmarkFixture-4', ns_per_op=time, bytes_per_op=size,
                allocs_per_op=allocations)


class DispatchSummaryTests(unittest.TestCase):
    def test_summary_keeps_all_samples_and_separates_allocation_from_time(self):
        reports = [
            dict(variant='before', samples=[sample(100, 96, 1), sample(300, 96, 1)]),
            dict(variant='after', samples=[sample(250, 0, 0), sample(50, 0, 0)]),
        ]
        result = summarize(reports)['BenchmarkFixture-4']
        self.assertEqual(result['before']['ns_per_op'], 200)
        self.assertEqual(result['after']['ns_per_op'], 150)
        self.assertEqual(result['time_ratio'], 0.75)
        self.assertEqual(result['before']['time_range'], [100, 300])
        self.assertEqual(result['after']['time_range'], [50, 250])
        self.assertEqual(result['after']['allocs_per_op'], 0)
        self.assertEqual(summarize(list(reversed(reports))), summarize(reports))

    def test_missing_variant_cannot_produce_a_comparison(self):
        with self.assertRaises(KeyError):
            summarize([dict(variant='after', samples=[sample(100, 0, 0)])])


if __name__ == '__main__':
    unittest.main()
