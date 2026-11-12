import unittest
from unittest.mock import patch
import psutil
from pipeline115 import common


class ProcessIdentityTests(unittest.TestCase):
    def test_kernel_identity_survives_wall_clock_create_time_shift(self):
        record = common.identity()
        process = psutil.Process(record['pid'])
        with patch.object(process, 'create_time', return_value=record['create_time'] + 1):
            self.assertTrue(common.alive(record))

    def test_wrong_kernel_start_or_boot_id_is_rejected(self):
        record = common.identity()
        record['proc_start_ticks'] += 1
        self.assertFalse(common.alive(record))
        record = common.identity()
        record['boot_id'] = 'different-boot'
        self.assertFalse(common.alive(record))

    def test_legacy_identity_keeps_strict_timestamp_check(self):
        record = common.identity()
        record.pop('proc_start_ticks'); record.pop('boot_id')
        record['create_time'] += 1
        self.assertFalse(common.alive(record))


if __name__ == '__main__':
    unittest.main()
