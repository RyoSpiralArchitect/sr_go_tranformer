import unittest

from compare_generation_pair import namespace_source


class GenerationNamespaceTests(unittest.TestCase):
    def test_only_the_package_declaration_changes(self):
        source = b'package main\n\nfunc example() string { return "main" }\n'
        self.assertEqual(namespace_source(source, 'before'), source.replace(b'package main', b'package before'))
        self.assertEqual(namespace_source(source, 'after'), source.replace(b'package main', b'package after'))

    def test_missing_duplicate_and_unexpected_packages_fail_closed(self):
        for source, package in ((b'package other\n', 'before'),
                                (b'package main\npackage main\n', 'after'),
                                (b'package main\n', '../outside')):
            with self.subTest(source=source, package=package), self.assertRaises(ValueError):
                namespace_source(source, package)


if __name__ == '__main__':
    unittest.main()
