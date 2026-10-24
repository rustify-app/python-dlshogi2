"""Wiki 原稿の公開ルールを確認します。"""
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(REPO_ROOT, 'wiki')
LINK_RE = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
NOT_A_PAGE = {'README.md'}


def wiki_pages():
    """公開対象の Markdown ページを返します。"""
    return sorted(name for name in os.listdir(WIKI_DIR)
                  if name.endswith('.md') and name not in NOT_A_PAGE)


def internal_links(text):
    """他の Wiki ページへのリンク先を返します。"""
    for target in LINK_RE.findall(text):
        target = target.strip()
        if target.startswith(('http://', 'https://', 'mailto:', '#', '/')):
            continue
        yield target


class WikiSourceTest(unittest.TestCase):
    def setUp(self):
        self.pages = wiki_pages()

    def test_required_pages_exist(self):
        for required in ('Home.md', '_Sidebar.md'):
            self.assertIn(required, self.pages)

    def test_pages_do_not_link_to_other_wiki_pages(self):
        links = []
        for name in self.pages:
            with open(os.path.join(WIKI_DIR, name), encoding='utf-8') as page:
                links.extend('{} -> {}'.format(name, target)
                             for target in internal_links(page.read()))
        self.assertEqual(links, [], 'Wiki ページ間のリンクがあります: {}'.format(links))

    def test_publish_script_is_executable(self):
        script = os.path.join(WIKI_DIR, 'publish.sh')
        self.assertTrue(os.access(script, os.X_OK))


if __name__ == '__main__':
    unittest.main()
