# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest import mock

from google.antigravity.utils import cli_utils


class CliUtilsTest(unittest.TestCase):
  """Validates CLI utilities in cli_utils.py."""

  @mock.patch("sys.stdout.write")
  @mock.patch("sys.stdout.flush")
  def test_spinner(self, mock_flush, mock_write):
    """Verifies spinner animation starts and stops."""
    spinner = cli_utils.Spinner("Loading")
    spinner.start()
    self.assertTrue(spinner._running)
    self.assertTrue(spinner._thread.is_alive())
    spinner.stop()
    self.assertFalse(spinner._running)
    # Check that it cleared the line
    mock_write.assert_any_call("\r\033[K")
    mock_flush.assert_called()

  def test_set_markdown_enabled(self):
    """Verifies set_markdown_enabled toggles the state."""
    cli_utils.set_markdown_enabled(True)
    self.assertTrue(cli_utils._MARKDOWN_ENABLED)
    cli_utils.set_markdown_enabled(False)
    self.assertFalse(cli_utils._MARKDOWN_ENABLED)

  def test_render_markdown_disabled(self):
    """Verifies _render_markdown returns raw text when disabled."""
    cli_utils.set_markdown_enabled(False)
    text = "# Heading\n**bold**"
    self.assertEqual(cli_utils._render_markdown(text), text)

  def test_render_markdown_enabled(self):
    """Verifies _render_markdown renders text when enabled."""
    cli_utils.set_markdown_enabled(True)
    text = "**bold**"
    rendered = cli_utils._render_markdown(text)
    self.assertIn("bold", rendered)
    self.assertNotEqual(rendered, text)
    cli_utils.set_markdown_enabled(False)

  def test_render_markdown_headings(self):
    """Verifies _render_markdown handles headings."""
    cli_utils.set_markdown_enabled(True)
    text = "# Heading 1\n## Heading 2"
    rendered = cli_utils._render_markdown(text)
    self.assertIn("Heading 1", rendered)
    self.assertIn("Heading 2", rendered)
    cli_utils.set_markdown_enabled(False)

  @mock.patch("sys.stdout.write")
  @mock.patch("time.sleep")
  def test_spinner_color_change(self, mock_sleep, mock_write):
    """Verifies spinner changes color after 4 frames."""
    spinner = cli_utils.Spinner("Loading")

    call_count = 0

    def mock_sleep_side_effect(_):
      nonlocal call_count
      call_count += 1
      if call_count >= 5:
        spinner._running = False

    mock_sleep.side_effect = mock_sleep_side_effect

    spinner._running = True
    spinner._animate()

    # Verify it called write with _MAGENTA ("\033[35m")
    found_magenta = False
    for call in mock_write.call_args_list:
      args, _ = call
      if "\033[35m" in args[0]:
        found_magenta = True
        break
    self.assertTrue(found_magenta)

  def test_display_width(self):
    """Verifies _display_width accounts for wide characters."""
    self.assertEqual(cli_utils._display_width("hello"), 5)
    # "你好" are wide characters
    self.assertEqual(cli_utils._display_width("你好"), 4)

  def test_center_display(self):
    """Verifies _center_display centers text correctly."""
    self.assertEqual(cli_utils._center_display("abc", 5), " abc ")
    # Padding is 3, left_pad=1, right_pad=2
    self.assertEqual(cli_utils._center_display("abc", 6), " abc  ")
    # If string is wider than width, return string as is
    self.assertEqual(cli_utils._center_display("abcdef", 5), "abcdef")

  @mock.patch("builtins.print")
  def test_print_cli_header(self, mock_print):
    """Verifies print_cli_header prints title and subtitle."""
    cli_utils.print_cli_header("Test Title")
    self.assertTrue(mock_print.called)
    found_title = False
    for call in mock_print.call_args_list:
      args, _ = call
      if "Test Title" in args[0]:
        found_title = True
        break
    self.assertTrue(found_title)

  @mock.patch("builtins.print")
  def test_print_cli_header_with_extra_lines(self, mock_print):
    """Verifies print_cli_header prints extra lines."""
    cli_utils.print_cli_header("Test Title", {"Label": "Value"})
    found_extra = False
    for call in mock_print.call_args_list:
      args, _ = call
      if "Label: Value" in args[0]:
        found_extra = True
        break
    self.assertTrue(found_extra)


if __name__ == "__main__":
  unittest.main()
