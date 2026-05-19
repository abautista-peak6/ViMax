import unittest

from utils.video_prompt_sanitizer import sanitize_video_prompt


class TestVideoPromptSanitizer(unittest.TestCase):
    def test_replaces_full_names_tags_and_partial_speaker_names(self):
        prompt = (
            "Close-up of <ANTHONY BAUTISTA> at a workstation. "
            "[Speaker] Anthony: ViMax stitches clips together. "
            "Anthony Bautista's expression is confident."
        )

        sanitized = sanitize_video_prompt(prompt, ["ANTHONY BAUTISTA"])

        self.assertNotIn("ANTHONY BAUTISTA", sanitized)
        self.assertNotIn("Anthony Bautista", sanitized)
        self.assertNotIn("Anthony:", sanitized)
        self.assertIn("the presenter", sanitized)
        self.assertIn("the presenter's expression", sanitized)

    def test_does_not_replace_unrelated_brand_names(self):
        prompt = "ViMax UI glows while Anthony explains the product."

        sanitized = sanitize_video_prompt(prompt, ["ANTHONY BAUTISTA"])

        self.assertIn("ViMax", sanitized)
        self.assertNotIn("Anthony", sanitized)

    def test_single_word_role_identifier_is_left_alone(self):
        prompt = "The CREATOR points at a timeline."

        sanitized = sanitize_video_prompt(prompt, ["CREATOR"])

        self.assertEqual(prompt, sanitized)


if __name__ == "__main__":
    unittest.main()
