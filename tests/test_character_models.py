import importlib.util
import os
import unittest


_CHARACTER_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "interfaces",
    "character.py",
)
_SPEC = importlib.util.spec_from_file_location("character_model_for_test", _CHARACTER_MODULE_PATH)
_CHARACTER_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHARACTER_MODULE)
CharacterInScene = _CHARACTER_MODULE.CharacterInScene


class TestCharacterInScene(unittest.TestCase):
    def test_missing_dynamic_features_are_normalized_to_empty_string(self):
        character = CharacterInScene.model_validate(
            {
                "idx": 2,
                "identifier_in_scene": "Infomercial Narrator",
                "is_visible": False,
                "static_features": "",
                "dynamic_features": None,
            }
        )

        self.assertEqual(character.dynamic_features, "")

    def test_missing_static_features_are_normalized_to_empty_string(self):
        character = CharacterInScene.model_validate(
            {
                "idx": 1,
                "identifier_in_scene": "Mascot",
                "is_visible": True,
                "static_features": None,
                "dynamic_features": None,
            }
        )

        self.assertEqual(character.static_features, "")
        self.assertEqual(character.dynamic_features, "")


if __name__ == "__main__":
    unittest.main()
