"""Integration tests for MiniMax provider support.

These tests verify end-to-end flow through the pipeline config loading
and chat model factory invocation.  They mock the factory so
no real API calls are made.

Heavy multimedia dependencies (moviepy, scenedetect, cv2, google-genai,
etc.) are stubbed at the module level so the pipeline modules can be
imported in a lightweight test environment.
"""

import importlib
import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

# ---- stub heavy deps before any project imports ----
_STUB_MODULES = [
    "moviepy", "cv2", "scenedetect", "scenedetect.detectors",
    "PIL", "PIL.Image",
    "requests",
    "tenacity",
    "aiohttp",
    "faiss",
    "google", "google.genai", "google.genai.types", "google.genai.errors",
    "langchain_community", "langchain_community.vectorstores",
    "langchain_community.vectorstores.FAISS",
]
_saved = {}
for _mod in _STUB_MODULES:
    _saved[_mod] = sys.modules.get(_mod)
    mock = MagicMock()
    # Give stub a __spec__ so importlib.util.find_spec() works
    mock.__spec__ = importlib.machinery.ModuleSpec(_mod, None)
    mock.__path__ = []
    sys.modules[_mod] = mock

_agents_stub = types.ModuleType("agents")
for _name in [
    "Screenwriter",
    "CharacterExtractor",
    "CharacterPortraitsGenerator",
    "StoryboardArtist",
    "CameraImageGenerator",
    "ReferenceImageSelector",
]:
    setattr(_agents_stub, _name, MagicMock())
sys.modules["agents"] = _agents_stub

_chat_factory_stub = types.ModuleType("utils.chat_model_factory")
_chat_factory_stub.create_chat_model = MagicMock()
_chat_factory_stub.chat_model_args_from_config = lambda section: {
    **dict(section.get("init_args", section)),
    **{
        key: section[key]
        for key in ("max_requests_per_minute", "max_requests_per_day")
        if section.get(key) is not None
    },
}
sys.modules["utils.chat_model_factory"] = _chat_factory_stub

from utils.provider_presets import resolve_chat_model_config


class TestPipelineConfigResolution(unittest.TestCase):
    """Integration: config dict -> resolved chat model kwargs."""

    def _make_minimax_config(self, **overrides):
        base = {
            "model": "MiniMax-M2.7",
            "model_provider": "minimax",
            "api_key": "test-key",
        }
        base.update(overrides)
        return base

    def test_full_minimax_config_resolution(self):
        config = self._make_minimax_config()
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(resolved["model"], "MiniMax-M2.7")
        self.assertEqual(resolved["api_key"], "test-key")

    def test_minimax_highspeed_model(self):
        config = self._make_minimax_config(model="MiniMax-M2.7-highspeed")
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model"], "MiniMax-M2.7-highspeed")
        self.assertEqual(resolved["model_provider"], "openai")

    def test_minimax_m25_model(self):
        config = self._make_minimax_config(model="MiniMax-M2.5")
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model"], "MiniMax-M2.5")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-api-key"})
    def test_env_key_fallback_in_config(self):
        config = {
            "model": "MiniMax-M2.7",
            "model_provider": "minimax",
        }
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["api_key"], "env-api-key")

    def test_openrouter_config_unchanged(self):
        """Existing OpenRouter configs must not be affected."""
        config = {
            "model": "google/gemini-2.5-flash-lite-preview-09-2025",
            "model_provider": "openai",
            "api_key": "or-key",
            "base_url": "https://openrouter.ai/api/v1",
        }
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(resolved["model"], "google/gemini-2.5-flash-lite-preview-09-2025")

    def test_minimax_resolves_to_openai_provider(self):
        """Verify that resolved kwargs have model_provider='openai'."""
        config = self._make_minimax_config()
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(resolved["model"], "MiniMax-M2.7")

    def test_temperature_clamping_in_pipeline_flow(self):
        config = self._make_minimax_config(temperature=2.0)
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["temperature"], 1.0)

    def test_extra_kwargs_preserved(self):
        config = self._make_minimax_config(max_tokens=4096, top_p=0.9)
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["max_tokens"], 4096)
        self.assertEqual(resolved["top_p"], 0.9)


class TestPipelineInitFromConfig(unittest.TestCase):
    """Integration: full pipeline init_from_config with MiniMax config."""

    @patch("pipelines.idea2video_pipeline.create_chat_model")
    @patch("pipelines.idea2video_pipeline.RenderBackend.from_config")
    def test_idea2video_pipeline_minimax_config(self, mock_backend, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_backend.return_value = MagicMock(image_generator=MagicMock(), video_generator=MagicMock())

        from pipelines.idea2video_pipeline import Idea2VideoPipeline
        pipeline = Idea2VideoPipeline.init_from_config("configs/idea2video_minimax.yaml")

        mock_create.assert_called_once()
        call_args = mock_create.call_args[0][0]
        self.assertEqual(call_args["model_provider"], "minimax")
        self.assertEqual(call_args["model"], "MiniMax-M2.7")

    @patch("pipelines.script2video_pipeline.create_chat_model")
    @patch("pipelines.script2video_pipeline.RenderBackend.from_config")
    def test_script2video_pipeline_minimax_config(self, mock_backend, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_backend.return_value = MagicMock(image_generator=MagicMock(), video_generator=MagicMock())

        from pipelines.script2video_pipeline import Script2VideoPipeline
        pipeline = Script2VideoPipeline.init_from_config("configs/script2video_minimax.yaml")

        mock_create.assert_called_once()
        call_args = mock_create.call_args[0][0]
        self.assertEqual(call_args["model_provider"], "minimax")
        self.assertEqual(call_args["model"], "MiniMax-M2.7")

    @patch("pipelines.idea2video_pipeline.create_chat_model")
    @patch("pipelines.idea2video_pipeline.RenderBackend.from_config")
    def test_default_vertex_config_flows_to_factory(self, mock_backend, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_backend.return_value = MagicMock(image_generator=MagicMock(), video_generator=MagicMock())

        from pipelines.idea2video_pipeline import Idea2VideoPipeline
        pipeline = Idea2VideoPipeline.init_from_config("configs/idea2video.yaml")

        mock_create.assert_called_once()
        call_args = mock_create.call_args[0][0]
        self.assertEqual(call_args["model_provider"], "google_vertex")
        self.assertEqual(call_args["model"], "gemini-3.1-pro-preview")


class TestCameoRegistry(unittest.TestCase):
    def test_full_cameo_name_uses_first_name_for_story(self):
        from pipelines.idea2video_pipeline import Idea2VideoPipeline

        pipeline = Idea2VideoPipeline.__new__(Idea2VideoPipeline)

        self.assertEqual(pipeline._cameo_story_name("Anthony Bautista"), "Anthony")

    def test_single_visible_character_is_renamed_to_cameo_story_name(self):
        from interfaces import CharacterInScene
        from pipelines.idea2video_pipeline import Idea2VideoPipeline

        pipeline = Idea2VideoPipeline.__new__(Idea2VideoPipeline)
        with tempfile.TemporaryDirectory() as working_dir:
            pipeline.working_dir = working_dir
            characters = [
                CharacterInScene(
                    idx=0,
                    identifier_in_scene="Alex",
                    is_visible=True,
                    static_features="Alex is a presenter.",
                    dynamic_features="Alex stands in a studio.",
                )
            ]

            story, characters = pipeline._normalize_cameo_character_name(
                story="Alex introduces ViMax.",
                characters=characters,
                cameo_story_name="Anthony",
            )
            registry = pipeline._build_cameo_portrait_registry(
                characters=characters,
                cameo_image_path="cameo/abautista.png",
                cameo_name="Anthony",
                cameo_description=None,
            )

        self.assertEqual(story, "Anthony introduces ViMax.")
        self.assertEqual(characters[0].identifier_in_scene, "Anthony")
        self.assertIn("Anthony", registry)
        self.assertEqual(registry["Anthony"]["front"]["path"], "cameo/abautista.png")

    def test_ambiguous_cameo_still_requires_a_clear_name(self):
        from interfaces import CharacterInScene
        from pipelines.idea2video_pipeline import Idea2VideoPipeline

        pipeline = Idea2VideoPipeline.__new__(Idea2VideoPipeline)
        characters = [
            CharacterInScene(
                idx=0,
                identifier_in_scene="Alex",
                is_visible=True,
                static_features="A presenter.",
                dynamic_features="Standing in a studio.",
            ),
            CharacterInScene(
                idx=1,
                identifier_in_scene="Sam",
                is_visible=True,
                static_features="A product user.",
                dynamic_features="Using a laptop.",
            ),
        ]

        with self.assertRaises(ValueError):
            pipeline._build_cameo_portrait_registry(
                characters=characters,
                cameo_image_path="cameo/abautista.png",
                cameo_name="Anthony",
                cameo_description=None,
            )

    def test_seeded_cameo_front_is_copied_not_regenerated(self):
        from interfaces import CharacterInScene
        from pipelines.idea2video_pipeline import Idea2VideoPipeline

        class _ImageOutput:
            def __init__(self, data):
                self.data = data

            def save(self, path):
                with open(path, "wb") as f:
                    f.write(self.data)

        class _PortraitGenerator:
            front_calls = 0
            side_calls = 0
            back_calls = 0

            async def generate_front_portrait(self, character, style):
                self.front_calls += 1
                return _ImageOutput(b"generated-front")

            async def generate_side_portrait(self, character, front_image_path):
                self.side_calls += 1
                return _ImageOutput(b"generated-side")

            async def generate_back_portrait(self, character, front_image_path):
                self.back_calls += 1
                return _ImageOutput(b"generated-back")

        with tempfile.TemporaryDirectory() as working_dir:
            seed_path = os.path.join(working_dir, "abautista.png")
            with open(seed_path, "wb") as f:
                f.write(b"cameo-reference")

            pipeline = Idea2VideoPipeline.__new__(Idea2VideoPipeline)
            pipeline.working_dir = working_dir
            generator = _PortraitGenerator()
            pipeline.character_portraits_generator = generator
            character = CharacterInScene(
                idx=0,
                identifier_in_scene="Anthony",
                is_visible=True,
                static_features="A presenter.",
                dynamic_features="Standing in a studio.",
            )

            registry = asyncio.run(
                pipeline.generate_character_portraits(
                    characters=[character],
                    character_portraits_registry={
                        "Anthony": {
                            "front": {
                                "path": seed_path,
                                "description": "Uploaded cameo headshot.",
                            }
                        }
                    },
                    style="realistic",
                )
            )

            front_path = registry["Anthony"]["front"]["path"]
            with open(front_path, "rb") as f:
                front_bytes = f.read()

        self.assertEqual(front_bytes, b"cameo-reference")
        self.assertEqual(generator.front_calls, 0)
        self.assertEqual(generator.side_calls, 1)
        self.assertEqual(generator.back_calls, 1)


class TestScript2VideoRetryPrompts(unittest.TestCase):
    def _shot_description(self):
        from interfaces import ShotDescription

        return ShotDescription(
            idx=0,
            is_last=True,
            cam_idx=0,
            visual_desc="A brave kitten in a tiny helmet crosses a stormy beach.",
            variation_type="medium",
            variation_reason="The kitten advances from the surf to the ridge.",
            ff_desc="The kitten stands in wet sand under storm clouds.",
            ff_vis_char_idxs=[0],
            lf_desc="The kitten reaches a small flag at the ridge.",
            lf_vis_char_idxs=[0],
            motion_desc="The kitten runs forward through rain and ocean spray.",
            audio_desc="Dramatic orchestral music, rain, waves, and distant rumbles.",
        )

    def test_text_only_retry_prompt_uses_current_shot(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline

        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        pipeline._video_prompt_character_identifiers = []

        prompt = pipeline._build_text_only_fallback_prompt(self._shot_description())

        self.assertIn("brave kitten", prompt)
        self.assertNotIn("phone", prompt.lower())
        self.assertNotIn("infomercial", prompt.lower())
        self.assertNotIn("glass of water", prompt.lower())

    def test_safe_retry_prompt_does_not_inject_old_product_demo(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline

        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        pipeline._video_prompt_character_identifiers = []

        prompt = pipeline._build_safe_video_retry_prompt(self._shot_description())

        self.assertIn("stormy beach", prompt)
        self.assertNotIn("phone", prompt.lower())
        self.assertNotIn("product-demo", prompt.lower())

    def test_input_image_policy_error_retries_without_reference_frames(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline

        class FakeVideoOutput:
            def save(self, path):
                with open(path, "wb") as f:
                    f.write(b"video")

        class FakeVideoGenerator:
            def __init__(self):
                self.calls = []

            async def generate_single_video(self, prompt, reference_image_paths):
                self.calls.append((prompt, reference_image_paths))
                if len(self.calls) == 1:
                    raise RuntimeError(
                        "Video generation failed: {'message': "
                        "\"Veo could not generate videos because the input image "
                        "violates Vertex AI's usage guidelines.\"}"
                    )
                return FakeVideoOutput()

        async def run_test():
            with tempfile.TemporaryDirectory() as working_dir:
                pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
                pipeline.working_dir = working_dir
                pipeline.video_generator = FakeVideoGenerator()
                pipeline._video_prompt_character_identifiers = []
                first_frame_event = asyncio.Event()
                first_frame_event.set()
                last_frame_event = asyncio.Event()
                last_frame_event.set()
                pipeline.frame_events = {
                    0: {
                        "first_frame": first_frame_event,
                        "last_frame": last_frame_event,
                    }
                }

                await pipeline.generate_video_for_single_shot(self._shot_description())

                calls = pipeline.video_generator.calls
                self.assertEqual(len(calls), 2)
                self.assertTrue(calls[0][1])
                self.assertEqual(calls[1][1], [])
                self.assertTrue(os.path.exists(os.path.join(working_dir, "shots", "0", "video.mp4")))

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
