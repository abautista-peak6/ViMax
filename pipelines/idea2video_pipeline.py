import os
import logging
import re
import filecmp
import shutil
from agents import Screenwriter, CharacterExtractor, CharacterPortraitsGenerator
from pipelines.script2video_pipeline import Script2VideoPipeline
from interfaces import CharacterInScene
from typing import List, Dict, Optional
import asyncio
import json
from moviepy import VideoFileClip, concatenate_videoclips
import yaml
from tools.render_backend import RenderBackend
from utils.chat_model_factory import create_chat_model


class Idea2VideoPipeline:
    def __init__(
        self,
        chat_model: str,
        image_generator: str,
        video_generator: str,
        working_dir: str,
    ):
        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        self.screenwriter = Screenwriter(chat_model=self.chat_model)
        self.character_extractor = CharacterExtractor(
            chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(
            image_generator=self.image_generator)

    @classmethod
    def init_from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        chat_model = create_chat_model(config["chat_model"]["init_args"])
        backend = RenderBackend.from_config(config)

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config["working_dir"],
        )

    async def extract_characters(
        self,
        story: str,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(
                character) for character in characters]
            print(f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(story)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump()
                          for character in characters], f, ensure_ascii=False, indent=4)
            print(
                f"✅ Extracted {len(characters)} characters from story and saved to {save_path}.")

        return characters

    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
    ):
        character_portraits_registry_path = os.path.join(
            self.working_dir, "character_portraits_registry.json")
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}

        tasks = [
            self.generate_portraits_for_single_character(
                character,
                style,
                character_portraits_registry.get(character.identifier_in_scene),
            )
            for character in characters
            if (
                character.identifier_in_scene not in character_portraits_registry
                or not {"front", "side", "back"}.issubset(
                    character_portraits_registry[character.identifier_in_scene].keys()
                )
            )
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry,
                              f, ensure_ascii=False, indent=4)

            print(
                f"✅ Completed character portrait generation for {len(characters)} characters.")
        else:
            print(
                "🚀 All characters already have portraits, skipping portrait generation.")

        with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
            json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)

        return character_portraits_registry

    async def develop_story(
        self,
        idea: str,
        user_requirement: str,
    ):
        save_path = os.path.join(self.working_dir, "story.txt")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                story = f.read()
            print(f"🚀 Loaded story from existing file.")
        else:
            print("🧠 Developing story...")
            story = await self.screenwriter.develop_story(idea=idea, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(story)
            print(f"✅ Developed story and saved to {save_path}.")

        return story

    async def write_script_based_on_story(
        self,
        story: str,
        user_requirement: str,
    ):
        save_path = os.path.join(self.working_dir, "script.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                script = json.load(f)
            print(f"🚀 Loaded script from existing file.")
        else:
            print("🧠 Writing script based on story...")
            script = await self.screenwriter.write_script_based_on_story(story=story, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=4)
            print(f"✅ Written script based on story and saved to {save_path}.")
        return script

    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
        existing_registry_item: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        character_dir = os.path.join(
            self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)

        front_portrait_path = os.path.join(character_dir, "front.png")
        seeded_front = (existing_registry_item or {}).get("front", {})
        seeded_front_path = seeded_front.get("path")
        front_updated_from_seed = False

        if seeded_front_path and os.path.exists(seeded_front_path):
            if (
                not os.path.exists(front_portrait_path)
                or not filecmp.cmp(seeded_front_path, front_portrait_path, shallow=False)
            ):
                shutil.copy2(seeded_front_path, front_portrait_path)
                front_updated_from_seed = True
                print(
                    f"🎭 Copied cameo reference to {front_portrait_path} for "
                    f"{character.identifier_in_scene}."
                )
        elif os.path.exists(front_portrait_path):
            pass
        else:
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)

        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path) and not front_updated_from_seed:
            pass
        else:
            side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
            side_portrait_output.save(side_portrait_path)

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path) and not front_updated_from_seed:
            pass
        else:
            back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
            back_portrait_output.save(back_portrait_path)

        print(
            f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")

        return {
            character.identifier_in_scene: {
                "front": {
                    "path": front_portrait_path,
                    "description": seeded_front.get(
                        "description",
                        f"A front view portrait of {character.identifier_in_scene}.",
                    ),
                },
                "side": {
                    "path": side_portrait_path,
                    "description": f"A side view portrait of {character.identifier_in_scene}.",
                },
                "back": {
                    "path": back_portrait_path,
                    "description": f"A back view portrait of {character.identifier_in_scene}.",
                },
            }
        }

    async def __call__(
        self,
        idea: str,
        user_requirement: str,
        style: str,
        cameo_image_path: Optional[str] = None,
        cameo_name: Optional[str] = None,
        cameo_description: Optional[str] = None,
    ):

        cameo_story_name = self._cameo_story_name(cameo_name) if cameo_image_path else None
        idea, user_requirement = self._apply_cameo_story_constraints(
            idea=idea,
            user_requirement=user_requirement,
            cameo_story_name=cameo_story_name,
        )

        story = await self.develop_story(idea=idea, user_requirement=user_requirement)

        characters = await self.extract_characters(story=story)
        if cameo_story_name:
            story, characters = self._normalize_cameo_character_name(
                story=story,
                characters=characters,
                cameo_story_name=cameo_story_name,
            )

        seeded_portraits_registry = self._build_cameo_portrait_registry(
            characters=characters,
            cameo_image_path=cameo_image_path,
            cameo_name=cameo_story_name or cameo_name,
            cameo_description=cameo_description,
        )

        character_portraits_registry = await self.generate_character_portraits(
            characters=characters,
            character_portraits_registry=seeded_portraits_registry,
            style=style,
        )

        scene_scripts = await self.write_script_based_on_story(story=story, user_requirement=user_requirement)

        all_video_paths = []

        for idx, scene_script in enumerate(scene_scripts):
            scene_working_dir = os.path.join(self.working_dir, f"scene_{idx}")
            os.makedirs(scene_working_dir, exist_ok=True)
            script2video_pipeline = Script2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                working_dir=scene_working_dir,
            )
            final_video_path = await script2video_pipeline(
                script=scene_script,
                user_requirement=user_requirement,
                style=style,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
            )
            all_video_paths.append(final_video_path)

        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped concatenating videos, already exists.")
        else:
            print(f"🎬 Starting concatenating videos...")
            video_clips = [VideoFileClip(final_video_path)
                           for final_video_path in all_video_paths]
            final_video = concatenate_videoclips(video_clips)
            final_video.write_videofile(
                final_video_path,
                codec="libx264",
                audio_codec="aac",
            )
            print(f"☑️ Concatenated videos, saved to {final_video_path}.")
        return final_video_path

    def _cameo_story_name(self, cameo_name: Optional[str]) -> Optional[str]:
        if not cameo_name:
            return None
        name_parts = re.findall(r"[A-Za-z][A-Za-z'’-]*", cameo_name.strip())
        return name_parts[0] if name_parts else cameo_name.strip()

    def _apply_cameo_story_constraints(
        self,
        idea: str,
        user_requirement: str,
        cameo_story_name: Optional[str],
    ):
        if not cameo_story_name:
            return idea, user_requirement

        cameo_instruction = (
            f"The cameo presenter character must be named {cameo_story_name}. "
            f"Use {cameo_story_name} consistently in the story and script. "
            "Do not rename the presenter to Alex or any other placeholder name. "
            "Do not use the presenter's last name or full real name in dialogue or on-screen text."
        )

        if cameo_story_name.casefold() not in idea.casefold():
            idea = f"{idea.strip()}\n\n{cameo_instruction}"
        user_requirement = f"{user_requirement.strip()}\n{cameo_instruction}"
        return idea, user_requirement

    def _normalize_cameo_character_name(
        self,
        story: str,
        characters: List[CharacterInScene],
        cameo_story_name: str,
    ):
        candidates = [character for character in characters if character.is_visible] or characters
        matched_character = self._match_cameo_character(candidates, cameo_story_name)

        if matched_character is None and len(candidates) == 1:
            matched_character = candidates[0]

        if matched_character is None:
            return story, characters

        original_name = matched_character.identifier_in_scene
        if original_name.strip().casefold() == cameo_story_name.strip().casefold():
            return story, characters

        story = self._replace_character_name(story, original_name, cameo_story_name)
        for character in characters:
            character.static_features = self._replace_character_name(
                character.static_features,
                original_name,
                cameo_story_name,
            )
            character.dynamic_features = self._replace_character_name(
                character.dynamic_features,
                original_name,
                cameo_story_name,
            )
        matched_character.identifier_in_scene = cameo_story_name

        with open(os.path.join(self.working_dir, "story.txt"), "w", encoding="utf-8") as f:
            f.write(story)
        with open(os.path.join(self.working_dir, "characters.json"), "w", encoding="utf-8") as f:
            json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)

        print(f"🎭 Renamed cameo character {original_name!r} to {cameo_story_name!r}.")
        return story, characters

    def _replace_character_name(self, text: Optional[str], old_name: str, new_name: str) -> Optional[str]:
        if not text or not old_name:
            return text
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old_name)}(?![A-Za-z0-9_])")
        return pattern.sub(new_name, text)

    def _build_cameo_portrait_registry(
        self,
        characters: List[CharacterInScene],
        cameo_image_path: Optional[str],
        cameo_name: Optional[str],
        cameo_description: Optional[str],
    ):
        if not cameo_image_path:
            return None

        candidates = [character for character in characters if character.is_visible] or characters
        matched_character = None

        if cameo_name:
            matched_character = self._match_cameo_character(candidates, cameo_name)

        if matched_character is None and len(candidates) == 1:
            matched_character = candidates[0]
            print(
                f"🎭 Using cameo image for extracted character "
                f"{matched_character.identifier_in_scene!r}."
            )

        if matched_character is None:
            available = ", ".join(character.identifier_in_scene for character in characters)
            raise ValueError(
                f"Cameo character '{cameo_name}' was not found unambiguously in extracted characters: {available}. "
                "Use a cameo name that appears in the generated character list, or run with a single visible character."
            )

        description = cameo_description or (
            f"A front-view reference photo of {matched_character.identifier_in_scene}. "
            "Use this image as the identity and face reference for the cameo character."
        )

        return {
            matched_character.identifier_in_scene: {
                "front": {
                    "path": cameo_image_path,
                    "description": description,
                },
            }
        }

    def _match_cameo_character(
        self,
        candidates: List[CharacterInScene],
        cameo_name: str,
    ) -> Optional[CharacterInScene]:
        target_name = cameo_name.strip().casefold()
        exact_matches = [
            character for character in candidates
            if character.identifier_in_scene.strip().casefold() == target_name
        ]
        partial_matches = [
            character for character in candidates
            if target_name in character.identifier_in_scene.strip().casefold().split()
        ]
        matches = exact_matches or partial_matches
        return matches[0] if len(matches) == 1 else None
