import argparse
import asyncio
import os
import shutil

from pipelines.idea2video_pipeline import Idea2VideoPipeline


DEFAULT_IDEA = """
A beaufitul fit woman with black hair, great butt and thigs is exercising in a
gym surrounded by glass windows with a beautiful beach view on the outside.
She is performing glute exercises that highlight her beautiful back and sexy outfit
and showing the audience the proper form. Between the 3 different exercises she looks
at the camera with a gorgeous look asking the viewer understood the proper form.
"""
DEFAULT_USER_REQUIREMENT = """
For adults, do not exceed 3 scenes. Each scene should be no more than 5 shots.
"""
DEFAULT_STYLE = "Realistic, warm feel"
DEFAULT_WORKING_DIR = ".working_dir/idea2video"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a video from an idea prompt.")
    parser.add_argument("--idea", default=DEFAULT_IDEA, help="Prompt / idea to turn into a video.")
    parser.add_argument("--idea-file", help="Read the prompt / idea from a text file.")
    parser.add_argument("--requirement", default=DEFAULT_USER_REQUIREMENT, help="Creative constraints.")
    parser.add_argument("--style", default=DEFAULT_STYLE, help="Visual style for generated images/video.")
    parser.add_argument("--config", default="configs/idea2video.yaml", help="Pipeline config path.")
    parser.add_argument("--cameo-image", help="Path to a photo to use as a character identity reference.")
    parser.add_argument(
        "--cameo-name",
        help="Name for the cameo presenter. Full names are shortened to first name in the story/script.",
    )
    parser.add_argument(
        "--cameo-description",
        help="Optional description for the cameo reference image.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the idea2video working directory before running.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    idea = args.idea
    if args.idea_file:
        with open(args.idea_file, "r", encoding="utf-8") as f:
            idea = f.read()

    if args.fresh:
        shutil.rmtree(DEFAULT_WORKING_DIR, ignore_errors=True)

    cameo_image_path = None
    if args.cameo_image:
        os.makedirs(os.path.join(DEFAULT_WORKING_DIR, "cameo"), exist_ok=True)
        cameo_image_path = os.path.join(
            DEFAULT_WORKING_DIR,
            "cameo",
            os.path.basename(args.cameo_image),
        )
        shutil.copy2(args.cameo_image, cameo_image_path)

    pipeline = Idea2VideoPipeline.init_from_config(config_path=args.config)
    final_video_path = await pipeline(
        idea=idea,
        user_requirement=args.requirement,
        style=args.style,
        cameo_image_path=cameo_image_path,
        cameo_name=args.cameo_name,
        cameo_description=args.cameo_description,
    )
    print(f"Final video saved to {final_video_path}")


if __name__ == "__main__":
    asyncio.run(main())
