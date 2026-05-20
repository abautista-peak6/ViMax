import argparse
import asyncio
import os
import shutil
from datetime import datetime

from pipelines.script2video_pipeline import Script2VideoPipeline


DEFAULT_SCRIPT = """
EXT. SCHOOL GYM - DAY
A group of students are practicing basketball in the gym. The gym is large and open, with a basketball hoop at one end and a large crowd of spectators at the other end. John (18, male, tall, athletic) is the star player, and he is practicing his dribble and shot. Jane (17, female, short, athletic) is the assistant coach, and she is helping John with his practice. The other students are watching the practice and cheering for John.
John: (dribbling the ball) I'm going to score a basket!
Jane: (smiling) Good job, John!
John: (shooting the ball) Yes!
John:(The shot misses. He seems frustrated.) Argh! My follow-through feels off today.
Jane:(Walks over, analytical.) Your elbow is drifting out. Remember, straight as an arrow.
John:(Nods, taking the ball again.) Straight as an arrow... Let me try again.
(John takes another shot. This time, the ball swishes through the net perfectly.)
Jane:(Clapping.) There it is! Perfect form! That's the shot we need for the championship.
John:(Retrieving the ball, smiling with renewed confidence.) Thanks, Coach Jane. I just needed you to point it out. One more time?
"""
DEFAULT_USER_REQUIREMENT = """
Fast-paced with no more than 15 shots.
"""
DEFAULT_STYLE = "Anime Style"
DEFAULT_CONFIG = "configs/script2video.yaml"
DEFAULT_WORKING_DIR = ".working_dir/script2video"


def archive_existing_working_dir(working_dir: str):
    if not os.path.exists(working_dir):
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = f"{working_dir}_{timestamp}"
    shutil.move(working_dir, archive_dir)
    print(f"Archived existing working directory to {archive_dir}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a video from a manually written scene script.")
    parser.add_argument("--script", default=DEFAULT_SCRIPT, help="Scene script to turn into a video.")
    parser.add_argument("--script-file", help="Read the scene script from a text file.")
    parser.add_argument("--requirement", default=DEFAULT_USER_REQUIREMENT, help="Creative constraints.")
    parser.add_argument("--style", default=DEFAULT_STYLE, help="Visual style for generated images/video.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Pipeline config path.")
    parser.add_argument("--working-dir", default=DEFAULT_WORKING_DIR, help="Working directory for cached artifacts.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive the script2video working directory before running.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    script = args.script
    if args.script_file:
        with open(args.script_file, "r", encoding="utf-8") as f:
            script = f.read()

    if args.fresh:
        archive_existing_working_dir(args.working_dir)

    pipeline = Script2VideoPipeline.init_from_config(config_path=args.config)
    if args.working_dir:
        pipeline.working_dir = args.working_dir
        os.makedirs(pipeline.working_dir, exist_ok=True)

    final_video_path = await pipeline(script=script, user_requirement=args.requirement, style=args.style)
    print(f"Final video saved to {final_video_path}")


if __name__ == "__main__":
    asyncio.run(main())
