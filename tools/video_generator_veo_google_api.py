import logging
from typing import List, Optional
import asyncio
from google.genai import types
from google.genai.errors import ClientError
from interfaces.video_output import VideoOutput
from utils.google_vertex import create_genai_client
from utils.rate_limiter import RateLimiter

# https://ai.google.dev/gemini-api/docs/video-generation?hl=zh-cn


class VideoGeneratorVeoGoogleAPI:
    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        t2v_model: str = "veo-3.1-generate-001",
        ff2v_model: str = "veo-3.1-generate-001",
        flf2v_model: str = "veo-3.1-generate-001",
        generate_audio: bool = True,
        person_generation: str = "allow_adult",
        output_gcs_uri: Optional[str] = None,
        use_vertex_ai: bool = True,
        api_version: str = "v1",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.api_key = api_key
        self.t2v_model = t2v_model
        self.ff2v_model = ff2v_model
        self.flf2v_model = flf2v_model
        self.generate_audio = generate_audio
        self.person_generation = person_generation
        self.output_gcs_uri = output_gcs_uri
        self.rate_limiter = rate_limiter

        self.client = create_genai_client(
            api_key=api_key,
            project=project,
            location=location,
            use_vertex_ai=use_vertex_ai,
            api_version=api_version,
        )
    
    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        resolution: str = "1080p",
        aspect_ratio: str = "16:9",
        duration: int = 8,
    ) -> VideoOutput:

        params = {
            "prompt": prompt,
        }
        config_params = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
            "generate_audio": self.generate_audio,
            "person_generation": self.person_generation,
        }
        if self.output_gcs_uri:
            config_params["output_gcs_uri"] = self.output_gcs_uri

        if len(reference_image_paths) == 0:
            params["model"] = self.t2v_model
        elif len(reference_image_paths) == 1:
            params["model"] = self.ff2v_model
            params["image"] = types.Image.from_file(location=reference_image_paths[0])
        elif len(reference_image_paths) == 2:
            params["model"] = self.flf2v_model
            params["image"] = types.Image.from_file(location=reference_image_paths[0])
            config_params["last_frame"] = types.Image.from_file(location=reference_image_paths[1])
        else:
            raise ValueError("The number of reference images must be no more than 2")

        logging.info(f"Calling {params['model']} to generate video...")

        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Retry logic for rate limit errors
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                operation = self.client.models.generate_videos(
                    **params,
                    config=types.GenerateVideosConfig(**config_params),
                )
                break
            except ClientError as e:
                if e.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Rate limit hit (429), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        while not operation.done:
            await asyncio.sleep(2)
            operation = self.client.operations.get(operation)
            logging.info(f"Video generation not completed, waiting 2 seconds...")

        # Check if operation completed successfully
        if operation.error:
            error_msg = f"Video generation failed: {operation.error}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        if not operation.response:
            error_msg = "Video generation completed but no response received"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
            error_msg = "Video generation completed but no videos were generated"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        generated_video = operation.response.generated_videos[0]
        video = generated_video.video
        video_bytes = getattr(video, "video_bytes", None)

        if not video_bytes and hasattr(self.client, "files"):
            self.client.files.download(file=video)
            video_bytes = getattr(video, "video_bytes", None)

        video_uri = getattr(video, "uri", None) or getattr(video, "gcs_uri", None)

        if video_bytes:
            return VideoOutput(
                fmt="bytes",
                ext="mp4",
                data=video_bytes,
            )

        if video_uri:
            return VideoOutput(fmt="url", ext="mp4", data=video_uri)

        raise RuntimeError("Video generation completed but no video bytes or URI were returned")
