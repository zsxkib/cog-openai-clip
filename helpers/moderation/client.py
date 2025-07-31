import asyncio
import re
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI

from helpers.images.processing import optimized_base64

# All valid moderation categories
MODERATION_CATEGORIES = {
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
}


class ContentModerationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class OpenAIModerationClient:
    def __init__(self):
        api_key = (
            Path(".openai-api-key").read_text().strip()
            if Path(".openai-api-key").exists()
            else ""
        )
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "omni-moderation-latest"

    async def check_content(
        self,
        texts: List[str] | None = None,
        image_url: str | None = None,
        image_path: Path | None = None,
        timeout: float = 10.0,
    ) -> Dict:
        input_data = []
        if texts:
            input_data.extend({"type": "text", "text": text} for text in texts)
        if image_url:
            input_data.append({"type": "image_url", "image_url": {"url": image_url}})
        if image_path:
            base64_img = await optimized_base64(image_path, quality=60, max_dim=384)
            input_data.append({"type": "image_url", "image_url": {"url": base64_img}})

        response = await asyncio.wait_for(
            self.client.moderations.create(model=self.model, input=input_data),
            timeout=timeout,
        )

        flagged = [
            cat
            for cat in response.results[0].categories.__dict__
            if getattr(response.results[0].categories, cat)
        ]
        if flagged:
            print(f"Flagged categories: {', '.join(flagged)}")

        return response.results[0].model_dump()

    async def check_multiple(
        self,
        content_list: List[Dict],
        timeout: float = 10.0,
    ) -> List[Dict]:
        tasks = []
        for content in content_list:
            task = self.check_content(
                texts=content.get("texts"),
                image_url=content.get("image_url"),
                image_path=content.get("image_path"),
                timeout=timeout,
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def raise_if_flagged(
        self,
        content_list: List[Dict],
        types: List[str] = [],
        timeout: float = 10.0,
        substitute_child_terms: bool = False,
    ) -> None:
        invalid_types = set(types) - MODERATION_CATEGORIES
        if invalid_types:
            print(f"Invalid moderation types: {invalid_types}")
            print(f"Valid types are: {sorted(MODERATION_CATEGORIES)}")

        # Apply text substitution if enabled
        if substitute_child_terms:
            for content in content_list:
                if "texts" in content and content["texts"]:
                    content["texts"] = [
                        re.sub(r"\b(girl)\b", r"\1 (child)", text)
                        for text in content["texts"]
                    ]
                    content["texts"] = [
                        re.sub(r"\b(boy)\b", r"\1 (child)", text)
                        for text in content["texts"]
                    ]

        try:
            results = await self.check_multiple(
                content_list=content_list,
                timeout=timeout,
            )

            for result in results:
                if isinstance(result, Exception):
                    print(f"Warning: Moderation check failed: {result}")
                    continue

                if result["categories"].get("sexual/minors", False):
                    raise ContentModerationError(
                        "Content flagged and reported for containing illegal material"
                    )

                # Check if content is both sexual and has any other flag
                is_sexual = result["categories"].get("sexual", False)
                has_other_flag = any(
                    result["categories"].get(cat, False)
                    for cat in MODERATION_CATEGORIES
                    if cat not in ["sexual", "sexual/minors"]
                )
                if is_sexual and has_other_flag:
                    raise ContentModerationError(
                        "Content flagged for containing sexual content with other violations"
                    )

                flagged_types = []
                for check_type in types:
                    if check_type in MODERATION_CATEGORIES and result["categories"].get(
                        check_type, False
                    ):
                        flagged_types.append(check_type)

                if flagged_types:
                    raise ContentModerationError(
                        f"Content flagged for: {', '.join(flagged_types)}"
                    )

        except (asyncio.TimeoutError, Exception) as e:
            if isinstance(e, ContentModerationError):
                raise e from None
            if isinstance(e, asyncio.TimeoutError):
                print("Warning: Moderation timed out")
            else:
                print("Warning: Moderation check failed")
