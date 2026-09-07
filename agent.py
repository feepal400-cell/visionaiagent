"""Groq tool-calling agent for image analysis with automatic model fallback."""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from groq import BadRequestError, Groq, NotFoundError, RateLimitError
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from tools.vision_tool import detect_objects

load_dotenv()

_DETECTION_CACHE: dict[str, dict[str, Any]] = {}

# Ordered priority list — first available model wins.
# Includes verified free, fast, tool-calling supported models on Groq.
MODEL_PRIORITY = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

DEFAULT_PROMPT = (
    "Describe the detected objects and their positions. Use the detection tool "
    "before answering, and do not invent objects."
)

# If the best YOLO detection is below this confidence, ask the vision LLM to verify.
_VISION_VERIFY_THRESHOLD = 0.40

VISION_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "detect_objects",
        "description": "Detect objects in an image and return spatial metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "conf_threshold": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "default": 0.35,
                },
            },
            "required": ["image_path"],
        },
    },
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentError(RuntimeError):
    """Base class for agent failures."""


class ConfigurationError(AgentError):
    """Raised when required agent configuration is missing."""


class MalformedResponseError(AgentError):
    """Raised when Groq returns an unusable response."""


# ---------------------------------------------------------------------------
def get_groq_api_key() -> str | None:
    """Retrieve GROQ_API_KEY from environment or Streamlit Cloud secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            val = str(st.secrets["GROQ_API_KEY"]).strip()
            if val:
                os.environ["GROQ_API_KEY"] = val
                return val
    except Exception:
        pass
    return None


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_message(response: Any) -> Any:
    choices = _get_value(response, "choices")
    if not choices:
        raise MalformedResponseError("Groq response did not contain choices")
    message = _get_value(choices[0], "message")
    if message is None:
        raise MalformedResponseError("Groq response choice did not contain a message")
    return message


def _tool_call_dict(tool_call: Any) -> dict[str, Any]:
    function = _get_value(tool_call, "function")
    name = _get_value(function, "name")
    arguments = _get_value(function, "arguments")
    call_id = _get_value(tool_call, "id")
    if not name or not isinstance(arguments, str):
        raise MalformedResponseError("Groq returned an invalid tool call")
    return {"id": call_id or "detect_objects", "name": name, "arguments": arguments}


def _run_detection_tool(
    tool_call: dict[str, Any],
    fallback_path: str,
    conf_threshold: float,
) -> dict[str, Any]:
    if tool_call["name"] != "detect_objects":
        raise MalformedResponseError(
            f"Groq requested unsupported tool: {tool_call['name']}"
        )
    try:
        arguments = json.loads(tool_call["arguments"])
    except json.JSONDecodeError as error:
        raise MalformedResponseError("Groq tool arguments were not valid JSON") from error
    if not isinstance(arguments, dict):
        raise MalformedResponseError("Groq tool arguments must be a JSON object")

    requested_path = arguments.get("image_path") or fallback_path
    if not isinstance(requested_path, str) or not requested_path.strip():
        raise MalformedResponseError("Groq tool call did not provide image_path")
    try:
        metadata = json.loads(detect_objects(requested_path, conf_threshold))
    except json.JSONDecodeError as error:
        raise MalformedResponseError("Detection tool returned invalid JSON") from error
    if not isinstance(metadata, dict):
        raise MalformedResponseError("Detection tool returned a non-object response")
    _DETECTION_CACHE[str(Path(requested_path).expanduser().resolve())] = metadata
    return metadata


def _metadata_summary(metadata: dict[str, Any]) -> str:
    """Provide a deterministic response when Groq returns no text."""
    counts = metadata.get("counts", {})
    detections = metadata.get("detections", [])
    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(detections, list):
        detections = []

    if counts:
        objects = ", ".join(f"{count} {name}" for name, count in counts.items())
    else:
        objects = "no objects"

    locations = [
        str(item.get("spatial_location"))
        for item in detections
        if isinstance(item, dict) and item.get("spatial_location")
    ]
    location_text = (
        f" Detected object locations: {', '.join(locations)}."
        if locations
        else ""
    )
    return f"Verified detection found {objects}.{location_text}"


_MODEL_ERROR_TYPES: tuple[type[Exception], ...] = (
    NotFoundError,
    BadRequestError,
    RateLimitError,
)

# Free-tier Groq caps output tokens at 1000/min. Stay safely under.
_MAX_TOKENS = 350


def _create_completion(
    client: Groq,
    messages: Sequence[ChatCompletionMessageParam],
    tool_choice: Literal["auto", "none", "required"] = "auto",
) -> tuple[Any, str]:
    """Try each model in priority order; return (response, model_used).

    Falls back to the next model when a model-not-found, deprecation,
    or rate-limit error is raised by Groq.
    """
    last_error: Exception | None = None
    for model in MODEL_PRIORITY:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[VISION_TOOL],
                tool_choice=tool_choice,
                max_tokens=_MAX_TOKENS,
            )
            return response, model
        except _MODEL_ERROR_TYPES as error:
            # Model unavailable or rate-limited — try next.
            last_error = error
            continue
        except Exception as error:
            # Fall back to next model on other API errors as well
            last_error = error
            continue

    raise AgentError(
        f"All Groq models failed. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Run object detection and ask Groq to describe the resulting metadata.

    Returns a dict with keys:
        text               — LLM scene description
        annotated_image_path — path to the YOLO-annotated image
        raw_metadata       — full spatial JSON from the vision tool
        model_used         — which Groq model actually answered
        elapsed_seconds    — wall-clock time for the full pipeline
    """
    start = time.time()

    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image_path must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(conf_threshold, (int, float)) or not 0.1 <= conf_threshold <= 0.9:
        raise ValueError("conf_threshold must be between 0.1 and 0.9")

    api_key = get_groq_api_key()
    if not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not configured. For Streamlit Cloud, add GROQ_API_KEY in App Settings > Secrets. For local, add it to your .env file."
        )

    client = Groq(api_key=api_key)
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a precise visual assistant. Call detect_objects for the "
                "provided image before answering. The detector metadata is authoritative: "
                "use its exact counts and never infer, add, or duplicate objects. "
                "Return a concise plain-text scene description after the tool result."
            ),
        },
        {"role": "user", "content": f"{prompt}\nImage filepath: {image_path}"},
    ]

    # --- First request (tool-calling) ---
    response, model_used = _create_completion(client, messages, tool_choice="auto")
    message = _response_message(response)
    tool_calls = _get_value(message, "tool_calls", []) or []
    metadata: dict[str, Any] | None = None

    if tool_calls:
        calls = [_tool_call_dict(tool_call) for tool_call in tool_calls]
        metadata = _run_detection_tool(calls[0], image_path, float(conf_threshold))
    else:
        calls = []
        metadata = _run_detection_tool(
            {"name": "detect_objects", "arguments": json.dumps({"image_path": image_path})},
            image_path,
            float(conf_threshold),
        )

    counts = metadata.get("counts", {})
    detections = metadata.get("detections", [])
    max_conf = max((d.get("confidence", 0) for d in detections), default=0)
    needs_vision = not counts or max_conf < _VISION_VERIFY_THRESHOLD
    vision_fallback_success = False

    if needs_vision:
        try:
            base64_image = _encode_image(image_path)
            if counts:
                yolo_summary = ", ".join(
                    f"{d.get('class')} (conf {d.get('confidence', 0):.2f})"
                    for d in detections
                )
                vision_prompt = (
                    f"A standard object detector found these objects but with LOW confidence: "
                    f"{yolo_summary}. "
                    f"Please look at the image and tell me what the objects ACTUALLY are. "
                    f"Correct any wrong labels. {prompt}"
                )
            else:
                vision_prompt = f"The standard object detector found no objects. {prompt}"

            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]
            vision_response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=vision_messages,
                max_tokens=_MAX_TOKENS,
            )
            final_message = _response_message(vision_response)
            fallback_label = "Vision Verify" if counts else "Vision Fallback"
            model_used = f"llama-3.2-11b-vision-preview ({fallback_label})"
            vision_fallback_success = True
        except Exception:
            pass

    if not vision_fallback_success:
        if calls:
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                    "role": "assistant",
                    "content": _get_value(message, "content") or "",
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in calls
                    ],
                    },
                )
            )
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "name": calls[0]["name"],
                    "content": json.dumps(metadata),
                    },
                )
            )

            # --- Follow-up request (synthesis) ---
            final_response, model_used = _create_completion(
                client, messages, tool_choice="none"
            )
            final_message = _response_message(final_response)
        else:
            final_message = message

    text = _get_value(final_message, "content")
    if not isinstance(text, str) or not text.strip():
        text = _metadata_summary(metadata)
    annotated_path = metadata.get("annotated_image_path")
    if not isinstance(annotated_path, str) or not annotated_path:
        raise MalformedResponseError("Detection metadata lacked annotated_image_path")

    elapsed = round(time.time() - start, 2)

    return {
        "text": text,
        "annotated_image_path": annotated_path,
        "raw_metadata": metadata,
        "model_used": model_used,
        "elapsed_seconds": elapsed,
    }


def ask_agent_question(
    user_question: str,
    image_path: str,
    chat_history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Answer a follow-up question using cached detector metadata."""
    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("user_question must be a non-empty string")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image_path must be a non-empty string")

    cache_key = str(Path(image_path).expanduser().resolve())
    metadata = _DETECTION_CACHE.get(cache_key)
    if metadata is None:
        metadata = json.loads(detect_objects(image_path))
        if not isinstance(metadata, dict):
            raise AgentError("Detection tool returned invalid cached metadata")
        _DETECTION_CACHE[cache_key] = metadata

    history = chat_history if isinstance(chat_history, list) else []
    conversation: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You answer follow-up questions about one analyzed image. "
                "Use only the authoritative YOLO metadata below; never invent "
                "objects, counts, or positions. If the metadata does not answer "
                "the question, say that clearly.\n\n"
                f"YOLO spatial metadata:\n{json.dumps(metadata, indent=2)}"
            ),
        }
    ]
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            conversation.append({"role": role, "content": content})
    conversation.append({"role": "user", "content": user_question.strip()})

    api_key = get_groq_api_key()
    if not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not configured. For Streamlit Cloud, add GROQ_API_KEY in App Settings > Secrets. For local, add it to your .env file."
        )
    client = Groq(api_key=api_key)
    response, _ = _create_completion(client, conversation, tool_choice="none")
    answer = _get_value(_response_message(response), "content")
    if not isinstance(answer, str) or not answer.strip():
        raise MalformedResponseError("Groq returned an empty follow-up answer")

    updated_history: list[dict[str, str]] = [
        {
            "role": str(message["role"]),
            "content": str(message["content"]),
        }
        for message in history
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
    updated_history.extend(
        [
            {"role": "user", "content": user_question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    )
    return updated_history


run_agent = analyze_image
