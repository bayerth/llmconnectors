"""Shared helpers for multimodal (image) requests across provider clients."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ImageSource = str | bytes | Mapping[str, Any]


def build_json_schema_response_format(name: str, schema: dict, strict: bool = True) -> dict:
    """Build OpenAI Chat Completions ``response_format`` for JSON schema structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def resolve_response_format(kwargs: dict) -> dict | None:
    """Pop structured-output fields from ``kwargs`` and return ``response_format`` if set."""
    response_format = kwargs.pop("response_format", None)
    if response_format is not None:
        return response_format
    json_schema = kwargs.pop("json_schema", None)
    if json_schema is None:
        return None
    name = kwargs.pop("json_schema_name", "response")
    strict = kwargs.pop("json_schema_strict", True)
    return build_json_schema_response_format(name, json_schema, strict=strict)


def chat_response_format_to_responses_text(response_format: dict) -> dict:
    """Map Chat Completions ``response_format`` to Responses API ``text`` config."""
    if "format" in response_format and "type" not in response_format:
        return response_format
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        inner = response_format.get("json_schema") or {}
        return {
            "format": {
                "type": "json_schema",
                "name": inner.get("name", "response"),
                "schema": inner.get("schema", {}),
                "strict": inner.get("strict", True),
            }
        }
    if fmt_type == "json_object":
        return {"format": {"type": "json_object"}}
    if fmt_type in {"text", "json_schema", "json_object"}:
        return {"format": response_format}
    raise ValueError(f"Unsupported response_format for Responses API: {response_format!r}")


def resolve_responses_text_config(kwargs: dict) -> dict | None:
    """Pop structured-output kwargs and return Responses API ``text`` parameter value."""
    text = kwargs.pop("text", None)
    if text is not None:
        return text
    text_format = kwargs.pop("text_format", None)
    if text_format is not None:
        return {"format": text_format}
    response_format = resolve_response_format(kwargs)
    if response_format is None:
        return None
    return chat_response_format_to_responses_text(response_format)


def _guess_mime(path: str | None, data: bytes) -> str:
    if path:
        mime, _ = mimetypes.guess_type(path)
        if mime:
            return mime
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _parse_data_url(data_url: str) -> tuple[str, bytes]:
    if not data_url.startswith("data:") or ";base64," not in data_url:
        raise ValueError("Expected a data URL with base64 payload (data:<mime>;base64,...)")
    header, encoded = data_url.split(";base64,", 1)
    mime = header.removeprefix("data:")
    return mime, base64.standard_b64decode(encoded)


def resolve_image_source(source: ImageSource) -> dict[str, Any]:
    """Normalize one image source to ``data_url``, ``remote_url``, ``base64``, ``mime``, ``bytes``."""
    if isinstance(source, Mapping):
        if source.get("data_url"):
            mime, raw = _parse_data_url(source["data_url"])
            return {
                "data_url": source["data_url"],
                "remote_url": None,
                "base64": base64.standard_b64encode(raw).decode("ascii"),
                "mime": source.get("mime") or mime,
                "bytes": raw,
            }
        if source.get("url"):
            url = source["url"]
            if not str(url).startswith(("http://", "https://")):
                raise ValueError(f"Unsupported image url: {url!r}")
            return {
                "data_url": None,
                "remote_url": url,
                "base64": None,
                "mime": source.get("mime"),
                "bytes": source.get("bytes"),
            }
        path_value = source.get("path")
        if path_value:
            return resolve_image_source(str(path_value))
        raw = source.get("data") or source.get("bytes")
        if raw is not None:
            if isinstance(raw, str):
                raw = base64.standard_b64decode(raw)
            mime = source.get("mime") or _guess_mime(None, raw)
            encoded = base64.standard_b64encode(raw).decode("ascii")
            data_url = f"data:{mime};base64,{encoded}"
            return {
                "data_url": data_url,
                "remote_url": None,
                "base64": encoded,
                "mime": mime,
                "bytes": raw,
            }
        raise ValueError(f"Unsupported image mapping: {source!r}")

    if isinstance(source, bytes):
        mime = _guess_mime(None, source)
        encoded = base64.standard_b64encode(source).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        return {
            "data_url": data_url,
            "remote_url": None,
            "base64": encoded,
            "mime": mime,
            "bytes": source,
        }

    if not isinstance(source, str):
        raise TypeError(f"Unsupported image source type: {type(source)!r}")

    if source.startswith("data:image"):
        mime, raw = _parse_data_url(source)
        return {
            "data_url": source,
            "remote_url": None,
            "base64": base64.standard_b64encode(raw).decode("ascii"),
            "mime": mime,
            "bytes": raw,
        }

    if source.startswith(("http://", "https://")):
        return {
            "data_url": None,
            "remote_url": source,
            "base64": None,
            "mime": None,
            "bytes": None,
        }

    path = Path(source)
    if path.is_file():
        raw = path.read_bytes()
        mime = _guess_mime(str(path), raw)
        encoded = base64.standard_b64encode(raw).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        return {
            "data_url": data_url,
            "remote_url": None,
            "base64": encoded,
            "mime": mime,
            "bytes": raw,
        }

    raise ValueError(f"Image path does not exist: {source!r}")


def resolve_image_sources(images: Sequence[ImageSource] | None) -> list[dict[str, Any]]:
    if not images:
        return []
    return [resolve_image_source(item) for item in images]


def build_openai_user_content(user_message: str, images: Iterable[ImageSource] | None) -> str | list:
    """OpenAI Chat Completions user ``content`` (string or multimodal parts)."""
    resolved = resolve_image_sources(list(images) if images is not None else None)
    if not resolved:
        return user_message
    parts: list[dict] = [{"type": "text", "text": user_message}]
    for item in resolved:
        url = item["remote_url"] or item["data_url"]
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def build_lmstudio_input_items(
    user_message: str,
    images: Iterable[ImageSource] | None,
    *,
    prior_text_contents: Sequence[str] | None = None,
) -> list[dict]:
    """LM Studio native ``/api/v1/chat`` input list (``text`` + ``image`` items)."""
    items: list[dict] = []
    for text in prior_text_contents or ():
        items.append({"type": "text", "content": text})
    items.append({"type": "text", "content": user_message})
    for item in resolve_image_sources(list(images) if images is not None else None):
        data_url = item["data_url"]
        if not data_url and item["remote_url"]:
            raise ValueError(
                "LM Studio native chat requires local images (file path, bytes, or data URL); "
                "remote http(s) URLs are not embedded automatically."
            )
        items.append({"type": "image", "data_url": data_url})
    return items


def build_ollama_image_base64_list(images: Iterable[ImageSource] | None) -> list[str]:
    """Ollama chat ``images`` field: raw base64 strings without a data-URL prefix."""
    encoded: list[str] = []
    for item in resolve_image_sources(list(images) if images is not None else None):
        if item["base64"] is None:
            raise ValueError(
                "Ollama requires embeddable image data (file path, bytes, or data URL); "
                "remote http(s) URLs are not downloaded automatically."
            )
        encoded.append(item["base64"])
    return encoded


def build_responses_user_content(user_message: str, images: Iterable[ImageSource] | None) -> list[dict]:
    """OpenAI Responses API user message ``content`` blocks."""
    content: list[dict] = [{"type": "input_text", "text": user_message}]
    for item in resolve_image_sources(list(images) if images is not None else None):
        url = item["remote_url"] or item["data_url"]
        content.append({"type": "input_image", "image_url": url})
    return content
