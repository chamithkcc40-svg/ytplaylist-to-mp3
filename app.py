#!/usr/bin/env python3
"""
YouTube Playlist to MP3 Telegram Bot
Designed for Hugging Face Docker Spaces with embedded aiohttp health-check/webhook server.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
from urllib.parse import urlparse

import yt_dlp
from aiohttp import web
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import (
    BadRequest,
    Forbidden,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
# Reduce noisy third-party HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

logger = logging.getLogger("yt_mp3_bot")

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DOWNLOAD_ROOT = Path("/tmp/downloads")
COOKIES_FILE_PATH = Path("/tmp/cookies.txt")
MAX_TELEGRAM_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB Telegram Bot API limit
TRACK_COOLDOWN_SECONDS = 1.5
CANCEL_CALLBACK_DATA = "cancel_job"

YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
FILENAME_UNSAFE_PATTERN = re.compile(r'[\\/:*?"<>|\r\n]+')


def _parse_int_env(names: tuple[str, ...], default: int, minimum: int = 1) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            try:
                value = int(raw.strip())
                if value >= minimum:
                    return value
                logger.warning("Env %s=%r is below %d; using default %d.", name, raw, minimum, default)
            except ValueError:
                logger.warning("Invalid integer for env %s=%r; using default %d.", name, raw, default)
    return default


BOT_TOKEN: str = (os.getenv("BOT_TOKEN") or "").strip()
MAX_PLAYLIST_SIZE: int = _parse_int_env(("MAX_PLAYLIST_SIZE", "MAX_PLAYLIST_LIMIT"), default=30, minimum=1)
COOKIES_CONTENT: str = os.getenv("COOKIES_CONTENT") or ""
WEBHOOK_URL: str = (os.getenv("WEBHOOK_URL") or "").strip()
WEBHOOK_SECRET: str = (os.getenv("WEBHOOK_SECRET") or "").strip()
PORT: int = _parse_int_env(("PORT",), default=7860, minimum=1)

# ---------------------------------------------------------------------------
# In-Memory Concurrency & Cancellation State (per chat_id)
# ---------------------------------------------------------------------------
active_jobs: dict[int, asyncio.Task[Any]] = {}
cancel_flags: dict[int, asyncio.Event] = {}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackItem:
    url: str
    title: str
    uploader: Optional[str] = None
    duration: Optional[int] = None


@dataclass(frozen=True)
class ExtractionResult:
    tracks: list[TrackItem]
    total_found: int
    was_truncated: bool
    playlist_title: Optional[str] = None


@dataclass(frozen=True)
class DownloadedTrack:
    file_path: Path
    title: str
    performer: Optional[str]
    duration: Optional[int]


# ---------------------------------------------------------------------------
# Startup Utilities
# ---------------------------------------------------------------------------
def setup_runtime_environment() -> bool:
    """Ensure /tmp/downloads exists and materialize /tmp/cookies.txt if COOKIES_CONTENT is set."""
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    if COOKIES_CONTENT.strip():
        try:
            content = COOKIES_CONTENT.strip() + "\n"
            COOKIES_FILE_PATH.write_text(content, encoding="utf-8")
            os.chmod(COOKIES_FILE_PATH, 0o600)
            logger.info("Wrote YouTube cookies to %s (%d bytes).", COOKIES_FILE_PATH, len(content))
            return True
        except OSError as exc:
            logger.error("Failed to write cookies file %s: %s", COOKIES_FILE_PATH, exc)
            return False

    # Remove stale cookies file if env var was unset
    if COOKIES_FILE_PATH.exists():
        try:
            COOKIES_FILE_PATH.unlink()
        except OSError:
            pass
    return False


USE_COOKIES: bool = False


def extract_youtube_url(text: str) -> Optional[str]:
    """Find and validate the first YouTube URL in a message."""
    if not text:
        return None

    for match in URL_PATTERN.finditer(text):
        candidate = match.group(0).rstrip(".,);!]>")
        try:
            parsed = urlparse(candidate)
        except ValueError:
            continue

        host = (parsed.hostname or "").lower()
        if parsed.scheme in {"http", "https"} and host in YOUTUBE_DOMAINS:
            return candidate

    return None


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """Create a clean, Telegram-friendly MP3 filename."""
    cleaned = FILENAME_UNSAFE_PATTERN.sub(" ", name or "audio").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        cleaned = "audio"
    return f"{cleaned[:max_length].rstrip()}.mp3"


def cancel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Cancel", callback_data=CANCEL_CALLBACK_DATA)]]
    )


def cleanup_directory(path: Path) -> None:
    """Safely remove a directory and all files inside it."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Telegram API Retry & Flood-Control Helper
# ---------------------------------------------------------------------------
async def telegram_call_with_retry(
    operation: Callable[[], Coroutine[Any, Any, Any]],
    *,
    max_retries: int = 5,
    ignore_not_modified: bool = False,
) -> Any:
    """
    Execute an async Telegram API call with automatic RetryAfter handling
    and transient network backoff.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await operation()
        except RetryAfter as exc:
            wait_seconds = float(exc.retry_after) + 1.0
            logger.warning(
                "Telegram rate limit hit (RetryAfter=%s). Sleeping %.1fs (attempt %d/%d).",
                exc.retry_after,
                wait_seconds,
                attempt,
                max_retries,
            )
            if attempt >= max_retries:
                raise
            await asyncio.sleep(wait_seconds)
        except BadRequest as exc:
            if ignore_not_modified and "message is not modified" in str(exc).lower():
                return None
            raise
        except (TimedOut, NetworkError) as exc:
            if attempt >= max_retries:
                raise
            backoff = min(2.0 * attempt, 10.0)
            logger.warning(
                "Transient Telegram network error: %s. Retrying in %.1fs (attempt %d/%d).",
                exc,
                backoff,
                attempt,
                max_retries,
            )
            await asyncio.sleep(backoff)


async def safe_edit_status(
    bot: Any,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    """Edit the live status message safely, ignoring MessageNotModified errors."""
    try:
        await telegram_call_with_retry(
            lambda: bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
            ),
            ignore_not_modified=True,
        )
    except (BadRequest, Forbidden) as exc:
        logger.debug("Could not edit status message %s in chat %s: %s", message_id, chat_id, exc)
    except TelegramError as exc:
        logger.warning("Failed to update status message in chat %s: %s", chat_id, exc)


# ---------------------------------------------------------------------------
# yt-dlp Blocking Operations (Executed inside asyncio.to_thread)
# ---------------------------------------------------------------------------
def _base_ytdlp_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
    }
    if USE_COOKIES and COOKIES_FILE_PATH.exists():
        opts["cookiefile"] = str(COOKIES_FILE_PATH)
    return opts


def _normalize_duration(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)) and value > 0:
        return int(round(value))
    return None


def _extract_metadata_sync(url: str, max_items: int) -> ExtractionResult:
    """Fast flat metadata extraction for single YouTube videos or playlists."""
    ydl_opts = {
        **_base_ytdlp_opts(),
        "extract_flat": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return ExtractionResult(tracks=[], total_found=0, was_truncated=False)

    raw_entries = info.get("entries")
    tracks: list[TrackItem] = []

    if raw_entries is not None:
        # Playlist or multi-entry source
        materialized = list(raw_entries)
        valid_entries: list[TrackItem] = []
        for entry in materialized:
            if not isinstance(entry, dict):
                continue

            title = (entry.get("title") or "").strip()
            if title in {"[Private video]", "[Deleted video]"}:
                continue

            entry_url = entry.get("url") or entry.get("webpage_url")
            entry_id = entry.get("id")
            if entry_url and str(entry_url).startswith(("http://", "https://")):
                track_url = str(entry_url)
            elif entry_id:
                track_url = f"https://www.youtube.com/watch?v={entry_id}"
            else:
                continue

            valid_entries.append(
                TrackItem(
                    url=track_url,
                    title=title or f"Track {len(valid_entries) + 1}",
                    uploader=entry.get("uploader") or entry.get("channel"),
                    duration=_normalize_duration(entry.get("duration")),
                )
            )

        total_found = len(valid_entries)
        was_truncated = total_found > max_items
        tracks = valid_entries[:max_items]
        return ExtractionResult(
            tracks=tracks,
            total_found=total_found,
            was_truncated=was_truncated,
            playlist_title=info.get("title"),
        )

    # Single video
    track_url = info.get("webpage_url") or url
    title = (info.get("title") or "YouTube Audio").strip()
    uploader = info.get("uploader") or info.get("channel")
    duration = _normalize_duration(info.get("duration"))
    tracks.append(TrackItem(url=track_url, title=title, uploader=uploader, duration=duration))
    return ExtractionResult(tracks=tracks, total_found=1, was_truncated=False)


def _download_and_convert_track_sync(track: TrackItem, track_dir: Path) -> Optional[DownloadedTrack]:
    """Download a single YouTube track and convert it to 192kbps MP3 via FFmpeg."""
    track_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(track_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        **_base_ytdlp_opts(),
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": outtmpl,
        "retries": 3,
        "fragment_retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(track.url, download=True)

    mp3_files = sorted(track_dir.glob("*.mp3"))
    if not mp3_files:
        return None

    mp3_path = mp3_files[0]
    info_dict = info if isinstance(info, dict) else {}

    title = (
        info_dict.get("track")
        or info_dict.get("title")
        or track.title
        or mp3_path.stem
    ).strip()
    performer = (
        info_dict.get("artist")
        or info_dict.get("uploader")
        or info_dict.get("channel")
        or track.uploader
    )
    if isinstance(performer, str):
        performer = performer.strip() or None

    duration = _normalize_duration(info_dict.get("duration")) or track.duration

    return DownloadedTrack(
        file_path=mp3_path,
        title=title,
        performer=performer,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Sequential Download / Upload / Cleanup Pipeline
# ---------------------------------------------------------------------------
async def run_download_job(
    chat_id: int,
    url: str,
    bot: Any,
    cancel_event: asyncio.Event,
) -> None:
    """
    Full sequential job lifecycle for one chat:
    1. Flat-extract metadata
    2. Download -> Size Check -> Upload -> Immediate Delete (one by one)
    3. Final summary & guaranteed resource cleanup
    """
    user_dir = DOWNLOAD_ROOT / str(chat_id)
    status_message_id: Optional[int] = None

    try:
        user_dir.mkdir(parents=True, exist_ok=True)

        # Send initial extracting status
        init_msg = await telegram_call_with_retry(
            lambda: bot.send_message(
                chat_id=chat_id,
                text="🔍 Fetching video/playlist metadata...",
                reply_markup=cancel_keyboard(),
            )
        )
        status_message_id = init_msg.message_id

        # Extract track list without blocking event loop
        try:
            extraction = await asyncio.to_thread(_extract_metadata_sync, url, MAX_PLAYLIST_SIZE)
        except Exception as exc:
            logger.exception("Metadata extraction failed for chat %s (%s): %s", chat_id, url, exc)
            await safe_edit_status(
                bot,
                chat_id,
                status_message_id,
                "❌ Failed to extract metadata from that YouTube link. Please verify the URL is public and try again.",
                reply_markup=None,
            )
            return

        if cancel_event.is_set():
            cleanup_directory(user_dir)
            await safe_edit_status(bot, chat_id, status_message_id, "Job cancelled.", reply_markup=None)
            return

        if not extraction.tracks:
            await safe_edit_status(
                bot,
                chat_id,
                status_message_id,
                "⚠️ No downloadable videos were found at that link (the video/playlist may be private or unavailable).",
                reply_markup=None,
            )
            return

        tracks = extraction.tracks
        total = len(tracks)
        sent_count = 0
        skipped_count = 0

        truncation_notice = ""
        if extraction.was_truncated:
            truncation_notice = (
                f"\n⚠️ Playlist has {extraction.total_found} tracks — "
                f"truncated to the first {MAX_PLAYLIST_SIZE} tracks."
            )

        await safe_edit_status(
            bot,
            chat_id,
            status_message_id,
            f"🎵 Found {total} track{'s' if total != 1 else ''}. Starting download...{truncation_notice}",
            reply_markup=cancel_keyboard(),
        )

        # Sequential one-by-one pipeline
        for idx, track in enumerate(tracks, start=1):
            # 1. Check Cancel Flag before starting each track
            if cancel_event.is_set():
                cleanup_directory(user_dir)
                await safe_edit_status(
                    bot,
                    chat_id,
                    status_message_id,
                    f"Job cancelled.\nSent: {sent_count} / {total} | Skipped: {skipped_count}",
                    reply_markup=None,
                )
                logger.info("Job cancelled for chat %s at track %d/%d.", chat_id, idx, total)
                return

            short_title = (track.title[:60] + "…") if len(track.title) > 60 else track.title
            await safe_edit_status(
                bot,
                chat_id,
                status_message_id,
                (
                    f"⏳ Processing {idx} / {total}: {short_title}\n"
                    f"Progress: {sent_count} / {total} sent | Skipped: {skipped_count}"
                    f"{truncation_notice}"
                ),
                reply_markup=cancel_keyboard(),
            )

            track_dir = user_dir / f"track_{idx:03d}"
            try:
                # 2. Download & convert in worker thread
                try:
                    downloaded = await asyncio.to_thread(
                        _download_and_convert_track_sync,
                        track,
                        track_dir,
                    )
                except Exception as exc:
                    logger.warning(
                        "Download error for chat %s track %d (%s): %s",
                        chat_id,
                        idx,
                        track.url,
                        exc,
                    )
                    downloaded = None

                # Check cancellation again before uploading
                if cancel_event.is_set():
                    cleanup_directory(user_dir)
                    await safe_edit_status(
                        bot,
                        chat_id,
                        status_message_id,
                        f"Job cancelled.\nSent: {sent_count} / {total} | Skipped: {skipped_count}",
                        reply_markup=None,
                    )
                    logger.info("Job cancelled mid-track for chat %s at track %d/%d.", chat_id, idx, total)
                    return

                if not downloaded or not downloaded.file_path.exists():
                    skipped_count += 1
                    logger.warning("Skipping track %d/%d for chat %s (download failed): %s", idx, total, chat_id, track.title)
                    await telegram_call_with_retry(
                        lambda t=track.title: bot.send_message(
                            chat_id=chat_id,
                            text=f"⚠️ Skipped \"{t}\" — failed to download or convert audio.",
                        )
                    )
                else:
                    # 3. 50MB Size Check
                    file_size = downloaded.file_path.stat().st_size
                    if file_size > MAX_TELEGRAM_AUDIO_BYTES:
                        skipped_count += 1
                        size_mb = file_size / (1024 * 1024)
                        logger.info(
                            "Skipping oversized track for chat %s (%s: %.2f MB).",
                            chat_id,
                            downloaded.title,
                            size_mb,
                        )
                        await telegram_call_with_retry(
                            lambda t=downloaded.title, mb=size_mb: bot.send_message(
                                chat_id=chat_id,
                                text=(
                                    f"⚠️ Skipped \"{t}\" — file size ({mb:.1f} MB) "
                                    f"exceeds Telegram's 50 MB limit."
                                ),
                            )
                        )
                    else:
                        # 4 & 5. Upload with metadata + RetryAfter flood protection
                        audio_filename = sanitize_filename(downloaded.title)

                        async def _upload_audio() -> Any:
                            with downloaded.file_path.open("rb") as audio_fp:
                                return await bot.send_audio(
                                    chat_id=chat_id,
                                    audio=audio_fp,
                                    filename=audio_filename,
                                    title=downloaded.title[:64],
                                    performer=downloaded.performer[:64] if downloaded.performer else None,
                                    duration=downloaded.duration,
                                )

                        try:
                            await telegram_call_with_retry(_upload_audio)
                            sent_count += 1
                        except TelegramError as exc:
                            skipped_count += 1
                            logger.error(
                                "Failed to upload track %d/%d for chat %s (%s): %s",
                                idx,
                                total,
                                chat_id,
                                downloaded.title,
                                exc,
                            )
                            await telegram_call_with_retry(
                                lambda t=downloaded.title: bot.send_message(
                                    chat_id=chat_id,
                                    text=f"⚠️ Skipped \"{t}\" — Telegram upload failed.",
                                )
                            )
            finally:
                # 6. Immediate Cleanup of MP3 and any partial/temp files for this track
                cleanup_directory(track_dir)

            # 7. Live Progress Update after each track (keep Cancel button attached)
            await safe_edit_status(
                bot,
                chat_id,
                status_message_id,
                (
                    f"Progress: {sent_count} / {total} sent | Skipped: {skipped_count}"
                    f"{truncation_notice}"
                ),
                reply_markup=cancel_keyboard(),
            )

            # 8. Cooldown Delay between tracks to avoid Telegram flood limits
            if idx < total and not cancel_event.is_set():
                await asyncio.sleep(TRACK_COOLDOWN_SECONDS)

        # Final completion summary (remove Cancel button)
        summary = (
            f"✅ Done!\n"
            f"Total sent: {sent_count} / {total}\n"
            f"Skipped / Failed: {skipped_count}"
            f"{truncation_notice}"
        )
        await safe_edit_status(bot, chat_id, status_message_id, summary, reply_markup=None)
        logger.info(
            "Completed job for chat %s: sent=%d, skipped=%d, total=%d.",
            chat_id,
            sent_count,
            skipped_count,
            total,
        )

    except asyncio.CancelledError:
        cleanup_directory(user_dir)
        if status_message_id is not None:
            await safe_edit_status(bot, chat_id, status_message_id, "Job cancelled.", reply_markup=None)
        raise
    except Exception as exc:
        logger.exception("Unhandled error in download job for chat %s: %s", chat_id, exc)
        if status_message_id is not None:
            await safe_edit_status(
                bot,
                chat_id,
                status_message_id,
                "❌ An unexpected error occurred while processing your request.",
                reply_markup=None,
            )
    finally:
        # Always clean up the user's temp folder and release active job lock
        cleanup_directory(user_dir)
        active_jobs.pop(chat_id, None)
        cancel_flags.pop(chat_id, None)


# ---------------------------------------------------------------------------
# Telegram Bot Handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    await update.effective_message.reply_text(
        "👋 Send me a YouTube video or playlist link (`youtube.com`, `youtu.be`, or `music.youtube.com`) "
        f"and I will convert and send each track as a 192 kbps MP3 (up to {MAX_PLAYLIST_SIZE} tracks per playlist).\n\n"
        "Use /cancel at any time to stop an active download.",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    await update.effective_message.reply_text(
        "📌 *How to use this bot:*\n"
        "1. Paste any public YouTube video or playlist URL.\n"
        "2. Tracks are downloaded, converted to MP3, uploaded, and deleted one by one.\n"
        "3. Tracks larger than 50 MB are skipped automatically.\n"
        "4. Press *❌ Cancel* or send /cancel to abort a running job.",
        parse_mode="Markdown",
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if not chat or not message:
        return

    chat_id = chat.id
    task = active_jobs.get(chat_id)
    flag = cancel_flags.get(chat_id)

    if task and not task.done() and flag:
        flag.set()
        await message.reply_text("🛑 Cancellation requested. Stopping your active download...")
        logger.info("Cancel command received for chat %s.", chat_id)
    else:
        await message.reply_text("ℹ️ You don't have an active download job right now.")


async def handle_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return

    chat = update.effective_chat
    if not chat:
        await query.answer()
        return

    chat_id = chat.id
    task = active_jobs.get(chat_id)
    flag = cancel_flags.get(chat_id)

    if task and not task.done() and flag:
        flag.set()
        await query.answer("Cancelling job...")
        logger.info("Cancel button clicked for chat %s.", chat_id)
    else:
        await query.answer("No active job to cancel.", show_alert=False)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if not chat or not message or not message.text:
        return

    url = extract_youtube_url(message.text)
    if not url:
        await message.reply_text(
            "⚠️ Please send a valid YouTube video or playlist link "
            "(youtube.com, youtu.be, or music.youtube.com)."
        )
        return

    chat_id = chat.id
    existing_task = active_jobs.get(chat_id)
    if existing_task and not existing_task.done():
        await message.reply_text(
            "⏳ You already have an active download in progress. "
            "Please wait for it to finish or send /cancel to stop it."
        )
        return

    cancel_event = asyncio.Event()
    cancel_flags[chat_id] = cancel_event

    task = asyncio.create_task(
        run_download_job(chat_id=chat_id, url=url, bot=context.bot, cancel_event=cancel_event),
        name=f"download_job_{chat_id}",
    )
    active_jobs[chat_id] = task
    logger.info("Started download job for chat %s: %s", chat_id, url)


# ---------------------------------------------------------------------------
# Embedded Async HTTP Server (Health Check + Optional Webhook)
# ---------------------------------------------------------------------------
def build_web_app(ptb_app: Application, webhook_enabled: bool) -> web.Application:
    web_app = web.Application()

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "mode": "webhook" if webhook_enabled else "polling",
                "active_jobs": len(active_jobs),
            },
            status=200,
        )

    async def handle_webhook(request: web.Request) -> web.Response:
        if not webhook_enabled:
            return web.json_response({"error": "Webhook mode is not enabled"}, status=404)

        if WEBHOOK_SECRET:
            incoming_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not hmac.compare_digest(incoming_secret, WEBHOOK_SECRET):
                logger.warning("Rejected webhook request with invalid secret token.")
                return web.json_response({"error": "Unauthorized"}, status=403)

        try:
            payload = await request.json()
            update = Update.de_json(payload, ptb_app.bot)
            await ptb_app.update_queue.put(update)
            return web.json_response({"ok": True}, status=200)
        except Exception as exc:
            logger.exception("Error processing incoming webhook update: %s", exc)
            return web.json_response({"error": "Bad request"}, status=400)

    web_app.router.add_get("/", handle_health)
    web_app.router.add_get("/health", handle_health)
    web_app.router.add_post("/webhook", handle_webhook)
    return web_app


def resolve_webhook_endpoint(raw_url: str) -> str:
    """
    Preserve custom webhook URLs (e.g. Cloudflare Worker URLs) while ensuring
    bare root URLs automatically route cleanly if desired.
    """
    parsed = urlparse(raw_url)
    if parsed.path in {"", "/"}:
        return raw_url.rstrip("/") + "/webhook"
    return raw_url


# ---------------------------------------------------------------------------
# Application Entrypoint
# ---------------------------------------------------------------------------
async def run() -> None:
    global USE_COOKIES

    if not BOT_TOKEN:
        logger.critical("Missing required environment variable: BOT_TOKEN. Exiting.")
        sys.exit(1)

    USE_COOKIES = setup_runtime_environment()
    webhook_enabled = bool(WEBHOOK_URL)

    logger.info(
        "Starting YouTube MP3 Telegram Bot (mode=%s, port=%d, max_playlist_size=%d, cookies=%s)",
        "webhook" if webhook_enabled else "polling",
        PORT,
        MAX_PLAYLIST_SIZE,
        USE_COOKIES,
    )

    ptb_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .connect_timeout(30.0)
        .read_timeout(120.0)
        .write_timeout(120.0)
        .pool_timeout(30.0)
        .build()
    )

    ptb_app.add_handler(CommandHandler("start", cmd_start))
    ptb_app.add_handler(CommandHandler("help", cmd_help))
    ptb_app.add_handler(CommandHandler("cancel", cmd_cancel))
    ptb_app.add_handler(CallbackQueryHandler(handle_cancel_callback, pattern=f"^{CANCEL_CALLBACK_DATA}$"))
    ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start embedded aiohttp server on 0.0.0.0:PORT
    web_app = build_web_app(ptb_app, webhook_enabled=webhook_enabled)
    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    logger.info("HTTP server listening on http://0.0.0.0:%d", PORT)

    # Start python-telegram-bot application in the same asyncio event loop
    await ptb_app.initialize()
    await ptb_app.start()

    if webhook_enabled:
        webhook_target = resolve_webhook_endpoint(WEBHOOK_URL)
        webhook_kwargs: dict[str, Any] = {
            "url": webhook_target,
            "allowed_updates": Update.ALL_TYPES,
        }
        if WEBHOOK_SECRET:
            webhook_kwargs["secret_token"] = WEBHOOK_SECRET
        await ptb_app.bot.set_webhook(**webhook_kwargs)
        logger.info("Telegram webhook registered at %s", webhook_target)
    else:
        await ptb_app.bot.delete_webhook(drop_pending_updates=False)
        if ptb_app.updater is None:
            raise RuntimeError("PTB updater is unavailable in polling mode.")
        await ptb_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Telegram bot polling started.")

    # Wait until SIGINT or SIGTERM
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down gracefully...")

        # Signal any active download jobs to stop and wait briefly for cleanup
        for flag in list(cancel_flags.values()):
            flag.set()
        pending_tasks = [t for t in active_jobs.values() if not t.done()]
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if not webhook_enabled and ptb_app.updater and ptb_app.updater.running:
            await ptb_app.updater.stop()

        await ptb_app.stop()
        await ptb_app.shutdown()
        await runner.cleanup()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass