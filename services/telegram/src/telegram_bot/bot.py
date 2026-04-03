from __future__ import annotations

import structlog
from core.config import Settings, get_settings
from core.exceptions import ConfigurationError
from observability import setup_observability
from orchestrator.graph import resume_turn, run_turn
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from telegram_bot.scheduler import build_scheduler

logger = structlog.get_logger(__name__)

# Maps user_id → thread_id for pending HITL approvals
_PENDING_HITL: dict[str, str] = {}


def _check_auth(user_id: int, settings: Settings) -> bool:
    return user_id == settings.telegram_allowed_user_id


def _require_message(update: Update) -> Message | None:
    """Return update.message if present; callers should return early if None."""
    return update.message


# ── Command handlers ──────────────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        await msg.reply_text("Unauthorized.")
        return
    await msg.reply_text(
        "Sovereign Edge online.\n\n"
        "Ask me anything — Bible questions, job leads, market data, or creative help.\n"
        "Type /help for commands.",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    text = (
        "*Sovereign Edge Commands*\n\n"
        "/start — Wake up the system\n"
        "/brief — Trigger morning brief now\n"
        "/jobs — Run job scan now\n"
        "/market — Market summary now\n"
        "/digest — Weekly research digest\n"
        "/approve — Approve pending HITL action\n"
        "/reject — Reject pending HITL action\n"
        "/help — This message"
    )
    await msg.reply_text(text, parse_mode="Markdown")


async def cmd_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        return
    await msg.reply_text("Generating morning brief...")
    result = await run_turn(
        "Generate morning brief: Bible verse, market summary, top job leads.",
        thread_id=f"brief_{update.effective_user.id}",
        schedule_trigger="manual_brief",
    )
    await msg.reply_text(result, parse_mode="Markdown")


async def cmd_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        return
    await msg.reply_text("Scanning job boards...")
    from career import CareerSquad

    result = await CareerSquad().daily_job_scan()
    text = result or "No new DFW job postings in the last 48 hours."
    await msg.reply_text(text, parse_mode="Markdown")


async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        return
    from intelligence import IntelligenceSquad

    result = await IntelligenceSquad().daily_market_summary()
    await msg.reply_text(result or "No watchlist alerts today.", parse_mode="Markdown")


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        return
    await msg.reply_text("Building research digest...")
    from intelligence import IntelligenceSquad

    result = await IntelligenceSquad().weekly_digest()
    await msg.reply_text(result, parse_mode="Markdown")


# ── HITL approval handlers ────────────────────────────────────────────────────


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        return
    thread_id = _PENDING_HITL.pop(str(update.effective_user.id), None)
    if not thread_id:
        await msg.reply_text("No pending action to approve.")
        return
    result = await resume_turn(thread_id, approved=True)
    await msg.reply_text(f"Approved.\n\n{result}", parse_mode="Markdown")


async def cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        return
    thread_id = _PENDING_HITL.pop(str(update.effective_user.id), None)
    if not thread_id:
        await msg.reply_text("No pending action to reject.")
        return
    await resume_turn(thread_id, approved=False)
    await msg.reply_text("Action cancelled.")


async def hitl_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard approve/reject buttons."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()

    settings = get_settings()
    user_id = query.from_user.id if query.from_user else 0
    if not _check_auth(user_id, settings):
        return

    action, thread_id = query.data.split(":", 1)
    _PENDING_HITL.pop(str(user_id), None)

    approved = action == "approve"
    result = await resume_turn(thread_id, approved=approved)

    label = "Approved" if approved else "Cancelled"
    await query.edit_message_text(f"{label}.\n\n{result}", parse_mode="Markdown")


# ── Main message handler ──────────────────────────────────────────────────────


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = _require_message(update)
    if msg is None:
        return
    settings = get_settings()
    if not update.effective_user or not _check_auth(update.effective_user.id, settings):
        await msg.reply_text("Unauthorized.")
        return

    text = msg.text or ""
    user_id = update.effective_user.id
    thread_id = f"user_{user_id}"

    logger.info("telegram.message", user_id=user_id, length=len(text))
    await context.bot.send_chat_action(chat_id=user_id, action="typing")

    result = await run_turn(text, thread_id=thread_id)

    if result == "" and thread_id not in _PENDING_HITL:
        _PENDING_HITL[str(user_id)] = thread_id
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"approve:{thread_id}"),
                    InlineKeyboardButton("Reject", callback_data=f"reject:{thread_id}"),
                ]
            ]
        )
        await msg.reply_text(
            "This action requires your approval. Review and confirm:",
            reply_markup=keyboard,
        )
    else:
        await msg.reply_text(result or "Done.", parse_mode="Markdown")


# ── Bot entry point ───────────────────────────────────────────────────────────


def run_bot() -> None:
    """Start the Telegram bot (blocking — manages its own event loop)."""
    settings = get_settings()
    setup_observability(settings)

    if not settings.telegram_bot_token:
        raise ConfigurationError("TELEGRAM_BOT_TOKEN is required")
    if not settings.telegram_allowed_user_id:
        raise ConfigurationError("TELEGRAM_ALLOWED_USER_ID is required")

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("brief", cmd_brief))
    app.add_handler(CommandHandler("jobs", cmd_jobs))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("reject", cmd_reject))
    app.add_handler(CallbackQueryHandler(hitl_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    scheduler = build_scheduler(app.bot, settings, settings.telegram_allowed_user_id)
    scheduler.start()

    logger.info("telegram.bot_starting")
    app.run_polling(drop_pending_updates=True)
