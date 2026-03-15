import os
import asyncio
import tempfile
import requests
import replicate
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from pydub import AudioSegment
import subprocess
import json

BOT_TOKEN = os.environ["BOT_TOKEN"]
REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ فیلمەکە وەردەگرم...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download video
        video = update.message.video or update.message.document
        file = await context.bot.get_file(video.file_id)
        video_path = os.path.join(tmpdir, "input.mp4")
        await file.download_to_drive(video_path)

        await msg.edit_text("🎵 دەنگەکە دەردەهێنم...")

        # Extract audio as WAV for diarization
        audio_path = os.path.join(tmpdir, "audio.wav")
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            audio_path, "-y"
        ], check=True, capture_output=True)

        await msg.edit_text("🤖 Replicate AI دەنگەکان دەناسێنێت... (چەند خولەک دەخایەنێت)")

        # Upload audio to replicate
        with open(audio_path, "rb") as f:
            output = replicate.run(
                "meronym/speaker-diarization:latest",
                input={
                    "audio": f,
                    "num_speakers": 0,  # auto-detect
                }
            )

        # Parse diarization output
        # output is list of {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5}
        if not output:
            await msg.edit_text("❌ هیچ دەنگێک نەدۆزرایەوە")
            return

        await msg.edit_text("✂️ ڤیدیۆکان دەبڕێت...")

        # Group segments by speaker
        speakers = {}
        for segment in output:
            speaker = segment["speaker"]
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(segment)

        await msg.edit_text(f"✅ {len(speakers)} کارەکتەر دۆزرایەوە!\n⏳ ڤیدیۆکان دروست دەکرێن...")

        # For each speaker, create a video with their segments
        speaker_list = sorted(speakers.keys())
        for i, speaker in enumerate(speaker_list):
            segs = speakers[speaker]
            speaker_label = f"کارەکتەر {i+1}"

            # Build ffmpeg filter for cutting and concatenating segments
            filter_parts = []
            concat_parts = []
            for j, seg in enumerate(segs):
                start = seg["start"]
                end = seg["end"]
                duration = end - start
                filter_parts.append(
                    f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{j}];"
                    f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{j}]"
                )
                concat_parts.append(f"[v{j}][a{j}]")

            n = len(segs)
            filter_complex = ";".join(filter_parts)
            filter_complex += f";{''.join(concat_parts)}concat=n={n}:v=1:a=1[outv][outa]"

            out_path = os.path.join(tmpdir, f"speaker_{i+1}.mp4")

            subprocess.run([
                "ffmpeg", "-i", video_path,
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-c:a", "aac",
                out_path, "-y"
            ], check=True, capture_output=True)

            # Send video
            with open(out_path, "rb") as vf:
                await update.message.reply_video(
                    video=vf,
                    caption=f"🎭 {speaker_label} ({n} جار قسەی کردووە)"
                )

        await msg.edit_text(f"✅ تەواو بوو! {len(speakers)} ڤیدیۆ نارد.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سڵاو! 🎬\n\nفیلمەکەت بنێرە، ئۆتۆماتیک هەموو کارەکتەرەکان جیا دەکرێنەوە.\n\n"
        "⚠️ تێبینی: فیلمەکە دەبێت لە 50MB کەمتر بێت (سنووری تێلیگرام)"
    )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))
    print("بۆتەکە دەستی پێکرد...")
    app.run_polling()


if __name__ == "__main__":
    main()
