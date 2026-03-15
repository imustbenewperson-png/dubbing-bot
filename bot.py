import os
import asyncio
import tempfile
import subprocess
import re
from collections import defaultdict
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
import numpy as np

BOT_TOKEN = os.environ["BOT_TOKEN"]

user_sessions = {}

def parse_srt(srt_content):
    segments = []
    pattern = re.compile(
        r'\d+\n(\d{2}:\d{2}:\d{2},\d{3})\s-->\s(\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)',
        re.MULTILINE
    )
    for match in pattern.finditer(srt_content):
        start = srt_time_to_seconds(match.group(1))
        end = srt_time_to_seconds(match.group(2))
        text = match.group(3).strip()
        if end > start:
            segments.append({'start': start, 'end': end, 'text': text})
    return segments

def srt_time_to_seconds(time_str):
    time_str = time_str.replace(',', '.')
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def extract_audio_segment(video_path, start, end, out_path):
    duration = end - start
    subprocess.run([
        "ffmpeg", "-ss", str(start), "-i", video_path,
        "-t", str(duration),
        "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le",
        out_path, "-y"
    ], check=True, capture_output=True)

def get_audio_fingerprint(audio_path):
    result = subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-f", "f32le", "-ar", "8000", "-ac", "1", "pipe:1"
    ], capture_output=True)
    if len(result.stdout) < 100:
        return None
    audio_data = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio_data) == 0:
        return None
    chunk_size = max(1, len(audio_data) // 20)
    fingerprint = []
    for i in range(0, min(len(audio_data), chunk_size * 20), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        if len(chunk) > 0:
            fingerprint.extend([float(np.mean(np.abs(chunk))), float(np.std(chunk))])
    return np.array(fingerprint) if fingerprint else None

def compare_fingerprints(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    min_len = min(len(fp1), len(fp2))
    if min_len == 0:
        return 0.0
    fp1, fp2 = fp1[:min_len], fp2[:min_len]
    n1, n2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(fp1/n1, fp2/n2))

def cluster_speakers(fingerprints, threshold=0.82):
    speaker_labels = []
    speaker_reps = {}
    current_speaker = 0
    for fp in fingerprints:
        if fp is None:
            speaker_labels.append(-1)
            continue
        best_spk, best_score = -1, threshold
        for spk_id, rep in speaker_reps.items():
            score = compare_fingerprints(fp, rep)
            if score > best_score:
                best_score, best_spk = score, spk_id
        if best_spk == -1:
            speaker_labels.append(current_speaker)
            speaker_reps[current_speaker] = fp
            current_speaker += 1
        else:
            speaker_labels.append(best_spk)
            speaker_reps[best_spk] = (speaker_reps[best_spk] + fp) / 2
    return speaker_labels


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سڵاو! 🎬\n\n"
        "١. فیلمەکە بنێرە\n"
        "٢. فایلی SRT بنێرە\n\n"
        "بۆتەکە ئۆتۆماتیک هەموو کارەکتەرەکان جیا دەکاتەوە ✅"
    )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    video = update.message.video or update.message.document
    if user_id not in user_sessions:
        user_sessions[user_id] = {}
    user_sessions[user_id]['video_file_id'] = video.file_id
    await update.message.reply_text("✅ فیلمەکە وەرگرتم!\n\nئێستا فایلی SRT بنێرە 📄")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    doc = update.message.document

    if not doc.file_name.lower().endswith('.srt'):
        await update.message.reply_text("⚠️ تەنها فایلی .srt قبووڵ دەکرێت")
        return

    if user_id not in user_sessions or 'video_file_id' not in user_sessions[user_id]:
        await update.message.reply_text("⚠️ پێشتر فیلمەکە بنێرە!")
        return

    msg = await update.message.reply_text("⏳ کارەکتەرەکان دەناسێنم...")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_file = await context.bot.get_file(user_sessions[user_id]['video_file_id'])
        video_path = os.path.join(tmpdir, "input.mp4")
        await video_file.download_to_drive(video_path)

        srt_file = await context.bot.get_file(doc.file_id)
        srt_path = os.path.join(tmpdir, "subtitles.srt")
        await srt_file.download_to_drive(srt_path)

        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            srt_content = f.read()

        segments = parse_srt(srt_content)
        if not segments:
            await msg.edit_text("❌ فایلی SRT هەڵەیە یان بەتاڵە")
            return

        await msg.edit_text(f"🎵 {len(segments)} ڕستە دۆزرایەوە\n⏳ دەنگەکان دەناسێنم...")

        fingerprints = []
        for i, seg in enumerate(segments):
            audio_seg_path = os.path.join(tmpdir, f"seg_{i}.wav")
            try:
                extract_audio_segment(video_path, seg['start'], seg['end'], audio_seg_path)
                fp = get_audio_fingerprint(audio_seg_path)
                fingerprints.append(fp)
            except:
                fingerprints.append(None)

        await msg.edit_text("🧠 کارەکتەرەکان جیا دەکەمەوە...")

        speaker_labels = cluster_speakers(fingerprints)

        speaker_segments = defaultdict(list)
        for i, label in enumerate(speaker_labels):
            if label >= 0:
                speaker_segments[label].append(segments[i])

        num_speakers = len(speaker_segments)
        await msg.edit_text(f"✅ {num_speakers} کارەکتەر دۆزرایەوە!\n⏳ ڤیدیۆکان دروست دەکرێن...")

        for spk_id, segs in sorted(speaker_segments.items()):
            filter_parts = []
            concat_inputs = []
            for j, seg in enumerate(segs):
                filter_parts.append(
                    f"[0:v]trim=start={seg['start']}:end={seg['end']},setpts=PTS-STARTPTS[v{j}];"
                    f"[0:a]atrim=start={seg['start']}:end={seg['end']},asetpts=PTS-STARTPTS[a{j}]"
                )
                concat_inputs.append(f"[v{j}][a{j}]")

            n = len(segs)
            filter_complex = ";".join(filter_parts)
            filter_complex += f";{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"

            out_path = os.path.join(tmpdir, f"speaker_{spk_id}.mp4")
            try:
                subprocess.run([
                    "ffmpeg", "-i", video_path,
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]",
                    "-c:v", "libx264", "-c:a", "aac",
                    out_path, "-y"
                ], check=True, capture_output=True)

                with open(out_path, "rb") as vf:
                    await update.message.reply_video(
                        video=vf,
                        caption=f"🎭 کارەکتەر {spk_id + 1} — {n} ڕستە"
                    )
            except:
                await update.message.reply_text(f"⚠️ کارەکتەر {spk_id + 1} هەڵەیەک هەبوو")

        await msg.edit_text(f"✅ تەواو بوو! {num_speakers} ڤیدیۆ نارد 🎉")
        user_sessions.pop(user_id, None)


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    print("بۆتەکە دەستی پێکرد ✅")
    app.run_polling()


if __name__ == "__main__":
    main()
