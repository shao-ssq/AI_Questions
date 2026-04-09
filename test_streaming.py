# coding=utf-8
"""
流式 ASR API 测试客户端。

用法:
    # 使用 WAV 文件测试（通过发送切片模拟流式传输）
    python test_streaming.py --audio path/to/audio.wav

    # 使用合成正弦波测试（无需音频文件）
    python test_streaming.py --sine

    # 自定义服务器地址和切片时长
    python test_streaming.py --audio audio.wav --url http://localhost:8000 --chunk-ms 500

注意:
    服务器会在 3 秒未收到语音包后自动结束会话。
    无需显式调用 /api/finish。
"""
import argparse
import struct
import time
import wave

import numpy as np
import requests


def load_wav_as_float32_16k(path: str) -> np.ndarray:
    """加载 WAV 文件并重采样到 16kHz 单声道 float32。"""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # 解析 PCM 数据
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的采样位宽：{sampwidth}")

    # 混音为单声道
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # 重采样到 16kHz（线性插值）
    if framerate != 16000:
        ratio = 16000 / framerate
        out_len = max(0, round(len(samples) * ratio))
        indices = np.arange(out_len) / ratio
        idx0 = np.floor(indices).astype(int)
        idx1 = np.minimum(idx0 + 1, len(samples) - 1)
        t = indices - idx0
        samples = samples[idx0] * (1 - t) + samples[idx1] * t

    return samples.astype(np.float32)


def make_sine_wave(duration_sec: float = 3.0, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    """生成用于测试的正弦波（非真实语音，但可测试 API）。"""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)


def run_streaming_test(audio: np.ndarray, base_url: str, chunk_ms: int, realtime: bool):
    sr = 16000
    chunk_samples = int(sr * chunk_ms / 1000)
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    print(f"\n{'='*60}")
    print(f"Server  : {base_url}")
    print(f"Audio   : {len(audio)/sr:.2f}s  ({len(audio)} samples @ 16kHz)")
    print(f"Chunk   : {chunk_ms}ms  ({chunk_samples} samples)")
    print(f"Chunks  : {total_chunks}")
    print(f"Realtime: {realtime}")
    print(f"{'='*60}\n")

    # ── 1. 开始会话 ──────────────────────────────────────────
    print("[1/3] POST /api/start")
    r = requests.post(f"{base_url}/api/start", timeout=10)
    r.raise_for_status()
    session_id = r.json()["session_id"]
    print(f"      session_id = {session_id}\n")

    # ── 2. 发送切片 ────────────────────────────────────────────
    print("[2/3] POST /api/chunk  (流式传输中...)")
    t0 = time.time()
    last_text = ""

    for i in range(total_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]

        r = requests.post(
            f"{base_url}/api/chunk",
            params={"session_id": session_id},
            headers={"Content-Type": "application/octet-stream"},
            data=chunk.tobytes(),
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()

        text = result.get("text", "")
        lang = result.get("language", "")
        finalized = result.get("finalized_segments", [])
        vad_status = result.get("vad_status", {})
        elapsed = time.time() - t0

        # 检查 VAD 断句
        vad_icon = "🎤" if vad_status.get("is_speech", False) else "🔇"
        silence_ms = vad_status.get("silence_ms", 0)
        is_start = vad_status.get("is_start", False)
        is_end = vad_status.get("is_end", False)
        segment_index = vad_status.get("segment_index", 0)

        # 仅在文本变化或新句子断句时打印
        if text != last_text or finalized:
            status_line = f"  [{elapsed:5.1f}s] {vad_icon} chunk {i+1:3d}/{total_chunks}"
            status_line += f"  lang={lang!r:8s}  text={text!r}"
            flags = []
            if is_start:
                flags.append("▶️ START")
            if is_end:
                flags.append("⏹️ END")
            if flags:
                status_line += f"  [{' '.join(flags)}]"
            if finalized:
                status_line += f"\n      ✂️ 断句[{segment_index}]：{finalized[-1]!r} (共{len(finalized)}句)"
            print(status_line)
            last_text = text
        else:
            print(f"  [{elapsed:5.1f}s] {vad_icon} chunk {i+1:3d}/{total_chunks}  (silence={silence_ms:.0f}ms)", end="\r")

        if realtime:
            time.sleep(chunk_ms / 1000)

    print()

    # ── 3. 等待自动结束 ───────────────────────────────────────────
    # 服务器会在 3 秒未收到语音包后自动结束会话。
    # 短暂等待以让服务器处理完最后的切片。
    print("[3/3] 等待服务器自动结束（3 秒超时）...")
    time.sleep(3.5)
    print("      完成。会话应在服务器端自动结束。")

    print(f"\n{'='*60}")
    print(f"Total time     : {time.time() - t0:.2f}s")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description="Qwen3-ASR 流式 API 测试客户端")
    p.add_argument("--url", default="http://172.23.32.85:8000/", help="服务器基础地址")
    p.add_argument("--audio", default="yuan.WAV")
    p.add_argument("--sine", action="store_true", help="使用合成正弦波代替 WAV 文件")
    p.add_argument("--chunk-ms", type=int, default=500, help="切片时长（毫秒），默认 500")
    p.add_argument("--realtime", action="store_true", help="在切片之间休眠以模拟实时输入")
    args = p.parse_args()

    if args.audio:
        print(f"加载 WAV 文件：{args.audio}")
        audio = load_wav_as_float32_16k(args.audio)
    elif args.sine:
        print("使用合成正弦波（3 秒 @ 440Hz）")
        audio = make_sine_wave(duration_sec=3.0)
    else:
        p.error("请提供 --audio <file.wav> 或 --sine")

    run_streaming_test(audio, args.url.rstrip("/"), args.chunk_ms, args.realtime)


if __name__ == "__main__":
    main()