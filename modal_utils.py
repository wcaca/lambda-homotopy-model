"""
modal_utils.py — 多模态基础工具（增量新增，不修改任何原有文件）
功能：文本/图像/音频 ↔ 4096维标准向量 + 8维模态指纹；多模态融合。
λ函数单元 + 同伦形变大模型系统 · 多模态适配层
"""

from __future__ import annotations
import os
import numpy as np

# ── 全局模态常量 ────────────────────────────────────────────────────────
MODAL_TEXT  = "text"
MODAL_IMAGE = "image"
MODAL_AUDIO = "audio"
STD_DIM     = 4096
FINGER_DIM  = 8

# 8维模态指纹槽位定义（前3位为模态标记，后5位预留路由扩展）
_FINGER_SLOT = {MODAL_TEXT: 0, MODAL_IMAGE: 1, MODAL_AUDIO: 2}


def _make_finger(modal_type: str) -> np.ndarray:
    """生成归一化模态指纹（标记来源模态，供路由分发）"""
    f = np.zeros(FINGER_DIM, dtype=np.float32)
    slot = _FINGER_SLOT.get(modal_type, 0)
    f[slot] = 1.0
    return f.astype(np.float16)


def _align_to_std(arr: np.ndarray) -> np.ndarray:
    """将任意长度 float 数组对齐至 STD_DIM，截断或零填充"""
    arr = arr.flatten().astype(np.float32)
    if len(arr) >= STD_DIM:
        return arr[:STD_DIM]
    return np.concatenate([arr, np.zeros(STD_DIM - len(arr), dtype=np.float32)])


# ══════════════════════════════════════════════════════════════════════
# 文本编码 / 解码
# ══════════════════════════════════════════════════════════════════════
def text_to_vector(text: str) -> tuple[np.ndarray, np.ndarray]:
    """
    文本 → 4096维主干向量 + 8维模态指纹
    编码策略：字符级哈希散列 + TF 频率加权 + L2 归一化
    （接口固定，可替换为自研编码器而不影响下游）
    """
    vec = np.zeros(STD_DIM, dtype=np.float32)
    if text:
        for i, ch in enumerate(text):
            slot = (hash(ch) + i * 31) % STD_DIM
            vec[slot] += 1.0 / (i + 1)          # 位置衰减权重
        # 再叠加词粒度
        for word in text.split():
            slot = hash(word) % STD_DIM
            vec[slot] += 0.5
        norm = np.linalg.norm(vec) + 1e-8
        vec /= norm
    return vec.astype(np.float16), _make_finger(MODAL_TEXT)


def vector_to_text(vec: np.ndarray) -> str:
    """
    4096维向量 → 文本（简易 argmax 索引解码）
    生产替换：此接口插入自研语言解码器，签名不变
    """
    idx  = int(np.argmax(np.abs(vec.astype(np.float32))))
    conf = float(np.abs(vec).max())
    return f"[Text_Slot={idx} conf={conf:.4f}]"


# ══════════════════════════════════════════════════════════════════════
# 图像编码 / 解码
# ══════════════════════════════════════════════════════════════════════
def image_to_vector(img_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    图像文件 → 4096维主干向量 + 8维模态指纹
    依赖：Pillow（pillow）
    """
    try:
        from PIL import Image
        img = Image.open(img_path).convert("L").resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        # Pillow 未安装时降级为随机填充（保证接口不中断）
        arr = np.random.rand(64, 64).astype(np.float32)
        print("[modal_utils] 警告：Pillow 未安装，图像使用随机填充")
    except FileNotFoundError:
        arr = np.zeros((64, 64), dtype=np.float32)
        print(f"[modal_utils] 警告：图像文件不存在 {img_path}")

    vec = _align_to_std(arr)
    norm = np.linalg.norm(vec) + 1e-8
    vec /= norm
    return vec.astype(np.float16), _make_finger(MODAL_IMAGE)


def vector_to_image(vec: np.ndarray, save_path: str):
    """
    4096维向量 → 灰度图像文件（64×64）
    依赖：Pillow
    """
    try:
        from PIL import Image
        arr = vec.astype(np.float32)[:64*64].reshape(64, 64)
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(arr).save(save_path)
    except ImportError:
        print("[modal_utils] 警告：Pillow 未安装，跳过图像写出")


# ══════════════════════════════════════════════════════════════════════
# 音频编码 / 解码
# ══════════════════════════════════════════════════════════════════════
def audio_to_vector(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    音频文件 → 4096维主干向量 + 8维模态指纹
    依赖：librosa + soundfile（可选）
    特征：MFCC(40) + 过零率 + 短时能量，拼接对齐至4096维
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).flatten()
        zcr     = librosa.feature.zero_crossing_rate(y).flatten()
        energy  = librosa.feature.rms(y=y).flatten()
        feat    = np.concatenate([mfcc, zcr, energy])
    except ImportError:
        feat = np.random.rand(STD_DIM).astype(np.float32)
        print("[modal_utils] 警告：librosa 未安装，音频使用随机填充")
    except Exception as e:
        feat = np.zeros(STD_DIM, dtype=np.float32)
        print(f"[modal_utils] 警告：音频加载失败 {e}")

    vec = _align_to_std(feat)
    norm = np.linalg.norm(vec) + 1e-8
    vec /= norm
    return vec.astype(np.float16), _make_finger(MODAL_AUDIO)


def vector_to_audio(vec: np.ndarray, save_path: str, sr: int = 16000):
    """
    4096维向量 → 单声道音频文件（WAV）
    依赖：soundfile
    """
    try:
        import soundfile as sf
        audio = vec.astype(np.float32)[:sr * 3]          # 最多 3 秒
        audio = audio / (np.max(np.abs(audio)) + 1e-8)   # 归一化振幅
        sf.write(save_path, audio, sr)
    except ImportError:
        print("[modal_utils] 警告：soundfile 未安装，跳过音频写出")


# ══════════════════════════════════════════════════════════════════════
# 多模态融合
# ══════════════════════════════════════════════════════════════════════
def modal_fusion(vec_list: list[np.ndarray], rule: str = "sum") -> np.ndarray:
    """
    多分支模态向量融合（复用原有 sum/concat/weight 语义）
    vec_list: 各模态编码后的 4096维向量列表
    rule:     sum | concat | weight | mean
    """
    if not vec_list:
        return np.zeros(STD_DIM, dtype=np.float16)

    arrs = [v.astype(np.float32) for v in vec_list]
    # 确保等维后再合并
    min_dim = min(a.shape[-1] for a in arrs)
    arrs = [a[..., :min_dim] for a in arrs]

    if rule == "sum":
        out = np.sum(arrs, axis=0)
    elif rule == "concat":
        out = np.concatenate(arrs, axis=-1)
    elif rule == "weight":
        # 线性递增权重：越靠后的模态权重越大（可替换为指纹相似度权重）
        w = np.linspace(0.2, 1.0, len(arrs))
        w = w / w.sum()
        out = np.sum([a * wi for a, wi in zip(arrs, w)], axis=0)
    elif rule == "mean":
        out = np.mean(arrs, axis=0)
    else:
        out = np.mean(arrs, axis=0)

    # L2 归一化，保持与单模态向量同等量级
    norm = np.linalg.norm(out) + 1e-8
    return (out / norm).astype(np.float16)


# ══════════════════════════════════════════════════════════════════════
# 模态调度路由（基于 8维指纹）
# ══════════════════════════════════════════════════════════════════════
def route_by_finger(finger: np.ndarray) -> str:
    """
    根据 8维模态指纹自动识别来源模态
    finger[0]=1 → text | finger[1]=1 → image | finger[2]=1 → audio
    """
    idx = int(np.argmax(finger.astype(np.float32)[:3]))
    return [MODAL_TEXT, MODAL_IMAGE, MODAL_AUDIO][idx]
