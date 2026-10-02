"""XXH64 —— 真机 ``StringHelper.GetDeterministicHashCode`` 用的就是它。

``StringHelper.cs:139-152``::

    public static ulong GetDeterministicHashCode(string str)
    {
        int byteCount = Encoding.UTF8.GetByteCount(str);
        byteCount = Encoding.UTF8.GetBytes(str, _stringHashCache);
        return XxHash64.HashToUInt64(_stringHashCache.AsSpan().Slice(0, byteCount), 0L);
    }

即 **UTF-8 字节 + seed=0 的 XXH64**。事件 RNG 的派生要用到它
（``EventModel.cs:234``），所以必须与 .NET 的 ``System.IO.Hashing.XxHash64`` 逐位一致。

自测向量（xxHash 官方测试集，seed=0）::

    xxh64("")    == 0xEF46DB3751D8E999
    xxh64("a")   == 0xD24EC4F1A98C6E5B
    xxh64("abc") == 0x44BC2CF5AD770999

⚠️ 这不是"随便一个哈希"：换成 ``hashlib`` 之类会得到**另一条**随机序列，
于是"同一种子下事件里的随机量"与真机不符 —— 数值对不上，且不会报错。
"""

from __future__ import annotations

_MASK = 0xFFFFFFFFFFFFFFFF
_PRIME1 = 0x9E3779B185EBCA87
_PRIME2 = 0xC2B2AE3D27D4EB4F
_PRIME3 = 0x165667B19E3779F9
_PRIME4 = 0x85EBCA77C2B2AE63
_PRIME5 = 0x27D4EB2F165667C5


def _rotl(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (64 - bits))) & _MASK


def _round(acc: int, lane: int) -> int:
    acc = (acc + lane * _PRIME2) & _MASK
    acc = _rotl(acc, 31)
    return (acc * _PRIME1) & _MASK


def _merge_round(acc: int, value: int) -> int:
    acc ^= _round(0, value)
    return (acc * _PRIME1 + _PRIME4) & _MASK


def xxh64(data: bytes, seed: int = 0) -> int:
    """XXH64（小端 lane，与参考实现一致）。"""
    length = len(data)
    index = 0
    if length >= 32:
        v1 = (seed + _PRIME1 + _PRIME2) & _MASK
        v2 = (seed + _PRIME2) & _MASK
        v3 = seed & _MASK
        v4 = (seed - _PRIME1) & _MASK
        while index + 32 <= length:
            v1 = _round(v1, int.from_bytes(data[index:index + 8], "little"))
            v2 = _round(v2, int.from_bytes(data[index + 8:index + 16], "little"))
            v3 = _round(v3, int.from_bytes(data[index + 16:index + 24], "little"))
            v4 = _round(v4, int.from_bytes(data[index + 24:index + 32], "little"))
            index += 32
        acc = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _MASK
        for lane in (v1, v2, v3, v4):
            acc = _merge_round(acc, lane)
    else:
        acc = (seed + _PRIME5) & _MASK

    acc = (acc + length) & _MASK
    while index + 8 <= length:
        lane = int.from_bytes(data[index:index + 8], "little")
        acc ^= _round(0, lane)
        acc = (_rotl(acc, 27) * _PRIME1 + _PRIME4) & _MASK
        index += 8
    if index + 4 <= length:
        acc ^= (int.from_bytes(data[index:index + 4], "little") * _PRIME1) & _MASK
        acc = (_rotl(acc, 23) * _PRIME2 + _PRIME3) & _MASK
        index += 4
    while index < length:
        acc ^= (data[index] * _PRIME5) & _MASK
        acc = (_rotl(acc, 11) * _PRIME1) & _MASK
        index += 1

    acc ^= acc >> 33
    acc = (acc * _PRIME2) & _MASK
    acc ^= acc >> 29
    acc = (acc * _PRIME3) & _MASK
    acc ^= acc >> 32
    return acc


def xxh64_text(text: str, seed: int = 0) -> int:
    """``StringHelper.GetDeterministicHashCode(str)`` 的等价实现。"""
    return xxh64(text.encode("utf-8"), seed)
