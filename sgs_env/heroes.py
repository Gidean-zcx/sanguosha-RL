from __future__ import annotations
from typing import Dict

# Ten common heroes (simplified abilities)
# - zhangfei: paoxiao (unlimited sha)
# - guanyu: wusheng (red card as sha when playing)
# - zhaoyun: longdan (sha<->shan conversion)
# - machao: tieqi (sha judge: black -> defender cannot use shan)
# - caocao: jianxiong (on damage draw1)
# - simayi: fankui (on damage obtain 1 card from attacker if any)
# - xiahoudun: ganglie (on damage by sha, judge red -> attacker -1hp)
# - zhugeliang: kongcheng (cannot be target of sha/duel with no hand)
# - sunquan: zhiheng (ACTIVE: omitted for fixed action space)
# - huatuo: qingnang (ACTIVE: omitted)

HEROES = [
    "zhangfei",
    "guanyu",
    "zhaoyun",
    "machao",
    "caocao",
    "simayi",
    "xiahoudun",
    "zhugeliang",
    "sunquan",
    "huatuo",
]


def hero_flags(name: str) -> Dict[str, bool]:
    return {
        "paoxiao": name == "zhangfei",
        "wusheng": name == "guanyu",
        "longdan": name == "zhaoyun",
        "tieqi": name == "machao",
        "jianxiong": name == "caocao",
        "fankui": name == "simayi",
        "ganglie": name == "xiahoudun",
        "kongcheng": name == "zhugeliang",
    }