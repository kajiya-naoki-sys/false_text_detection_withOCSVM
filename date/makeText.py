# date/makeText.py
import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class Sample:
    text: str
    label: int  # 1=misinfo, 0=reliable


def pick(rng: random.Random, xs: List[str]) -> str:
    return rng.choice(xs)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def generate_samples(n: int, seed: int, difficulty: str, label_noise: float) -> List[Sample]:
    rng = random.Random(seed)

    # Fictional-only entities (安全のため実在を使わない)
    cities = [
        "Northbridge", "Arbor City", "Silverhaven", "Lakehurst", "Redwood Bay",
        "Eastport", "Westmere", "Pineford", "Kingswell", "Brighton Hollow"
    ]
    orgs = [
        "Civic Data Office", "Public Safety Bureau", "Harbor Transport Authority",
        "Institute of Materials Research", "Election Integrity Commission",
        "Metropolitan Utilities Agency", "Center for Digital Trust",
        "Transit Oversight Board", "City Health Council", "Budget Review Panel"
    ]
    topics = [
        "transport delays", "budget revisions", "school schedule changes",
        "power grid maintenance", "local policy updates", "weather advisories",
        "public works projects", "community grants", "cybersecurity drills",
        "water quality reporting"
    ]
    dates = [
        "Dec 3, 2025", "Nov 18, 2025", "Oct 9, 2025", "Sep 27, 2025",
        "Aug 14, 2025", "Jul 2, 2025", "Jun 11, 2025", "May 6, 2025"
    ]

    # Evidence-like words (両クラスで共有して重なりを増やす)
    evidence = [
        "a public statement", "an audited report", "a signed memo",
        "a recorded briefing", "a published dataset", "a meeting transcript",
        "an internal bulletin", "a quarterly update", "a technical note"
    ]

    # “rumor-ish” sources (ただし hard では露骨さを落とす)
    rumor_sources_strong = [
        "a friend of a friend", "an unnamed insider", "a hidden document",
        "a leaked screenshot", "people are saying", "a viral post", "a private chat log"
    ]
    rumor_sources_soft = [
        "unverified accounts", "a circulating document", "a forwarded message",
        "a screenshot shared online", "several posts", "an unsourced thread"
    ]

    # Tone markers
    sensational = [
        "BREAKING", "EXPOSED", "UNBELIEVABLE", "WAKE UP", "This changes everything"
    ]
    hedges = [
        "apparently", "supposedly", "allegedly", "it seems", "rumor has it", "some claim"
    ]

    # Shared neutral phrases (両クラスで出る)
    neutral_phrases = [
        "according to an update", "in a follow-up", "in a new report",
        "as part of routine operations", "in response to questions",
        "in a short statement", "in an earlier notice"
    ]

    # Details to make content plausible
    detail_pool = [
        "a two-week timeline adjustment and a revised staffing plan",
        "a temporary reroute affecting three routes and two stops",
        "a minor reporting error corrected in the latest release",
        "a scheduled maintenance window between 01:00 and 04:00 local time",
        "a revised eligibility rule for a small community grant program",
        "a security exercise planned with no impact on public services",
        "an updated FAQ clarifying the earlier announcement",
        "a change affecting only a limited set of neighborhoods"
    ]

    # Some “numbers” (fake but plausible)
    def make_number() -> str:
        return f"{rng.randint(3, 48)}%"

    # Difficulty knobs
    # style_flip: 正ラベルでも偽情報っぽい文体、負ラベルでも正情報っぽい文体を混ぜる確率
    # debunk_rate: 正情報(0)に「噂の否定/訂正」文を入れる比率（hardほど増）
    # pseudo_evidence_rate: 偽情報(1)に証拠っぽい語彙を混ぜる比率（hardほど増）
    if difficulty == "easy":
        style_flip = 0.05
        debunk_rate = 0.05
        pseudo_evidence_rate = 0.05
        rumor_sources = rumor_sources_strong
        sensational_rate = 0.75
    elif difficulty == "medium":
        style_flip = 0.20
        debunk_rate = 0.18
        pseudo_evidence_rate = 0.18
        rumor_sources = rumor_sources_soft + rumor_sources_strong[:2]
        sensational_rate = 0.35
    elif difficulty == "hard":
        style_flip = 0.35
        debunk_rate = 0.30
        pseudo_evidence_rate = 0.35
        rumor_sources = rumor_sources_soft  # 露骨さを落とす
        sensational_rate = 0.12
    else:
        raise ValueError("difficulty must be one of: easy, medium, hard")

    # Templates
    reliable_templates = [
        "[{date}] {org} in {city} released {evidence} on {topic}; {neutral}. It notes {detail}.",
        "[{date}] Officials from {org} in {city} addressed {topic} during a briefing and cited {evidence}. Key figure: {number}.",
        "[{date}] {org} published a summary for {city} on {topic} based on {evidence}; {neutral}.",
        "{org} in {city} issued {evidence} about {topic} on {date}. The notice describes {detail}."
    ]

    # Debunk/Correction (label=0 だけど噂語彙が出る)
    debunk_templates = [
        "[{date}] {org} in {city} said a circulating claim about {topic} is false; {neutral}. It referenced {evidence}.",
        "{org} in {city} on {date} corrected misinformation about {topic}, noting {detail}. The correction cites {evidence}.",
        "After rumors about {topic} spread, {org} in {city} stated on {date} that the claim was inaccurate and provided {evidence}."
    ]

    misinfo_templates_strong = [
        "{sensational}! {hedge}, {city} is hiding the truth about {topic}. Source: {rumor_source}. Share before it gets deleted!",
        "{sensational}: {rumor_source} claims {org} staged {topic} in {city}. No one is talking about this.",
        "{hedge} the numbers are fake—{org} is manipulating {topic} in {city}. I saw {rumor_source}.",
        "Everyone must read this: {city} will \"collapse\" because of {topic}. {rumor_source} confirms it."
    ]

    # “Sophisticated” misinfo: 落ち着いた文体 + 証拠っぽい語彙（label=1）
    misinfo_templates_soft = [
        "[{date}] Sources suggest {org} in {city} quietly altered {topic}; {neutral}. The change was mentioned in {pseudo_evidence}.",
        "{org} in {city} allegedly revised {topic} on {date}. A summary circulating online cites {pseudo_evidence}, but no official link is provided.",
        "A report shared online claims {topic} in {city} was manipulated by {org}; {neutral}. It references {pseudo_evidence} without verification.",
        "Several posts claim {org} in {city} changed {topic} on {date}; {neutral}. The posts reference {pseudo_evidence}."
    ]

    # Cross-style: 正情報でもSNS風/短文を混ぜる（label=0）
    reliable_social_templates = [
        "Update from {org} in {city} ({date}): {topic}. {detail}. See {evidence}.",
        "{date} — {org} in {city} clarified {topic}. {detail}.",
        "Quick note ({date}): {org} in {city} posted {evidence} about {topic}. {detail}."
    ]

    def render_reliable() -> str:
        city = pick(rng, cities)
        org = pick(rng, orgs)
        topic = pick(rng, topics)
        date = pick(rng, dates)
        base = {
            "city": city,
            "org": org,
            "topic": topic,
            "date": date,
            "evidence": pick(rng, evidence),
            "detail": pick(rng, detail_pool),
            "neutral": pick(rng, neutral_phrases),
            "number": make_number()
        }

        # debunkを入れる（hardほど増）
        if rng.random() < debunk_rate:
            tpl = pick(rng, debunk_templates)
            return tpl.format(**base)

        # SNS風も混ぜる（難易度が上がるほど増えやすい）
        if rng.random() < style_flip:
            tpl = pick(rng, reliable_social_templates)
            return tpl.format(**base)

        tpl = pick(rng, reliable_templates)
        return tpl.format(**base)

    def render_misinfo() -> str:
        city = pick(rng, cities)
        org = pick(rng, orgs)
        topic = pick(rng, topics)
        date = pick(rng, dates)
        base = {
            "city": city,
            "org": org,
            "topic": topic,
            "date": date,
            "hedge": pick(rng, hedges),
            "rumor_source": pick(rng, rumor_sources),
            "neutral": pick(rng, neutral_phrases),
            "pseudo_evidence": pick(rng, evidence),
            "sensational": pick(rng, sensational),
            "detail": pick(rng, detail_pool)
        }

        # hardほど「もっともらしい偽情報」を増やす
        use_soft = (rng.random() < pseudo_evidence_rate)

        # 露骨な煽りを減らす（hardほど）
        if not use_soft and rng.random() < sensational_rate:
            tpl = pick(rng, misinfo_templates_strong)
            return tpl.format(**base)

        # 強テンプレでも煽りを弱めた版（sensationalを落とす）
        if not use_soft:
            # 強テンプレから煽り要素を落とした文（過度に簡単にならないように）
            tpl = pick(rng, [
                "{hedge}, {city} may be hiding details about {topic}. Source: {rumor_source}.",
                "{rumor_source} claims {org} influenced {topic} in {city}; {neutral}.",
                "Some posts suggest {org} manipulated {topic} in {city}. The claim cites {rumor_source}."
            ])
            return tpl.format(**base)

        tpl = pick(rng, misinfo_templates_soft)
        return tpl.format(**base)

    samples: List[Sample] = []
    for _ in range(n):
        # base label balanced 50/50
        label = 1 if rng.random() < 0.5 else 0

        if label == 0:
            text = render_reliable()
        else:
            text = render_misinfo()

        # optional label noise (hardで少し入れるとさらに難しくなる)
        if label_noise > 0.0 and rng.random() < clamp01(label_noise):
            label = 1 - label

        samples.append(Sample(text=text, label=label))

    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/synthetic_misinfo.csv")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--label_noise", type=float, default=0.0,
                    help="0.0〜0.2程度を推奨（例: 0.02=2%だけラベル反転）")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(
        n=args.n,
        seed=args.seed,
        difficulty=args.difficulty,
        label_noise=args.label_noise
    )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for s in samples:
            w.writerow([s.text, s.label])

    n1 = sum(s.label for s in samples)
    n0 = len(samples) - n1
    print(f"Wrote {len(samples)} samples to {out_path.as_posix()}")
    print(f"label=1 (misinfo): {n1}, label=0 (reliable): {n0}")
    print(f"difficulty={args.difficulty}, label_noise={args.label_noise}")


if __name__ == "__main__":
    main()
