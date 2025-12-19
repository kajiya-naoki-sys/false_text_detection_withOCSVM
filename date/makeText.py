# synth_misinfo_dataset.py
import argparse
import csv
import random
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Sample:
    text: str
    label: int  # 1=misinformation, 0=reliable


def pick(rng: random.Random, xs: List[str]) -> str:
    return rng.choice(xs)


def generate_samples(n: int, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)

    # Fictional entities only (avoid real-world defamation / factual disputes)
    cities = [
        "Northbridge", "Arbor City", "Silverhaven", "Lakehurst", "Redwood Bay",
        "Eastport", "Westmere", "Pineford", "Kingswell", "Brighton Hollow"
    ]
    orgs = [
        "Civic Data Office", "Public Safety Bureau", "Harbor Transport Authority",
        "National Weather Service Lab", "Institute of Materials Research",
        "Election Integrity Commission", "Metropolitan Utilities Agency",
        "Center for Digital Trust", "Transit Oversight Board", "City Health Council"
    ]
    people = [
        "A. Monroe", "J. Kline", "R. Patel", "S. Chen", "M. Alvarez",
        "T. Okafor", "L. Novak", "D. Harris", "N. Ito", "P. Singh"
    ]
    topics = [
        "transport delays", "budget revisions", "school schedule changes",
        "power grid maintenance", "local policy updates", "weather advisories",
        "public works projects", "community grants", "cybersecurity drills",
        "water quality reporting"
    ]
    evidence = [
        "a public statement", "an audited report", "a signed memo",
        "a recorded briefing", "a published dataset", "a meeting transcript",
        "an independent review", "a press conference recording", "a regulatory filing"
    ]
    dates = [
        "Dec 3, 2025", "Nov 18, 2025", "Oct 9, 2025", "Sep 27, 2025",
        "Aug 14, 2025", "Jul 2, 2025", "Jun 11, 2025", "May 6, 2025"
    ]

    # Style markers (misinfo tends to use vague sources, urgency, absolutes)
    rumor_sources = [
        "a friend of a friend", "an unnamed insider", "a hidden document",
        "a leaked screenshot", "people are saying", "a viral post", "a private chat log"
    ]
    sensational = [
        "SHOCKING", "BREAKING", "EXPOSED", "THEY DON'T WANT YOU TO KNOW",
        "WAKE UP", "This changes EVERYTHING", "UNBELIEVABLE"
    ]
    hedges = [
        "apparently", "supposedly", "allegedly", "it seems", "rumor has it", "some claim"
    ]

    reliable_templates = [
        # Calm, sourced, specific
        "[{date}] {org} in {city} released {evidence} confirming {topic}. The update notes {detail}.",
        "[{date}] Officials from {org} in {city} addressed {topic} during a briefing, citing {evidence}. They reported {detail}.",
        "[{date}] An independent review referenced by {org} in {city} found {detail} related to {topic}.",
        "[{date}] {org} published a summary on {topic} for {city}, based on {evidence}. Key figure: {number}.",
    ]

    misinfo_templates = [
        # Vague, urgent, overconfident, conspiratorial—but about fictional events
        "{sensational}! {hedge}, {city} is hiding the truth about {topic}. Source: {rumor_source}. Share before it gets deleted!",
        "{sensational}: An {rumor_source} claims {org} staged {topic} in {city}. No one is talking about this!",
        "{hedge} the numbers are fake—{org} is manipulating {topic} in {city}. I saw {rumor_source}.",
        "Everyone must read this: {city} will \"collapse\" because of {topic}. {rumor_source} confirms it. Don't trust officials!",
    ]

    detail_pool = [
        "a two-week timeline adjustment and a revised staffing plan",
        "a temporary reroute affecting three bus lines and two stations",
        "a minor reporting error corrected in the latest release",
        "a scheduled maintenance window between 01:00 and 04:00 local time",
        "a revised eligibility rule for a small community grant program",
        "a security exercise planned with no impact on public services",
        "an updated FAQ clarifying the earlier announcement"
    ]

    def make_number() -> str:
        # plausible but fictional numbers
        return f"{rng.randint(3, 48)}%"

    samples: List[Sample] = []
    for i in range(n):
        is_misinfo = (rng.random() < 0.5)  # balanced by default
        city = pick(rng, cities)
        org = pick(rng, orgs)
        topic = pick(rng, topics)
        date = pick(rng, dates)

        if not is_misinfo:
            tpl = pick(rng, reliable_templates)
            text = tpl.format(
                date=date,
                org=org,
                city=city,
                topic=topic,
                evidence=pick(rng, evidence),
                detail=pick(rng, detail_pool),
                number=make_number(),
                person=pick(rng, people),
            )
            label = 0
        else:
            tpl = pick(rng, misinfo_templates)
            text = tpl.format(
                sensational=pick(rng, sensational),
                hedge=pick(rng, hedges),
                city=city,
                org=org,
                topic=topic,
                rumor_source=pick(rng, rumor_sources),
            )
            label = 1

        samples.append(Sample(text=text, label=label))

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="synthetic_misinfo.csv")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = generate_samples(args.n, seed=args.seed)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])  # label: 1=misinfo, 0=reliable
        for s in samples:
            w.writerow([s.text, s.label])

    # quick stats
    n1 = sum(s.label for s in samples)
    n0 = len(samples) - n1
    print(f"Wrote {len(samples)} samples to {args.out}")
    print(f"label=1 (misinfo): {n1}, label=0 (reliable): {n0}")


if __name__ == "__main__":
    main()
