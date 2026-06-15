from __future__ import annotations

import re
from dataclasses import dataclass

from .logger import debug_log
from .models import ClaimSource


@dataclass(frozen=True)
class QABlock:
    id: str
    source: ClaimSource
    block_index: int
    question: str
    answer: str
    quote: str


def split_qa_blocks(transcript: str, source: ClaimSource) -> list[QABlock]:
    blocks: list[QABlock] = []
    matches = list(re.finditer(r"(?im)^\s*Q:\s*(?P<question>.*)$", transcript))

    for index, match in enumerate(matches):
        block_start = match.start()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        chunk = transcript[block_start:next_start].strip()
        answer_match = re.search(r"(?ims)^\s*A:\s*(?P<answer>.*)$", chunk)

        if not answer_match:
            continue

        question = _clean(match.group("question"))
        answer = _clean(answer_match.group("answer"))

        if not answer:
            continue

        blocks.append(
            QABlock(
                id=f"{source}-block-{len(blocks) + 1}",
                source=source,
                block_index=len(blocks) + 1,
                question=question,
                answer=answer,
                quote=chunk,
            )
        )

    if not blocks:
        fallback_answer = _clean(transcript)
        blocks.append(
            QABlock(
                id=f"{source}-block-1",
                source=source,
                block_index=1,
                question="",
                answer=fallback_answer,
                quote=fallback_answer,
            )
        )

    _log_blocks(source, blocks)
    return blocks


def _log_blocks(source: ClaimSource, blocks: list[QABlock]) -> None:
    debug_log(
        "qa.blocks",
        {
            "source": source,
            "count": len(blocks),
            "blocks": [
                {
                    "id": block.id,
                    "source": block.source,
                    "blockIndex": block.block_index,
                    "question": block.question,
                    "answer": block.answer,
                    "quote": block.quote,
                }
                for block in blocks
            ],
        },
    )


def _clean(value: str) -> str:
    return " ".join(value.split()).strip()
