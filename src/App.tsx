import { useMemo, useState } from "react";
import type { AnalysisResult, ContradictionType } from "./types";
import { countByType, labelForFilter } from "./utils";
import { ClaimBlock } from "./components/ClaimBlock";
import { Metric } from "./components/Metric";
import { ScoreBar } from "./components/ScoreBar";
import { ScorePill } from "./components/ScorePill";

const TRANSCRIPT_1 = `Deposition of Marcus Webb - March 14, 2023

Q: Where were you on the evening of November 3rd?
A: I was at home all evening. I ordered pizza around 7pm and watched TV.

Q: Did you speak to anyone that night?
A: No, I was alone. My wife was visiting her sister in Portland.

Q: What time did you go to sleep?
A: Around 10, maybe 10:30. I had work the next morning.

Q: Have you ever been to the Hargrove Street warehouse?
A: No, never. I don't even know where that is.

Q: Do you own a grey Honda Civic?
A: I did at the time, yes. I sold it in January.

Q: Had you met Daniel Cho before November 3rd?
A: No. I'd never heard of him before this whole thing started.`;

const TRANSCRIPT_2 = `Deposition of Marcus Webb - September 9, 2023

Q: Walk me through the evening of November 3rd again.
A: I was home. I think I went out briefly to get some groceries, maybe around 7:30, but came right back.

Q: You mentioned last time you ordered pizza. Now you're saying groceries?
A: I might have done both. I don't remember exactly, it was almost a year ago.

Q: Did anyone see you that evening?
A: My neighbor, Tom, might have seen me. We waved or something in the parking lot.

Q: What time did you go to sleep?
A: It was late. Midnight maybe. I had trouble sleeping.

Q: Had you ever visited the Hargrove Street area?
A: I mean, I've driven through that part of town. I didn't say I'd never been in that general area.

Q: And Daniel Cho - did you know him?
A: I knew of him. We had mutual friends. I don't think I'd met him face to face.`;

const FILTERS: Array<ContradictionType | "ALL"> = ["ALL", "DIRECT", "INFERENTIAL", "FALSE_POSITIVE"];

export default function App() {
  const [transcript1, setTranscript1] = useState(TRANSCRIPT_1);
  const [transcript2, setTranscript2] = useState(TRANSCRIPT_2);
  const [activeFilter, setActiveFilter] = useState<ContradictionType | "ALL">("ALL");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const visibleContradictions = useMemo(() => {
    if (!analysis) {
      return [];
    }

    if (activeFilter === "ALL") {
      return analysis.contradictions;
    }

    return analysis.contradictions.filter((item) => item.type === activeFilter);
  }, [activeFilter, analysis]);

  async function analyze() {
    setLoading(true);
    setError("");
    setAnalysis(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript1, transcript2 })
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? payload.error ?? "Analysis failed.");
      }

      setAnalysis(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Legal AI review</p>
          <h1>Deposition Contradiction Detector</h1>
        </div>
        <button className="primary-action" type="button" disabled={loading} onClick={analyze}>
          {loading ? "Analyzing" : "Analyze"}
        </button>
      </header>

      <section className="workspace" aria-label="Transcript inputs">
        <label className="transcript-panel">
          <span>First deposition</span>
          <textarea value={transcript1} onChange={(event) => setTranscript1(event.target.value)} />
        </label>
        <label className="transcript-panel">
          <span>Second deposition</span>
          <textarea value={transcript2} onChange={(event) => setTranscript2(event.target.value)} />
        </label>
      </section>

      {error && <div className="error-banner">{error}</div>}

      {analysis && (
        <section className="results" aria-label="Contradiction results">
          <div className="results-header">
            <div>
              <p className="eyebrow">
                {analysis.provider} / {analysis.model} | embeddings: {analysis.embeddingProvider} / {analysis.embeddingModel}
              </p>
              <h2>{analysis.contradictions.length} review candidates</h2>
            </div>
            <div className="filter-group" role="tablist" aria-label="Contradiction type">
              {FILTERS.map((filter) => (
                <button
                  className={activeFilter === filter ? "filter active" : "filter"}
                  key={filter}
                  type="button"
                  onClick={() => setActiveFilter(filter)}
                >
                  {labelForFilter(filter)}
                </button>
              ))}
            </div>
          </div>

          <div className="summary-grid">
            <Metric label="Claims" value={analysis.claims.length} />
            <Metric label="Direct" value={countByType(analysis.contradictions, "DIRECT")} />
            <Metric label="Inferential" value={countByType(analysis.contradictions, "INFERENTIAL")} />
            <Metric label="False positive" value={countByType(analysis.contradictions, "FALSE_POSITIVE")} />
          </div>

          <div className="contradiction-list">
            {visibleContradictions.map((item, index) => (
              <article className="result-card" key={`${item.claim1.id}-${item.claim2.id}-${index}`}>
                <div className="card-topline">
                  <span className={`chip ${item.type.toLowerCase()}`}>{labelForFilter(item.type)}</span>
                  <span className="severity">Severity {item.severity}</span>
                  <ScorePill label="Confidence" value={item.confidence} />
                </div>

                <p className="rationale">{item.rationale}</p>

                <div className="claim-grid">
                  <ClaimBlock title="First" claim={item.claim1} />
                  <ClaimBlock title="Second" claim={item.claim2} />
                </div>

                <div className="score-grid">
                  <ScoreBar label="fr" value={item.fr} />
                  <ScoreBar label="fu" value={item.fu} />
                  <ScoreBar label="semantic" value={item.topicScore} />
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
