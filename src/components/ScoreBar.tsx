export function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="score-bar">
      <div className="score-label">
        <span>{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <div className="track">
        <div style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}
