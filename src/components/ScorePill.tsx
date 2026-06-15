export function ScorePill({ label, value }: { label: string; value: number }) {
  return (
    <span className="score-pill">
      {label} {Math.round(value * 100)}%
    </span>
  );
}
