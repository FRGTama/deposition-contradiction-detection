import type { Claim } from "../types";

export function ClaimBlock({ title, claim }: { title: string; claim: Claim }) {
  return (
    <div className="claim-block">
      <span className="claim-title">{title}</span>
      <blockquote>{claim.evidence}</blockquote>
      <dl>
        <div>
          <dt>Relation</dt>
          <dd>
            {claim.relation} {claim.object}
          </dd>
        </div>
        <div>
          <dt>Family</dt>
          <dd>{claim.relationFamily}</dd>
        </div>
        <div>
          <dt>Polarity</dt>
          <dd>{claim.polarity}</dd>
        </div>
      </dl>
    </div>
  );
}
