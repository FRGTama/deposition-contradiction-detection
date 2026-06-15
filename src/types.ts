export type ContradictionType = "DIRECT" | "INFERENTIAL" | "FALSE_POSITIVE";

export type RelationFamily =
  | "location_at_time"
  | "movement"
  | "sleep_time"
  | "ownership"
  | "knowledge"
  | "contact"
  | "action"
  | "unknown";

export interface Claim {
  id: string;
  source: "first" | "second";
  topic: string;
  subject: string;
  relation: string;
  relationFamily: RelationFamily;
  object: string;
  polarity: "affirmed" | "negated" | "unknown";
  uncertaintyMarkers: string[];
  evidence: string;
  time?: {
    raw: string;
    minutes?: number;
    approximate: boolean;
  };
  location?: string;
}

export interface Contradiction {
  fr: number;
  fu: number;
  confidence: number;
  topicScore: number;
  type: ContradictionType;
  severity: "HIGH" | "MEDIUM" | "LOW";
  rationale: string;
  claim1: Claim;
  claim2: Claim;
}

export interface AnalysisResult {
  claims: Claim[];
  contradictions: Contradiction[];
  provider: string;
  model: string;
  embeddingProvider: string;
  embeddingModel: string;
}
