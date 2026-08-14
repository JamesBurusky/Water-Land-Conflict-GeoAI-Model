// Color scales and category palettes used across the map (point/risk
// symbology) and other visualizations, kept in one place so choices
// stay consistent (e.g. the same topic gets the same color everywhere).

export const RISK_COLORS: Record<string, string> = {
  Low: "#2a9d8f",
  Lower: "#2a9d8f",
  Medium: "#e9c46a",
  High: "#e76f51",
  Higher: "#e76f51",
};
export const NO_DATA_COLOR = "#dddddd";

// Distinct, colorblind-considerate categorical palette for topics --
// cycles via modulo if there are more topics than colors.
export const CATEGORICAL_PALETTE = [
  "#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#8ecae6",
  "#f4a261", "#606c38", "#bc6c25", "#6a4c93", "#c9184a",
];

export function colorForTopic(topicId: number | null | undefined): string {
  if (topicId == null || topicId === -1) return NO_DATA_COLOR;
  return CATEGORICAL_PALETTE[topicId % CATEGORICAL_PALETTE.length];
}

export const SENTIMENT_COLORS: Record<string, string> = {
  negative: "#e76f51",
  neutral: "#9aa5b1",
  positive: "#2a9d8f",
};

/** Interpolates green -> yellow -> red for a 0..1 value -- used for the
 * continuous risk-probability map style and the severity point style.
 * Simple linear RGB interpolation, not perceptually uniform, but
 * sufficient (and dependency-free) for this use. */
export function colorForRatio(ratio: number): string {
  const clamped = Math.max(0, Math.min(1, ratio));
  const stops: [number, [number, number, number]][] = [
    [0, [42, 157, 143]],   // teal  (--color-teal)
    [0.5, [233, 196, 106]], // yellow (--color-yellow)
    [1, [231, 111, 81]],   // orange (--color-orange)
  ];
  let lower = stops[0], upper = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamped >= stops[i][0] && clamped <= stops[i + 1][0]) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }
  const span = upper[0] - lower[0] || 1;
  const t = (clamped - lower[0]) / span;
  const rgb = lower[1].map((c, i) => Math.round(c + t * (upper[1][i] - c)));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

export function colorForComposite(isComposite: boolean | null | undefined): string {
  return isComposite ? "#f4a261" : "#264653";
}
