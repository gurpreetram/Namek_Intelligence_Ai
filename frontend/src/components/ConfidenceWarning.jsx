// Confidence-score honesty warning -- see the Portfolio tab's calibration
// section for the underlying analysis (risk/probability_calibration.py's
// calibrate() run against 1201 real Validated Breakout trades).
//
// Only Validated Breakout has actually been checked, and it came back
// miscalibrated (Brier score 0.502 -- worse than a flat 50% guess): a
// 1.00-confidence signal won 32.8% of the time, barely different from a
// 0.75-confidence signal's 33.2%. The other three strategies have NOT
// been checked either way -- this component deliberately does not imply
// they are fine just because we have not found a problem with them yet.
const VERIFIED_MISCALIBRATED = new Set(["validated_breakout"])

function strategyLabel(strategyName) {
  return strategyName ? strategyName.replace(/_/g, " ") : null
}

export default function ConfidenceWarning({ strategyName, style }) {
  const label = strategyLabel(strategyName)
  const isVerifiedBad = strategyName && VERIFIED_MISCALIBRATED.has(strategyName)
  const otherLabel = label ? `${label}'s confidence score has` : "Other strategies' confidence scores have"

  return (
    <div
      className="text-muted"
      style={{
        border: "1px solid var(--border)", background: "var(--surface-raised)",
        padding: "8px 12px", fontSize: 11, lineHeight: 1.5, ...style,
      }}
    >
      {isVerifiedBad ? (
        <>
          <span style={{ color: "var(--accent)" }}>{"⚠"} Confidence score not calibrated.</span> Checked
          against 1201 real {label} trades: a 1.00-confidence signal won 32.8% of the time, barely
          different from a 0.75-confidence signal's 33.2% (Brier score 0.502 -- worse than a coin
          flip). Does not reliably reflect actual win probability. See the Portfolio tab for the
          full calibration report.
        </>
      ) : (
        <>
          <span style={{ color: "var(--accent)" }}>{"⚠"} Confidence score calibration unverified.</span>{" "}
          Only Validated Breakout's confidence score has been checked against real outcomes so far,
          and it came back <b>not reliable</b> (see Portfolio tab). {otherLabel} not been tested
          either way -- absence of a bad finding here is not evidence the score is good.
        </>
      )}
    </div>
  )
}
