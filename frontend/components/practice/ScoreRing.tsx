import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import { colors } from '../../constants/theme';

type Props = {
  /** Score value 0–100. */
  score: number;
  /** Outer diameter in dp. Default 160 (matches w-40 in result.html). */
  size?: number;
  /** Ring stroke width. Default 12 (matches border-[12px] in result.html). */
  trackWidth?: number;
};

/**
 * Circular progress ring for pronunciation score.
 * Based on the score-circle / conic-gradient in result.html.
 *
 * Uses react-native-svg (two Circle elements with strokeDasharray/Offset)
 * to avoid the clipping artefacts of the pure-RN half-circle technique.
 *
 *  - Track circle : full 360°, grey (slate200)
 *  - Progress arc : starts at 12 o'clock (rotation -90°), clockwise,
 *                   strokeLinecap="round" for the rounded tip
 *  - Colour: ≥70 → success green, ≥40 → warning amber, <40 → danger red
 */
export default function ScoreRing({ score, size = 160, trackWidth = 12 }: Props) {
  const pct = Math.min(1, Math.max(0, score / 100));

  // Radius sits at the stroke centre so the full ring stays inside the SVG bounds.
  const r = size / 2 - trackWidth / 2;
  const circumference = 2 * Math.PI * r;
  // offset=0 → full circle, offset=circumference → nothing drawn
  const dashOffset = circumference * (1 - pct);

  const progressColor =
    score >= 70 ? colors.success : score >= 40 ? colors.warning : colors.danger;
  const textColor =
    score >= 70 ? colors.successDark : score >= 40 ? colors.warning : colors.danger;

  const cx = size / 2;
  const cy = size / 2;

  return (
    <View style={[styles.outer, { width: size, height: size, borderRadius: size / 2 }]}>
      <Svg width={size} height={size} style={StyleSheet.absoluteFill}>
        {/* Track — full grey ring */}
        <Circle
          cx={cx}
          cy={cy}
          r={r}
          stroke={colors.slate200}
          strokeWidth={trackWidth}
          fill="none"
        />
        {/* Progress arc — starts at 12 o'clock, clockwise */}
        <Circle
          cx={cx}
          cy={cy}
          r={r}
          stroke={progressColor}
          strokeWidth={trackWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          // rotate -90° around the circle centre so the arc starts at the top
          rotation={-90}
          origin={`${cx}, ${cy}`}
        />
      </Svg>

      {/* Centred score text overlaid on the SVG */}
      <View style={styles.center} pointerEvents="none">
        <Text style={[styles.scoreText, { color: textColor }]}>{score}</Text>
        <Text style={styles.outOf}>/ 100</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  outer: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.successLight,   // bg-emerald-50 matches result.html
  },
  center: {
    alignItems: 'center',
  },
  scoreText: {
    fontSize: 48,
    fontWeight: '900',
    lineHeight: 56,
  },
  outOf: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.slate400,
    marginTop: 2,
  },
});
