import React from 'react';
import { StyleSheet, View } from 'react-native';

type Props = {
  /** Heights in dp for each bar (left → right). */
  heights: number[];
  color: string;
  /** Bar width in dp. Default 4 (matches w-1 in result.html mini-waveform). */
  barWidth?: number;
  /** Gap between bars in dp. Default 3. */
  gap?: number;
  /** Overall opacity. Default 0.2 (matches opacity-20 in practice.html). */
  opacity?: number;
  /** 'center' aligns bars to vertical mid; 'bottom' hangs them from baseline. */
  align?: 'center' | 'bottom';
};

/**
 * Animated waveform bar visualizer (static heights — animate from outside if needed).
 *
 * practice.html usage  → opacity-20, primary color, larger bars, center-aligned
 * result.html usage    → opacity-100, waveform mock, 4 dp bars, mini heights
 */
export default function WaveformBars({
  heights,
  color,
  barWidth = 4,
  gap = 3,
  opacity = 0.2,
  align = 'center',
}: Props) {
  return (
    <View
      style={[
        styles.row,
        { opacity, alignItems: align === 'center' ? 'center' : 'flex-end', gap },
      ]}
    >
      {heights.map((h, i) => (
        <View
          key={i}
          style={[
            styles.bar,
            { height: h, width: barWidth, backgroundColor: color },
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
  },
  bar: {
    borderRadius: 999,
  },
});
