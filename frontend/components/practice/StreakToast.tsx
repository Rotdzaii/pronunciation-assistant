import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { colors, radius, spacing, typography } from '../../constants/theme';

type Props = {
  message: string;
  /** Toast lifetime in ms. Default 3000. */
  durationMs?: number;
  /** Called when the progress bar finishes — remove this toast from the list. */
  onExpire: () => void;
};

/**
 * Floating streak-milestone notification.
 *
 * Slides in from the top on mount, then a progress bar
 * runs from 100 % → 0 % over `durationMs` using Animated.timing
 * (useNativeDriver: false because we animate the `width` style property).
 * Calls `onExpire` when the animation completes.
 */
export default function StreakToast({ message, durationMs = 3000, onExpire }: Props) {
  const slideY    = useRef(new Animated.Value(-88)).current;
  const progress  = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    // Slide in (native driver — translateY)
    Animated.spring(slideY, {
      toValue: 0,
      damping: 18,
      stiffness: 200,
      useNativeDriver: true,
    }).start();

    // Progress bar shrinks to 0 over durationMs then fires onExpire
    Animated.timing(progress, {
      toValue: 0,
      duration: durationMs,
      useNativeDriver: false,
    }).start(({ finished }) => {
      if (finished) onExpire();
    });
  // run once on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const barWidth = progress.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <Animated.View style={[styles.card, { transform: [{ translateY: slideY }] }]}>
      <View style={styles.row}>
        <MaterialCommunityIcons name="fire" size={22} color={colors.warning} />
        <Text style={styles.message} numberOfLines={2}>{message}</Text>
      </View>
      <View style={styles.track}>
        <Animated.View style={[styles.bar, { width: barWidth }]} />
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.white,
    borderRadius: radius.xl,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.14,
    shadowRadius: 16,
    elevation: 8,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.xs,
  },
  message: {
    ...typography.labelBold,
    color: colors.onSurface,
    flex: 1,
  },
  track: {
    height: 3,
    backgroundColor: colors.slate200,
    borderRadius: 2,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    backgroundColor: colors.micActive,
    borderRadius: 2,
  },
});
