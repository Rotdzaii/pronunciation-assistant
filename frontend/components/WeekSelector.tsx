import { useMemo, useState } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { colors } from './AppUI';
import { useTheme } from '../lib/theme';
import {
  WeekInMonth,
  formatDateRange,
  formatMonthYear,
  getWeeksOfMonth,
} from '../lib/dateRange';

type WeekSelectorProps = {
  selectedWeek: WeekInMonth;
  onChange: (week: WeekInMonth) => void;
};

export function WeekSelector({ selectedWeek, onChange }: WeekSelectorProps) {
  const { theme, mode } = useTheme();
  const [open, setOpen] = useState(false);
  const [visibleMonth, setVisibleMonth] = useState(() => ({
    year: selectedWeek.start.getFullYear(),
    monthIndex: selectedWeek.start.getMonth(),
  }));

  const weeks = useMemo(
    () => getWeeksOfMonth(visibleMonth.year, visibleMonth.monthIndex),
    [visibleMonth.monthIndex, visibleMonth.year],
  );

  const changeMonth = (delta: number) => {
    const next = new Date(visibleMonth.year, visibleMonth.monthIndex + delta, 1);
    const nextMonth = {
      year: next.getFullYear(),
      monthIndex: next.getMonth(),
    };
    const nextWeeks = getWeeksOfMonth(nextMonth.year, nextMonth.monthIndex);
    setVisibleMonth(nextMonth);
    if (nextWeeks[0]) {
      onChange(nextWeeks[0]);
    }
  };

  return (
    <View style={styles.wrap}>
      <Pressable
        accessibilityRole="button"
        onPress={() => setOpen((value) => !value)}
        style={[
          styles.trigger,
          {
            backgroundColor: theme.softBlue,
            borderColor: mode === 'dark' ? theme.border : '#BFDBFE',
          },
        ]}
      >
        <View style={styles.triggerCopy}>
          <View style={styles.triggerTopRow}>
            <Text style={[styles.triggerTitle, { color: theme.primary }]}>{selectedWeek.label}</Text>
            {selectedWeek.isCurrent ? (
              <Text style={[styles.currentBadge, { backgroundColor: theme.surfaceAlt, color: theme.primary }]}>Tuần hiện tại</Text>
            ) : null}
          </View>
          <Text style={[styles.triggerRange, { color: theme.text }]}>{formatDateRange(selectedWeek)}</Text>
        </View>
        <MaterialCommunityIcons name={open ? 'chevron-up' : 'chevron-down'} size={20} color={theme.primary} />
      </Pressable>

      <Modal transparent visible={open} animationType="fade" onRequestClose={() => setOpen(false)}>
        <View style={styles.modalRoot}>
          <Pressable accessibilityRole="button" accessibilityLabel="Đóng chọn tuần" onPress={() => setOpen(false)} style={styles.modalBackdrop} />
          <View style={[styles.dropdown, { backgroundColor: theme.surface, borderColor: theme.border, shadowColor: theme.shadow }]}>
            <View style={styles.monthRow}>
              <Pressable accessibilityRole="button" onPress={() => changeMonth(-1)} style={[styles.monthButton, { backgroundColor: theme.softBlue }]}>
                <MaterialCommunityIcons name="chevron-left" size={18} color={theme.primary} />
                <Text style={[styles.monthButtonText, { color: theme.primary }]}>Tháng trước</Text>
              </Pressable>
              <Text style={[styles.monthTitle, { color: theme.text }]}>{formatMonthYear(new Date(visibleMonth.year, visibleMonth.monthIndex, 1))}</Text>
              <Pressable accessibilityRole="button" onPress={() => changeMonth(1)} style={[styles.monthButton, { backgroundColor: theme.softBlue }]}>
                <Text style={[styles.monthButtonText, { color: theme.primary }]}>Tháng sau</Text>
                <MaterialCommunityIcons name="chevron-right" size={18} color={theme.primary} />
              </Pressable>
            </View>

            <ScrollView style={styles.weekScroll} contentContainerStyle={styles.weekList}>
              {weeks.map((week) => {
                const active = week.value === selectedWeek.value;
                return (
                  <Pressable
                    key={week.value}
                    accessibilityRole="button"
                    onPress={() => {
                      onChange(week);
                      setOpen(false);
                    }}
                    style={[
                      styles.weekOption,
                      {
                        backgroundColor: active ? theme.softBlue : theme.surface,
                        borderColor: active ? theme.primary : theme.border,
                      },
                    ]}
                  >
                    <View style={styles.weekOptionCopy}>
                      <Text style={[styles.weekOptionTitle, { color: active ? theme.primary : theme.text }]}>
                        {week.label}
                      </Text>
                      <Text style={[styles.weekOptionRange, { color: active ? theme.text : theme.textMuted }]}>
                        {formatDateRange(week)}
                      </Text>
                    </View>
                    {week.isCurrent ? (
                      <Text style={[styles.weekBadge, { backgroundColor: theme.surfaceAlt, color: theme.primary }]}>Tuần hiện tại</Text>
                    ) : null}
                  </Pressable>
                );
              })}
              {weeks.length === 0 ? <Text style={[styles.emptyText, { color: theme.textMuted }]}>Không có tuần trong tháng này.</Text> : null}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    minWidth: 190,
    position: 'relative',
    zIndex: 9999,
    elevation: 20,
  },
  trigger: {
    minHeight: 54,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#BFDBFE',
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  triggerCopy: {
    flex: 1,
    gap: 2,
  },
  triggerTopRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 6,
  },
  triggerTitle: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  triggerRange: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  currentBadge: {
    borderRadius: 999,
    backgroundColor: '#DBEAFE',
    color: colors.primary,
    fontSize: 10,
    fontWeight: '900',
    paddingHorizontal: 7,
    paddingVertical: 3,
  },
  modalRoot: {
    flex: 1,
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    zIndex: 10000,
    elevation: 30,
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(15, 23, 42, 0.02)',
  },
  dropdown: {
    position: 'absolute',
    top: 96,
    right: 24,
    width: 330,
    maxHeight: 360,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#DCE3F0',
    backgroundColor: '#FFFFFF',
    padding: 10,
    gap: 10,
    zIndex: 10000,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 20,
  },
  monthRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  monthButton: {
    minHeight: 34,
    borderRadius: 8,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: 8,
  },
  monthButtonText: {
    color: colors.primary,
    fontSize: 11,
    fontWeight: '900',
  },
  monthTitle: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  weekScroll: {
    maxHeight: 280,
  },
  weekList: {
    gap: 6,
    paddingBottom: 2,
  },
  weekOption: {
    minHeight: 54,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  weekOptionCopy: {
    flex: 1,
    gap: 2,
  },
  weekOptionActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  weekOptionTitle: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  weekOptionTitleActive: {
    color: colors.primary,
  },
  weekOptionRange: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  weekOptionRangeActive: {
    color: colors.text,
  },
  weekBadge: {
    borderRadius: 999,
    backgroundColor: '#DBEAFE',
    color: colors.primary,
    fontSize: 10,
    fontWeight: '900',
    paddingHorizontal: 7,
    paddingVertical: 3,
  },
  emptyText: {
    color: colors.muted,
    fontSize: 13,
    padding: 10,
  },
});
