export type DateRange = {
  start: Date;
  end: Date;
};

export type WeekInMonth = DateRange & {
  index: number;
  label: string;
  value: string;
  isCurrent: boolean;
};

const DAY_MS = 24 * 60 * 60 * 1000;

export function startOfMondayWeek(date: Date): Date {
  const value = startOfDay(date);
  const day = value.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  value.setDate(value.getDate() + diff);
  return value;
}

export function endOfSundayWeek(date: Date): Date {
  const start = startOfMondayWeek(date);
  return new Date(start.getTime() + 6 * DAY_MS);
}

export function getCurrentWeekRange(date = new Date()): DateRange {
  return {
    start: startOfMondayWeek(date),
    end: endOfSundayWeek(date),
  };
}

export function getCurrentMonthRange(date = new Date()): DateRange {
  return {
    start: new Date(date.getFullYear(), date.getMonth(), 1),
    end: new Date(date.getFullYear(), date.getMonth() + 1, 0),
  };
}

export function formatDayMonth(date: Date): string {
  return `${pad2(date.getDate())}/${pad2(date.getMonth() + 1)}`;
}

export function formatMonthYear(date: Date): string {
  return `${pad2(date.getMonth() + 1)}/${date.getFullYear()}`;
}

export function formatDateRange(range: DateRange): string;
export function formatDateRange(startDate: Date, endDate: Date): string;
export function formatDateRange(rangeOrStartDate: DateRange | Date, endDate?: Date): string {
  const start = rangeOrStartDate instanceof Date ? rangeOrStartDate : rangeOrStartDate.start;
  const end = endDate ?? (rangeOrStartDate instanceof Date ? rangeOrStartDate : rangeOrStartDate.end);
  return `${formatDayMonth(start)} - ${formatDayMonth(end)}`;
}

export function getWeeksOfMonth(year: number, monthIndex: number, today = new Date()): WeekInMonth[] {
  const monthStart = new Date(year, monthIndex, 1);
  const monthEnd = new Date(year, monthIndex + 1, 0);
  const weeks: WeekInMonth[] = [];
  let cursor = startOfMondayWeek(monthStart);
  let index = 1;

  while (cursor <= monthEnd) {
    const start = new Date(cursor);
    const end = new Date(cursor.getTime() + 6 * DAY_MS);
    const visibleStart = start < monthStart ? monthStart : start;
    const visibleEnd = end > monthEnd ? monthEnd : end;
    const isCurrent = today >= visibleStart && today <= visibleEnd;
    weeks.push({
      index,
      start: visibleStart,
      end: visibleEnd,
      label: `Tuần ${index}`,
      value: `${year}-${pad2(monthIndex + 1)}-${index}`,
      isCurrent,
    });
    cursor = new Date(cursor.getTime() + 7 * DAY_MS);
    index += 1;
  }

  return weeks;
}

export function getWeeksInMonth(date = new Date()): WeekInMonth[] {
  return getWeeksOfMonth(date.getFullYear(), date.getMonth(), date);
}

export function getCurrentWeekOfMonth(date = new Date()): WeekInMonth {
  const weeks = getWeeksOfMonth(date.getFullYear(), date.getMonth(), date);
  return weeks.find((week) => week.isCurrent) ?? weeks[0];
}

export function getWeekLabel(week: WeekInMonth): string {
  return week.label;
}

function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function pad2(value: number): string {
  return String(value).padStart(2, '0');
}
