export type PracticeDay = {
  date: string;
  submissions: number;
  attempted: number;
  solved: number;
  newly_solved: number;
};
export type ProfilePractice = {
  visible: true;
  timezone: string;
  year: number;
  years: number[];
  total_problems: number;
  attempted: number;
  solved: number;
  attempting: number;
  total_submissions: number;
  accepted_submissions: number;
  failed_submissions: number;
  by_difficulty: { difficulty: string; total: number; solved: number }[];
  calendar: {
    start: string;
    end: string;
    submissions: number;
    active_days: number;
    current_streak: number;
    max_streak: number;
    days: PracticeDay[];
  };
  recent_accepted: {
    problem_id: string;
    number: number;
    title: string;
    title_en: string;
    difficulty: string;
    accepted_at: string;
  }[];
};
export type ProfilePracticeResponse = ProfilePractice | { visible: false };
export type CalendarCell = { date: string; inRange: boolean; day: PracticeDay };
const DAY_MS = 86_400_000;
export function emptyPracticeDay(date: string): PracticeDay {
  return { date, submissions: 0, attempted: 0, solved: 0, newly_solved: 0 };
}
/** Date-only arithmetic stays in UTC, independent of the viewer's DST/offset. */
export function calendarCells(start: string, end: string, days: PracticeDay[]): CalendarCell[] {
  const first = new Date(`${start}T00:00:00Z`);
  const last = new Date(`${end}T00:00:00Z`);
  if (!Number.isFinite(first.getTime()) || !Number.isFinite(last.getTime()) || first > last) return [];
  const from = first.getTime() - first.getUTCDay() * DAY_MS;
  const through = last.getTime() + (6 - last.getUTCDay()) * DAY_MS;
  const byDate = new Map(days.map((day) => [day.date, day]));
  const result: CalendarCell[] = [];
  for (let value = from; value <= through; value += DAY_MS) {
    const date = new Date(value).toISOString().slice(0, 10);
    result.push({ date, inRange: date >= start && date <= end, day: byDate.get(date) || emptyPracticeDay(date) });
  }
  return result;
}
export function activityLevel(submissions: number): number {
  if (submissions <= 0) return 0;
  if (submissions === 1) return 1;
  if (submissions <= 3) return 2;
  if (submissions <= 6) return 3;
  return 4;
}
export function browserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}
export function currentYear(timezone: string): number {
  return Number(new Intl.DateTimeFormat("en", { timeZone: timezone, year: "numeric" }).format(new Date()));
}
