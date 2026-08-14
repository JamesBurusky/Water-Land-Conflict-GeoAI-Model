// Trend/seasonal charts aggregate from the FILTERED CONFLICT RECORDS
// list (useConflicts()), not the monthly panel -- this is deliberate.
// The panel endpoint only supports county/subcounty/year filters; the
// conflicts endpoint supports those PLUS topic and free-text search.
// Aggregating from conflicts means every chart on the dashboard
// actually responds to every filter, matching "all dashboards should
// be dynamic on the filters, searches and selections made."

import type { ConflictRecord } from "../types/api";

export interface YearlyPoint {
  year: number;
  count: number;
  avgSeverity: number;
}

function parseYear(dateStr: string | undefined): number | null {
  if (!dateStr) return null;
  const d = new Date(dateStr);
  return Number.isNaN(d.getTime()) ? null : d.getFullYear();
}

function parseMonth(dateStr: string | undefined): number | null {
  if (!dateStr) return null;
  const d = new Date(dateStr);
  return Number.isNaN(d.getTime()) ? null : d.getMonth() + 1; // JS months are 0-indexed
}

export function aggregateByYear(records: ConflictRecord[]): YearlyPoint[] {
  const byYear = new Map<number, { count: number; severitySum: number; severityN: number }>();
  for (const r of records) {
    const year = parseYear(r.Date_Start);
    if (year === null) continue;
    const bucket = byYear.get(year) ?? { count: 0, severitySum: 0, severityN: 0 };
    bucket.count += 1;
    if (typeof r.severity_score === "number") {
      bucket.severitySum += r.severity_score;
      bucket.severityN += 1;
    }
    byYear.set(year, bucket);
  }
  return Array.from(byYear.entries())
    .sort(([a], [b]) => a - b)
    .map(([year, b]) => ({
      year,
      count: b.count,
      avgSeverity: b.severityN > 0 ? b.severitySum / b.severityN : 0,
    }));
}

export interface MonthlyPoint {
  month: number;
  monthLabel: string;
  count: number;
}

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function aggregateByMonth(records: ConflictRecord[]): MonthlyPoint[] {
  const byMonth = new Map<number, number>();
  for (const r of records) {
    const month = parseMonth(r.Date_Start);
    if (month === null) continue;
    byMonth.set(month, (byMonth.get(month) ?? 0) + 1);
  }
  return Array.from({ length: 12 }, (_, i) => i + 1).map((month) => ({
    month,
    monthLabel: MONTH_LABELS[month - 1],
    count: byMonth.get(month) ?? 0,
  }));
}
