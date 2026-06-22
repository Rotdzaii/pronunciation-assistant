import { Platform } from 'react-native';

export type CsvExportResult = {
  ok: boolean;
  message: string;
};

export function exportCsv(filename: string, headers: string[], rows: Array<Array<string | number | null | undefined>>): CsvExportResult {
  const csv = toCsv(headers, rows);

  if (Platform.OS !== 'web' || typeof window === 'undefined' || typeof document === 'undefined') {
    return {
      ok: false,
      message: 'Xuất CSV hiện chỉ hỗ trợ trên phiên bản web.',
    };
  }

  const blob = new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8;' });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.URL.revokeObjectURL(url);

  return {
    ok: true,
    message: `Đã tạo file ${filename}.`,
  };
}

function toCsv(headers: string[], rows: Array<Array<string | number | null | undefined>>): string {
  return [headers, ...rows]
    .map((row) => row.map((cell) => escapeCsvCell(cell === null || cell === undefined ? '' : String(cell))).join(','))
    .join('\r\n');
}

function escapeCsvCell(value: string): string {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}
