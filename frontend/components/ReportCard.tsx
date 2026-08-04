"use client";

import { FileText, Download, CheckCircle2 } from "lucide-react";
import { ReportInfo } from "@/services/api";

interface Props {
  report: ReportInfo;
  isArabic?: boolean;
}

export default function ReportCard({ report, isArabic }: Props) {
  const url = report.download_url || report.proxy_download_path;

  return (
    <div
      dir={isArabic ? "rtl" : "ltr"}
      className="mt-2 w-full max-w-sm rounded-xl border border-border bg-surface shadow-card overflow-hidden animate-fade-up"
    >
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border bg-primary/10">
        <div className="w-9 h-9 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
          <FileText className="w-4 h-4 text-primary-light" />
        </div>
        <span className="text-sm font-medium text-ink truncate" title={report.object_name}>
          {report.object_name}
        </span>
      </div>

      <div className="px-4 py-3 space-y-2.5">
        <div className="flex items-center gap-1.5 text-xs font-medium text-success">
          <CheckCircle2 className="w-3.5 h-3.5" />
          {isArabic ? "تم إنشاء التقرير بنجاح" : "Report generated successfully"}
        </div>

        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          download
          className="flex items-center justify-center gap-2 w-full rounded-lg bg-gradient-primary text-white
            text-sm font-medium py-2 hover:shadow-glow transition-all"
        >
          <Download className="w-3.5 h-3.5" />
          {isArabic ? "تحميل التقرير" : "Download Report"}
        </a>
      </div>
    </div>
  );
}
