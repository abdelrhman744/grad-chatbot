import {
  FileText,
  FileSpreadsheet,
  FileImage,
  FileJson,
  File as FileIcon,
  LucideIcon,
} from "lucide-react";
import { BadgeColor } from "@/components/ui/Badge";

interface FileTypeMeta {
  label: string;
  icon: LucideIcon;
  color: BadgeColor;
}

const FILE_TYPE_META: Record<string, FileTypeMeta> = {
  pdf: { label: "PDF", icon: FileText, color: "danger" },
  docx: { label: "Word", icon: FileText, color: "primary" },
  doc: { label: "Word", icon: FileText, color: "primary" },
  txt: { label: "Text", icon: FileText, color: "muted" },
  markdown: { label: "Markdown", icon: FileText, color: "muted" },
  json: { label: "JSON", icon: FileJson, color: "warning" },
  image: { label: "Image", icon: FileImage, color: "warning" },
  excel: { label: "Excel", icon: FileSpreadsheet, color: "success" },
  unknown: { label: "File", icon: FileIcon, color: "muted" },
};

export function getFileTypeMeta(fileType: string): FileTypeMeta {
  return FILE_TYPE_META[fileType] ?? FILE_TYPE_META.unknown;
}
