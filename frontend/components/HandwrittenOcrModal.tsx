"use client";

import { useCallback, useRef, useState } from "react";
import { Copy, Check, X, ScanText, Loader2, AlertCircle, UploadCloud } from "lucide-react";
import { ocrHandwritten, HandwrittenLanguage } from "@/services/api";
import Card from "@/components/ui/Card";

// Kept local, matching the existing per-component convention (see
// AnswerBox.tsx / ChatBox.tsx) rather than introducing a shared util for
// a single regex.
const isArabicText = (text: string) => /[؀-ۿ]/.test(text);

const ACCEPTED = ".png,.jpg,.jpeg,.bmp,.tiff,.webp";

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function HandwrittenOcrModal({ open, onClose }: Props) {
  const [language, setLanguage] = useState<HandwrittenLanguage>("en");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pickFile = useCallback((f: File | undefined) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setCopied(false);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
  }, []);

  const reset = useCallback(() => {
    setFile(null);
    setResult(null);
    setError(null);
    setCopied(false);
    setLoading(false);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  }, []);

  const handleClose = useCallback(() => {
    reset();
    onClose();
  }, [reset, onClose]);

  const runOcr = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await ocrHandwritten(file, language);
      setResult(res.text);
    } catch (e: any) {
      setError(e?.message || "Handwritten OCR failed.");
    } finally {
      setLoading(false);
    }
  }, [file, language]);

  const copyResult = useCallback(async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError("Could not copy to clipboard.");
    }
  }, [result]);

  if (!open) return null;

  const resultRtl = result ? isArabicText(result) : language === "ar";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        onClick={handleClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
      />

      <Card
        padding="lg"
        className="relative z-10 w-full max-w-lg max-h-[90vh] overflow-y-auto animate-scale-in"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-primary flex items-center justify-center flex-shrink-0">
              <ScanText className="w-4 h-4 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-ink">Handwritten OCR</h2>
              <p className="text-xs text-muted">Free, local — Arabic &amp; English</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-lg text-muted hover:text-ink hover:bg-surface-2 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Language selector */}
        <div className="flex gap-2 mb-4">
          {(["en", "ar"] as HandwrittenLanguage[]).map((lang) => (
            <button
              key={lang}
              onClick={() => setLanguage(lang)}
              className={`flex-1 h-9 rounded-xl text-sm font-medium transition-all border
                ${
                  language === lang
                    ? "bg-gradient-primary text-white border-transparent shadow-glow"
                    : "bg-surface-2 text-muted border-border hover:text-ink hover:border-border-light"
                }`}
            >
              {lang === "en" ? "English" : "العربية"}
            </button>
          ))}
        </div>

        {/* Upload zone */}
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pickFile(e.dataTransfer.files?.[0]);
          }}
          className={`
            cursor-pointer rounded-2xl border-2 border-dashed p-4
            flex items-center gap-3 select-none transition-all duration-200
            ${
              dragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40 hover:bg-surface-2/60"
            }
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            onChange={(e) => {
              pickFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Selected handwriting"
              className="w-14 h-14 rounded-lg object-cover border border-border flex-shrink-0"
            />
          ) : (
            <div className="w-10 h-10 rounded-full bg-surface-2 flex items-center justify-center flex-shrink-0">
              <UploadCloud className="w-4 h-4 text-muted" strokeWidth={1.75} />
            </div>
          )}
          <div className="min-w-0">
            <p className="text-sm font-medium text-ink truncate">
              {file ? file.name : "Upload a handwritten image"}
            </p>
            <p className="text-xs text-muted mt-0.5">
              {file ? "Click to choose a different image" : "A cropped text line works best — PNG, JPG, BMP, TIFF, WEBP"}
            </p>
          </div>
        </div>

        {/* Run button */}
        <button
          onClick={runOcr}
          disabled={!file || loading}
          className="w-full h-10 mt-4 rounded-xl flex items-center justify-center gap-2
            text-sm font-medium text-white bg-gradient-primary
            disabled:opacity-40 disabled:cursor-not-allowed
            hover:shadow-glow transition-all"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running OCR… (first run may download the model)
            </>
          ) : (
            "Run OCR"
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-4 flex items-start gap-2 text-sm text-danger animate-fade-in">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Result */}
        {result !== null && (
          <div className="mt-4 animate-fade-in">
            <div className="flex items-center justify-between mb-1.5">
              <p className="text-xs font-mono text-muted uppercase tracking-widest">
                Extracted text
              </p>
              <button
                onClick={copyResult}
                disabled={!result}
                className="flex items-center gap-1 text-xs font-medium text-muted hover:text-ink transition-colors disabled:opacity-40"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-success" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy
                  </>
                )}
              </button>
            </div>
            <div
              dir={resultRtl ? "rtl" : "ltr"}
              className={`
                rounded-xl border border-border bg-surface-2 p-3.5
                text-sm leading-relaxed whitespace-pre-wrap min-h-[3rem]
                ${resultRtl ? "text-right font-arabic" : "text-left"}
              `}
            >
              {result || (
                <span className="text-muted">
                  No text was recognized in this image.
                </span>
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
