/**
 * Relay Surf — document extraction (server-only).
 *
 * When a page is a downloadable document (PDF, DOCX, CSV, JSON, XML, TXT) the
 * engine can't feed raw bytes to the model — this layer converts them into
 * clean, bounded text with an honest note about what was lost. PDF/DOCX are
 * text-extracted heuristically (no binary parser is bundled); scanned
 * documents are reported as such rather than silently returning gibberish.
 */

export type DocumentType = "text" | "csv" | "json" | "xml" | "pdf" | "docx" | "doc";

export interface ExtractedDocument {
  type: DocumentType;
  text: string;
  truncated: boolean;
  pages?: number;
  note?: string;
}

const TEXT_EXTENSIONS = new Set(["txt", "md", "markdown", "csv", "json", "xml", "html", "htm"]);
const DOC_EXTENSIONS = new Set(["pdf", "docx", "doc"]);

/** Infer document type from the URL extension or response content-type. */
export function documentTypeOf(url: string, contentType?: string): DocumentType | null {
  let ext = "";
  try {
    ext = new URL(url).pathname.split("/").pop()?.split("?")[0].toLowerCase() ?? "";
  } catch {
    ext = url.toLowerCase();
  }
  const match = ext.match(/\.([a-z0-9]+)$/);
  const fileExt = match?.[1];

  if (fileExt && TEXT_EXTENSIONS.has(fileExt)) return fileExt as DocumentType;
  if (fileExt && DOC_EXTENSIONS.has(fileExt)) return fileExt as DocumentType;

  if (contentType) {
    const ct = contentType.toLowerCase();
    if (ct.includes("pdf")) return "pdf";
    if (ct.includes("msword") || ct.includes("wordprocessingml")) return "docx";
    if (ct.includes("csv") || ct.includes("spreadsheet")) return "csv";
    if (ct.includes("json")) return "json";
    if (ct.includes("xml")) return "xml";
    if (ct.includes("text/")) return "text";
  }
  return null;
}

/** Convert raw document content into bounded, model-safe text. */
export function extractDocument(
  url: string,
  rawText: string,
  contentType?: string,
  maxChars = 15_000,
): ExtractedDocument {
  const type = documentTypeOf(url, contentType) ?? "text";
  const budget = Math.min(Math.max(maxChars, 500), 60_000);

  switch (type) {
    case "csv": {
      const rows = rawText.split(/\r?\n/).filter((l) => l.trim());
      const text = rows
        .slice(0, 60)
        .map((r) => r.split(",").map((c) => c.replace(/^"|"$/g, "")).join(" | "))
        .join("\n");
      return { type, text, truncated: rows.length > 60, pages: rows.length };
    }

    case "json": {
      try {
        const pretty = JSON.stringify(JSON.parse(rawText), null, 2);
        return { type, text: pretty.slice(0, budget), truncated: pretty.length > budget };
      } catch {
        return { type, text: rawText.slice(0, budget), truncated: rawText.length > budget };
      }
    }

    case "xml": {
      const text = rawText
        .replace(/<\?xml[^>]*\?>/i, "")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      return { type, text: text.slice(0, budget), truncated: rawText.length > budget };
    }

    case "pdf": {
      // No bundled binary PDF parser — pull visible text out of the raw
      // content stream (works for simple, non-scanned PDFs).
      const text = rawText
        .replace(/\(([^()]{1,120})\)\s*Tj/gi, "$1 ")
        .replace(/<[^>]*>/g, " ")
        .replace(/[^\S\r\n]+/g, " ")
        .replace(/\s{2,}/g, " ")
        .trim();
      const hasContent = text.length > 100;
      return {
        type,
        text: hasContent ? text.slice(0, budget) : "",
        truncated: hasContent && text.length > budget,
        note: hasContent
          ? "PDF text extracted heuristically; scanned/image PDFs need a real OCR pipeline."
          : "This PDF appears to be scanned or image-based — no extractable text was found.",
      };
    }

    case "docx":
    case "doc": {
      const text = rawText
        .replace(/<w:t[^>]*>([^<]*)<\/w:t>/g, "$1 ")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      return {
        type,
        text: text ? text.slice(0, budget) : "",
        truncated: text.length > budget,
        note: text
          ? "Extracted from the document's XML heuristically; complex formatting may be lost."
          : "Document could not be parsed — the file may be binary-encoded.",
      };
    }

    case "text":
    default:
      return { type: "text", text: rawText.slice(0, budget), truncated: rawText.length > budget };
  }
}
