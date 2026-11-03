#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMuPDF 逐页文本提取脚本（替代 PDFBox，对 CAD/排版软件生成的 PDF 提取更稳）。
用法: python extract_pdf_pages.py <pdf_path>
输出: stdout 一行 JSON {"pages":[{"page":1,"text":"..."},...]}
失败: 输出 {"error":"..."} 并以非零退出码退出
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing pdf path"}), flush=True)
        return 2

    pdf_path = sys.argv[1]
    try:
        import pymupdf
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": "pymupdf not available: %s" % exc}), flush=True)
        return 3

    # Windows 下确保 stdout 为 UTF-8（Python 3.7+）
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        pages = []
        with pymupdf.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                text = page.get_text("text", sort=True)
                if text and text.strip():
                    pages.append({"page": page_index + 1, "text": text})
        print(json.dumps({"pages": pages}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        return 4


if __name__ == "__main__":
    sys.exit(main())
