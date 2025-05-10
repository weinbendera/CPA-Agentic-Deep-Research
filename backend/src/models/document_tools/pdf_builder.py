import markdown2
import pdfkit
import re

def generate_pdf_from_markdown(markdown_text: str, output_filename: str = "report.pdf") -> None:
    # Convert Markdown → HTML
    html = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])

    # Add ID anchors to headings
    html = re.sub(
        r'<(h1|h2|h3)>(.*?)</\1>',
        lambda m: f'<{m.group(1)} id="{re.sub(r"\\W+", "-", m.group(2).strip().lower()).strip("-") if m.group(2).strip().lower() != "table of contents" else "table-of-contents"}">{m.group(2)}</{m.group(1)}>',
        html
    )

    # Insert page breaks
    html = re.sub(
        r'(<h2[^>]*>Table of Contents</h2>)',
        r'<div style="page-break-before: always;"></div>\1',
        html,
        flags=re.IGNORECASE
    )
    html = re.sub(
        r'(<h3[^>]*>\s*\d+\.\s+.*?</h3>)',
        r'<div style="page-break-before: always;"></div>\1',
        html
    )

    # Wrap in styled HTML
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page {{
      size: A4;
      margin: 40px;
      @bottom-right {{
        content: "Page " counter(page);
        font-size: 12px;
        color: #6b7280;
      }}
    }}
    body {{
      font-family: 'Segoe UI', sans-serif;
      font-size: 18px;
      color: #1f2937;
      background-color: #f9fbfd;
      line-height: 1.6;
    }}
    h1 {{
      font-size: 45px;
      color: #1e3a8a;
      background-color: #dbeafe;
      padding: 40px;
      border-radius: 14px;
      box-shadow: 0 6px 12px rgba(0,0,0,0.15);
      width: 90%;
      margin: 60px auto 30px auto;
      text-align: center;
    }}
    h2 {{
      font-size: 35px;
      color: #1e40af;
      border-bottom: 2px solid #cbd5e1;
      padding-bottom: 6px;
      margin-top: 50px;
    }}
    h3 {{
      font-size: 36px;
      color: #1d4ed8;
      margin-top: 40px;
    }}
    #table-of-contents + ul, 
    #table-of-contents + ol {{
      font-size: 20px;
      line-height: 1.8;
      margin-top: 20px;
      padding-left: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    th, td {{
      border: 1px solid #e2e8f0;
      padding: 14px;
      text-align: left;
    }}
    th {{
      background-color: #eff6ff;
    }}
    a {{
      color: #2563eb;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
{html}
</body>
</html>"""

    # Generate PDF
    pdfkit.from_string(full_html, output_filename, options={
        'disable-javascript': None,
        'enable-internal-links': None,
        'quiet': ''
    })
    print(f"PDF generated at: {output_filename}")