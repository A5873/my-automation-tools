# {{ title }}

**Date:** {{ date }}
**Author:** {{ author | default("Document Processor Suite") }}

---

## Executive Summary

{{ summary | default("This is an auto-generated document report.") }}

## Document Details

{% if documents %}
### Processed Documents

{% for doc in documents %}
#### {{ doc.filename }}

- **File Type:** {{ doc.type }}
- **Size:** {{ "%.2f"|format(doc.size / 1024) }} KB
- **Word Count:** {{ doc.word_count | default("N/A") }}
- **Last Modified:** {{ doc.modified }}

{% if doc.readability_score %}
- **Readability Score:** {{ "%.1f"|format(doc.readability_score) }}
{% endif %}

{% endfor %}
{% endif %}

## Statistics

{% if stats %}
- **Total Documents Processed:** {{ stats.processed }}
- **Successful Conversions:** {{ stats.converted }}
- **Reports Generated:** {{ stats.generated }}
- **Errors Encountered:** {{ stats.errors }}
{% endif %}

## Analysis Results

{{ analysis_content | default("No analysis data available.") }}

---

## Recommendations

{% if recommendations %}
{% for rec in recommendations %}
- {{ rec }}
{% endfor %}
{% else %}
- Review document formatting for consistency
- Consider batch processing for large document sets
- Implement regular document maintenance workflows
{% endif %}

---

*Report generated on {{ generation_date }} by Document Processor Suite*

## Footer

For more information about the Document Processor Suite, visit our documentation or contact support.

---

**Processing Configuration:**
- Input Formats: {{ input_formats | join(', ') | default('.docx, .pdf, .txt, .md, .html, .xlsx, .csv') }}
- Output Formats: {{ output_formats | join(', ') | default('.docx, .pdf, .txt, .md, .html, .xlsx, .csv') }}
- Features Used: {{ features_used | join(', ') | default('Document conversion, Analysis, Reporting') }}
