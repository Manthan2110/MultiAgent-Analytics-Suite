from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def export_report_to_pdf(report_text, plots, output_path="final_report.pdf"):
    styles = getSampleStyleSheet()
    story = []

    # Add Title
    story.append(Paragraph("<b>AI Multi-Agent Data Analyst Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # Add report text
    for line in report_text.split("\n"):
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 6))

    # Add Plots
    story.append(Paragraph("<b>Visualizations</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))

    for p in plots:
        try:
            story.append(Image(p, width=5*inch, height=3*inch))
            story.append(Spacer(1, 12))
        except:
            pass

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    doc.build(story)

    return output_path