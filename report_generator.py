from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from datetime import datetime
import os

def generate_pdf_report(patient_data, filename="medical_report.pdf"):
    """
    Generates a professional medical AI report in PDF format with an app logo.
    """
    # Create the PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    story = []
    styles = getSampleStyleSheet()

    # --- CUSTOM STYLES ---
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#1e293b"),
        alignment=1, # Centered
        spaceAfter=2,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor("#3b82f6"),
        alignment=1, # Centered
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#1e293b"),
        borderPadding=5,
        spaceBefore=15,
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )

    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        fontName='Helvetica-Bold'
    )

    value_style = ParagraphStyle(
        'ValueStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#1e293b"),
        fontName='Helvetica'
    )

    # --- HEADER SECTION (Centered) ---
    story.append(Paragraph("Diabetes Prediction Report", title_style))
    story.append(Paragraph("HealthGuard AI", subtitle_style))

    story.append(Paragraph(f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                          ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=1, fontSize=9, textColor=colors.grey)))
    story.append(Spacer(1, 0.3 * inch))

    # --- PATIENT INFORMATION SECTION ---
    story.append(Paragraph("Patient Information", section_header_style))
    
    patient_info = [
        [Paragraph("Patient Name:", label_style), Paragraph(str(patient_data.get('patient_name', 'N/A')), value_style),
         Paragraph("Age:", label_style), Paragraph(str(patient_data.get('age', 'N/A')), value_style)],
        [Paragraph("Glucose Level:", label_style), Paragraph(f"{patient_data.get('glucose', 'N/A')} mg/dL", value_style),
         Paragraph("Insulin Level:", label_style), Paragraph(f"{patient_data.get('insulin', 'N/A')} mu U/ml", value_style)],
        [Paragraph("BMI:", label_style), Paragraph(str(patient_data.get('bmi', 'N/A')), value_style),
         Paragraph("Blood Pressure:", label_style), Paragraph(f"{patient_data.get('blood_pressure', 'N/A')} mmHg", value_style)]
    ]

    patient_table = Table(patient_info, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    patient_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.1, colors.lightgrey),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- PREDICTION RESULT SECTION ---
    story.append(Paragraph("Prediction Result", section_header_style))
    
    is_diabetic = patient_data.get('prediction_result') == 1
    res_color = colors.HexColor("#ef4444") if is_diabetic else colors.HexColor("#10b981")
    res_label = "POSITIVE (Risk Detected)" if is_diabetic else "NEGATIVE (No Significant Risk)"
    risk_level = "High" if is_diabetic else "Low"

    prediction_info = [
        [Paragraph("Diagnostic Result:", label_style), Paragraph(res_label, ParagraphStyle('ResLabel', parent=value_style, textColor=res_color, fontName='Helvetica-Bold'))],
        [Paragraph("Calculated Risk:", label_style), Paragraph(risk_level, value_style)]
    ]

    prediction_table = Table(prediction_info, colWidths=[1.5*inch, 4.5*inch])
    prediction_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
    ]))
    story.append(prediction_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- SUGGESTIONS SECTION ---
    story.append(Paragraph("Clinical Suggestions", section_header_style))
    suggestion_text = patient_data.get('suggestion', "No specific suggestions available.")
    
    # Convert suggestions into bullet points if comma separated
    suggestions = [s.strip() for s in suggestion_text.split(",") if s.strip()]
    
    bullet_style = ParagraphStyle(       
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=5
    )

    for item in suggestions:
        story.append(Paragraph(f"• {item.capitalize()}", bullet_style))
    
    story.append(Spacer(1, 0.3 * inch))

    # Build PDF
    doc.build(story)
    return filename
