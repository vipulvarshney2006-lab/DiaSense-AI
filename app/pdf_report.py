from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


def generate_report(filename,
                    patient,
                    risk,
                    health,
                    level):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(filename)

    story = []

    story.append(Paragraph(
        "<font size=22><b>🏥 DiaSense AI</b></font>",
        styles["Title"]
    ))

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            "<b>Early Diabetes Risk Assessment Report</b>",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1,20))

    data = [

        ["Patient Name", patient],

        ["Date",
         datetime.now().strftime("%d-%m-%Y")],

        ["Time",
         datetime.now().strftime("%I:%M %p")],

        ["Risk Score",
         f"{risk:.2f}%"],

        ["Health Score",
         f"{health:.2f}/100"],

        ["Risk Level",
         level]
    ]

    table = Table(data)

    table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2563EB")),

        ("TEXTCOLOR",(0,0),(-1,-1),colors.black),

        ("GRID",(0,0),(-1,-1),1,colors.grey),

        ("BACKGROUND",(0,0),(0,-1),colors.lightblue),

        ("BOTTOMPADDING",(0,0),(-1,-1),10),

    ]))

    story.append(table)

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            "<b>Recommendations</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
        """
        • Maintain a healthy diet.<br/>
        • Exercise regularly.<br/>
        • Monitor blood glucose levels.<br/>
        • Visit a healthcare professional if necessary.
        """,
        styles["BodyText"]
        )
    )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
        "<b>Disclaimer:</b> This report is generated using Machine Learning and is intended for educational purposes only.",
        styles["BodyText"]
        )
    )

    story.append(Spacer(1,30))

    story.append(
        Paragraph(
            "<b>Developed By:</b> Vipul Varshney",
            styles["BodyText"]
        )
    )

    doc.build(story)