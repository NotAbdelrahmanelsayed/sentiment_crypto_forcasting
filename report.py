from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.platypus import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_metrics_figure(metrics, title, best_params=None):
    """Generate a matplotlib figure from metrics and optional best parameters."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    text_str = title + "\n\n"
    text_str += "\n".join(f"{key}: {round(value, 6)}" for key, value in metrics.items())
    if best_params:
        text_str += "\n\nBest Params:\n" + "\n".join(f"{k}: {v}" for k, v in best_params.items())
    ax.text(0.5, 0.5, text_str, transform=ax.transAxes, ha='center', va='center', fontsize=12)
    return fig

def create_metrics_table(metrics, title, best_params={}):
    # Update metrics with best_params if any
    metrics.update(best_params)

    # Prepare data for the table
    data = [["Metric", "Value"]]  # Table header
    metrics_rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}
    for metric, value in metrics_rounded.items():
        data.append([metric, value])

    # Create the table
    table = Table(data)

    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ]))

    return table, title

def add_elements_to_pdf(pdf_path, elements):
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    doc.build(elements)